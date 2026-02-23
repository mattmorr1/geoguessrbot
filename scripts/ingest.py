import ee
import torch
import reverse_geocoder as rg
from datasets import load_dataset
from pymilvus import MilvusClient, DataType
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from torch.utils.data import DataLoader
import os
import s2sphere
import torch.nn as nn
import torch.nn.functional as F
import shutil
import zipfile
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm
from queue import Queue
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download, HfApi

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP", use_fast=True)
vision_model = CLIPModel.from_pretrained("geolocal/StreetCLIP").to(device)

ee.Authenticate()
ee.Initialize(project=os.getenv('GEE_PROJECT_ID'))

from pymilvus import MilvusClient, DataType

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
if "geoguessr" not in client.list_databases():
    client.create_database("geoguessr")

client = MilvusClient(uri="http://localhost:19530", token="root:Milvus", db_name="geoguessr")

schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="streetclip_vec", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="alphaearth_vec", datatype=DataType.FLOAT_VECTOR, dim=64)
schema.add_field(field_name="hyp_tangent_vec", datatype=DataType.FLOAT_VECTOR, dim=128)
schema.add_field(field_name="gps", datatype=DataType.JSON)
schema.add_field(field_name="s2sphere_boundary", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="country_code", datatype=DataType.VARCHAR, max_length=8)
schema.add_field(field_name="continent", datatype=DataType.VARCHAR, max_length=32)
schema.add_field(field_name="s2_token_l10", datatype=DataType.VARCHAR, max_length=32)

index_params = client.prepare_index_params()

index_params.add_index(
    field_name="streetclip_vec",
    metric_type="COSINE",
    index_type="HNSW",
    params={"M": 16, "efConstruction": 500}
)

index_params.add_index(
    field_name="alphaearth_vec",
    metric_type="COSINE",
    index_type="HNSW",
    params={"M": 8, "efConstruction": 200}
)

# L2 on tangent-space projections approximates Poincare distance
# for nearby points in the ball. can't use native hyperbolic metric
# in Milvus, so this is the proxy for hierarchical similarity search.
index_params.add_index(
    field_name="hyp_tangent_vec",
    metric_type="L2",
    index_type="HNSW",
    params={"M": 16, "efConstruction": 200}
)

# scalar indexes for constrained search
index_params.add_index(field_name="country_code", index_type="TRIE")
index_params.add_index(field_name="s2_token_l10", index_type="TRIE")

if client.has_collection(collection_name="world_locations"):
    print("Collection exists in 'geoguessr' db -> Replacing")
    client.drop_collection(collection_name="world_locations")

client.create_collection(
    collection_name="world_locations",
    schema=schema,
    index_params=index_params
)

CONTINENT_NAMES = ['Africa', 'Asia', 'Europe', 'North America', 'South America', 'Oceania', 'Antarctica']

CC_TO_CONTINENT = {}
for cc, cont in [
    *[(c, 'Africa') for c in 'DZ AO BJ BW BF BI CM CV CF TD KM CG CD CI DJ EG GQ ER SZ ET GA GM GH GN GW KE LS LR LY MG MW ML MR MU MA MZ NA NE NG RW ST SN SC SL SO ZA SS SD TZ TG TN UG ZM ZW RE YT'.split()],
    *[(c, 'Asia') for c in 'AF AM AZ BH BD BT BN KH CN CY GE IN ID IR IQ IL JP JO KZ KW KG LA LB MY MV MN MM NP KP OM PK PS PH QA SA SG KR LK SY TW TJ TH TL TR TM AE UZ VN YE HK MO'.split()],
    *[(c, 'Europe') for c in 'AL AD AT BY BE BA BG HR CZ DK EE FI FR DE GR HU IS IE IT XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH UA GB VA FO GI JE GG IM AX'.split()],
    *[(c, 'North America') for c in 'AG BS BB BZ CA CR CU DM DO SV GD GT HT HN JM MX NI PA KN LC VC TT US PR VI GL BM AW CW SX MQ GP TC KY MS AI'.split()],
    *[(c, 'South America') for c in 'AR BO BR CL CO EC GY PY PE SR UY VE GF FK'.split()],
    *[(c, 'Oceania') for c in 'AU FJ KI MH FM NR NZ PW PG WS SB TO TV VU NC PF GU AS CK NU TK WF MP'.split()],
    *[(c, 'Antarctica') for c in ['AQ']],
]:
    CC_TO_CONTINENT[cc] = cont

def get_s2_token(lat, lon, level=12):
    p = s2sphere.LatLng.from_degrees(lat, lon)
    return s2sphere.CellId.from_lat_lng(p).parent(level).to_token()

def get_s2_token_l10(lat, lon):
    p = s2sphere.LatLng.from_degrees(lat, lon)
    return s2sphere.CellId.from_lat_lng(p).parent(10).to_token()

def get_country_info(coords_list):
    results = rg.search(coords_list)
    out = []
    for r in results:
        cc = r['cc']
        out.append({
            'country_code': cc,
            'continent': CC_TO_CONTINENT.get(cc, 'Unknown'),
        })
    return out

def get_streetclip_embedding(images):
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        features = vision_model.get_image_features(**inputs)
    return features.cpu().numpy()

def alpha_earth_features(coords_list):
    points = [ee.Feature(ee.Geometry.Point([lon, lat])) for lat, lon in coords_list]
    fc = ee.FeatureCollection(points)
    ae_img = ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL").first()
    sampled = ae_img.reduceRegions(collection=fc, reducer=ee.Reducer.first(), scale=10)
    results = []
    for feat in sampled.getInfo()['features']:
        props = feat['properties']
        vec = [float(props.get(f'A{i:02d}', 0.0) or 0.0) for i in range(64)]
        results.append(vec)
    return results

import os
import time
import shutil
import zipfile
import pandas as pd
from PIL import Image
from tqdm import tqdm
from queue import Queue
from threading import Thread, Lock
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi, hf_hub_download

# --- Configuration ---
TEMP_DIR = "./temp_processing"
BATCH_SIZE = 32 
DB_FLUSH_SIZE = 320
MAX_BUFFERED_ZIPS = 2
MAX_INSERT_RETRIES = 3
HF_TOKEN = os.getenv("HF_TOKEN")

os.makedirs(TEMP_DIR, exist_ok=True)
download_queue = Queue(maxsize=MAX_BUFFERED_ZIPS)

# --- 1. Helper Functions ---

def download_and_extract(zip_name):
    """Downloads a zip from HF and extracts it to a temp folder."""
    try:
        zip_path = hf_hub_download(
            repo_id="osv5m/osv5m",
            filename=f"images/train/{zip_name}",
            repo_type='dataset',
            local_dir=TEMP_DIR,
            token=HF_TOKEN
        )
        extract_to = os.path.join(TEMP_DIR, "train_" + zip_name.replace(".zip", ""))
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        
        os.remove(zip_path)
        return extract_to
    except Exception as e:
        print(f"Failed to download/extract {zip_name}: {e}")
        return None

def producer_task(zip_names):
    for name in zip_names:
        path = download_and_extract(name)
        if path:
            download_queue.put(path)
    download_queue.put(None)

def get_all_images(directory):
    """Recursively finds all .jpg files."""
    image_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.jpg'):
                image_paths.append(os.path.join(root, f))
    return image_paths

def flush_to_milvus(records):
    """Insert records into Milvus with retry logic."""
    for attempt in range(1, MAX_INSERT_RETRIES + 1):
        try:
            client.insert(collection_name="world_locations", data=records)
            return True
        except Exception as e:
            print(f"  [!] Milvus insert failed (attempt {attempt}/{MAX_INSERT_RETRIES}): {e}")
            if attempt < MAX_INSERT_RETRIES:
                time.sleep(2 ** attempt)
    print(f"  [!!] DROPPING {len(records)} records after {MAX_INSERT_RETRIES} failed attempts")
    return False

# --- 2. Metadata Setup ---

def ensure_metadata():
    csv_path = "./datasets/osv5m/train.csv"
    if not os.path.exists(csv_path):
        hf_hub_download(repo_id="osv5m/osv5m", filename="train.csv", 
                        repo_type='dataset', local_dir="./datasets/osv5m", token=HF_TOKEN)
    
    df = pd.read_csv(csv_path)
    df['id'] = df['id'].astype(str)
    return df.set_index('id')

metadata_lookup = ensure_metadata()

# --- 3. Processing Logic ---

def process_batch_parallel(batch_images, batch_coords):
    with ThreadPoolExecutor(max_workers=3) as ex:
        clip_future = ex.submit(get_streetclip_embedding, batch_images)
        alpha_future = ex.submit(alpha_earth_features, batch_coords)
        geo_future = ex.submit(get_country_info, batch_coords)
        
        clip_vecs = clip_future.result()
        alpha_vecs = alpha_future.result()
        geo_info = geo_future.result()
    
    records = []
    for i in range(len(batch_images)):
        lat, lon = batch_coords[i]
        records.append({
            "streetclip_vec": clip_vecs[i].tolist(),
            "alphaearth_vec": alpha_vecs[i],
            "hyp_tangent_vec": [0.0] * 128,  # placeholder; backfilled post-training
            "gps": {"lat": float(lat), "lon": float(lon)},
            "s2sphere_boundary": get_s2_token(lat, lon),
            "country_code": geo_info[i]['country_code'],
            "continent": geo_info[i]['continent'],
            "s2_token_l10": get_s2_token_l10(lat, lon),
        })
    return records

# --- 4. Main Pipeline ---

def run_pipeline():
    zip_names = [f"{str(i).zfill(2)}.zip" for i in range(5)] 
    
    producer = Thread(target=producer_task, args=(zip_names,), daemon=True)
    producer.start()
    
    db, total_processed, total_failed = [], 0, 0
    
    while True:
        folder_path = download_queue.get()
        if folder_path is None: break
        
        image_paths = get_all_images(folder_path)
        print(f"\n>>> Processing {len(image_paths)} images from {os.path.basename(folder_path)}")
        
        batch_images, batch_coords, skipped = [], [], 0
        
        for img_path in tqdm(image_paths, desc="Batching"):
            img_id = os.path.basename(img_path).split('.')[0]
            
            if img_id not in metadata_lookup.index:
                skipped += 1
                continue
            
            try:
                row = metadata_lookup.loc[img_id]
                if isinstance(row, pd.DataFrame): row = row.iloc[0]
                
                img = Image.open(img_path).convert('RGB')
                batch_images.append(img)
                batch_coords.append((row['latitude'], row['longitude']))
            except Exception as e:
                print(f"  [!] Failed to load image {img_id}: {e}")
                continue
            
            if len(batch_images) >= BATCH_SIZE:
                # Process batch — separate try so insert errors don't kill the batch
                try:
                    db.extend(process_batch_parallel(batch_images, batch_coords))
                    total_processed += len(batch_images)
                except Exception as e:
                    total_failed += len(batch_images)
                    print(f"  [!] Batch processing failed ({len(batch_images)} images lost): {e}")
                batch_images, batch_coords = [], []
                
                # Flush to Milvus
                if len(db) >= DB_FLUSH_SIZE:
                    flush_to_milvus(db)
                    db = []
        
        # Flush remaining images from current folder
        if batch_images:
            try:
                db.extend(process_batch_parallel(batch_images, batch_coords))
                total_processed += len(batch_images)
            except Exception as e:
                total_failed += len(batch_images)
                print(f"  [!] Final batch processing failed ({len(batch_images)} images lost): {e}")

        # Cleanup folder
        shutil.rmtree(folder_path)
        print(f"  Done. Total so far: {total_processed}, Skipped: {skipped}, Failed: {total_failed}")
        download_queue.task_done()

    # Final database flush
    if db:
        flush_to_milvus(db)
    
    print(f"\nPipeline Complete! Processed: {total_processed}, Failed: {total_failed}")

run_pipeline()