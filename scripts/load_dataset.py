import os
import math
import time
import json
import queue
import zipfile
import pandas as pd
import reverse_geocoder as rg
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

from pymilvus import MilvusClient, DataType

load_dotenv()

os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
# High values spawn many Windows child processes that re-import this module; each import must stay cheap.
os.environ.setdefault("HF_XET_NUM_CONCURRENT_RANGE_GETS", "8")
# Workaround: A dependency (often ee or a specific python version config) sets an invalid SSL_CERT_FILE
os.environ.pop("SSL_CERT_FILE", None)

# Filled by _init_runtime() — must not load torch/transformers/ee at import time (HF Hub / mp spawn re-imports this file on Windows).
device = None
BATCH_SIZE = 32
processor = None
vision_model = None

DB_FLUSH_SIZE = 2048
MAX_BUFFERED_ZIPS = 2
MAX_INSERT_RETRIES = 3
HF_TOKEN = os.getenv("HF_TOKEN")
GEE_PROJECT = os.getenv("GEE_PROJECT_ID")
CHECKPOINT_FILE = "pipeline_checkpoint.json"

# Set True in _init_runtime() when GEE Alpha Earth is used (affects time estimates).
ALPHA_EARTH_ON = True


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes", "on")


def _train_sample_fraction() -> float:
    raw = (os.getenv("TRAIN_SAMPLE_FRACTION") or "1").strip()
    try:
        x = float(raw)
    except ValueError:
        print(f"[!] Invalid TRAIN_SAMPLE_FRACTION={raw!r}; using 1.0")
        return 1.0
    if x <= 0 or x > 1:
        print(f"[!] TRAIN_SAMPLE_FRACTION must be in (0, 1]; got {x}; using 1.0")
        return 1.0
    return x


def _checkpoint_path_for_sample_fraction(frac: float) -> str:
    custom = (os.getenv("PIPELINE_CHECKPOINT_FILE") or "").strip()
    if custom:
        return custom
    if frac < 1.0:
        pct = int(round(frac * 100))
        return f"pipeline_checkpoint_{pct}pct.json"
    return CHECKPOINT_FILE


def _print_ingest_estimate(num_ids: int, batch_size: int, *, alpha_earth: bool) -> None:
    if num_ids <= 0 or batch_size <= 0:
        return
    batches = math.ceil(num_ids / batch_size)
    if alpha_earth:
        print(
            f">>> Ingest estimate: {num_ids:,} train IDs -> ~{batches:,} batches "
            f"(batch_size={batch_size}; wall time dominated by Earth Engine getInfo per batch)"
        )
        for label, sec in (("low", 30), ("mid", 60), ("high", 90)):
            hrs = batches * sec / 3600
            print(f"    ~{hrs:.1f} h if ~{sec} s/batch ({label})")
    else:
        print(
            f">>> Ingest estimate: {num_ids:,} train IDs -> ~{batches:,} CLIP batches "
            f"(batch_size={batch_size}; SKIP_ALPHA_EARTH: GPU + I/O bound, not GEE)"
        )
        for label, sec in (("fast", 0.25), ("typical", 1.0), ("slow disk", 4.0)):
            hrs = batches * sec / 3600
            print(f"    ~{hrs:.2f} h if ~{sec} s/batch ({label})")


# --- Continent Mapping ---
CC_TO_CONTINENT = {}
for cc, cont in [
    *[(c, 'Africa') for c in 'DZ AO BJ BW BF BI CM CV CF TD KM CG CD CI DJ EG GQ ER SZ ET GA GM GH GN GW KE LS LR LY MG MW ML MR MU MA MZ NA NE NG RW ST SN SC SL SO ZA SS SD TZ TG TN UG ZM ZW RE YT'.split()],
    *[(c, 'Asia') for c in 'AF AM AZ BH BD BT BN KH CN CY GE IN ID IR IQ IL JP JO KZ KW KG LA LB MY MV MN MM NP KP OM PK PS PH QA SA SG KR LK SY TW TJ TH TL TR TM AE UZ VN YE HK MO'.split()],
    *[(c, 'Europe') for c in 'AL AD AT BY BE BA BG HR CZ DK EE FI FR DE GR HU IS IE IT XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH UA GB VA FO GI JE GG IM AX'.split()],
    *[(c, 'North America') for c in 'AG BS BB BZ CA CR CU DM DO SV GD GT HT HN JM MX NI PA KN LC VC TT US PR VI GL BM AW CW SX MQ GP TC KY MS AI'.split()],
    *[(c, 'South America') for c in 'AR BO BR CL CO EC GY PY PE SR UY VE GF FK'.split()],
    *[(c, 'Oceania') for c in 'AU FJ KI MH FM NR NZ PW PG WS SB TO TV VU NC PF GU AS CK NU TK WF MP'.split()],
    *[(c, 'Antarctica') for c in ['AQ']],
]: CC_TO_CONTINENT[cc] = cont


def _init_runtime():
    """Torch, CLIP, optional Earth Engine — only from the main pipeline (not on mp import)."""
    global device, BATCH_SIZE, processor, vision_model
    global get_s2_token, get_streetclip_embedding, alpha_earth_features
    global ALPHA_EARTH_ON

    import torch
    import transformers
    from transformers import CLIPProcessor, CLIPModel

    from geo_utils import get_s2_token as _gs2
    from geo_utils import get_streetclip_embedding as _gse

    get_s2_token = _gs2
    get_streetclip_embedding = _gse

    ALPHA_EARTH_ON = not _env_truthy("SKIP_ALPHA_EARTH")
    if ALPHA_EARTH_ON:
        from geo_utils import alpha_earth_features as _aef

        alpha_earth_features = _aef
    else:

        def alpha_earth_features(coords_list, max_retries=5):
            return [[0.0] * 64 for _ in coords_list]

        print(">>> SKIP_ALPHA_EARTH=1: alphaearth_vec = zeros (no GEE; much faster ingest).")

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    bs_env = (os.getenv("CLIP_BATCH_SIZE") or os.getenv("PIPELINE_BATCH_SIZE") or "").strip()
    if bs_env:
        try:
            BATCH_SIZE = max(1, int(bs_env))
        except ValueError:
            BATCH_SIZE = 128 if device == "cuda" else 32
            print(f"[!] Invalid CLIP_BATCH_SIZE/PIPELINE_BATCH_SIZE={bs_env!r}; using default {BATCH_SIZE}")
    else:
        BATCH_SIZE = 128 if device == "cuda" else 32
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Running on: {device} (batch_size={BATCH_SIZE})")

    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP", use_fast=True)
    transformers.logging.set_verbosity_error()
    vision_model = CLIPModel.from_pretrained("geolocal/StreetCLIP").to(device).half().eval()
    transformers.logging.set_verbosity_warning()

    if ALPHA_EARTH_ON:
        import ee

        ee.Initialize(project=GEE_PROJECT)


# Stubs so static analysis / accidental early use fail clearly (replaced in _init_runtime).
def get_s2_token(*args, **kwargs):
    raise RuntimeError("pipeline not initialized; call _init_runtime() from run_pipeline first")


def get_streetclip_embedding(*args, **kwargs):
    raise RuntimeError("pipeline not initialized; call _init_runtime() from run_pipeline first")


def alpha_earth_features(*args, **kwargs):
    raise RuntimeError("pipeline not initialized; call _init_runtime() from run_pipeline first")


# --- Milvus Initialization ---
def init_milvus():
    client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")
    if "geoguessr" not in client.list_databases():
        client.create_database("geoguessr")

    client = MilvusClient(uri="http://localhost:19530", token="root:Milvus", db_name="geoguessr")

    if client.has_collection("world_locations"):
        print("Using existing collection world_locations (no drop).")
        return client

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="streetclip_vec", datatype=DataType.FLOAT_VECTOR, dim=768)
    schema.add_field(field_name="alphaearth_vec", datatype=DataType.FLOAT_VECTOR, dim=64)
    schema.add_field(field_name="hyp_tangent_vec", datatype=DataType.FLOAT_VECTOR, dim=128)
    schema.add_field(field_name="country_code", datatype=DataType.VARCHAR, max_length=10)
    schema.add_field(field_name="continent", datatype=DataType.VARCHAR, max_length=50)
    schema.add_field(field_name="s2_token_l10", datatype=DataType.VARCHAR, max_length=20)
    schema.add_field(field_name="s2sphere_boundary", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="gps", datatype=DataType.JSON)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="streetclip_vec", metric_type="COSINE", index_type="HNSW", params={"M": 16, "efConstruction": 200})
    index_params.add_index(field_name="alphaearth_vec", metric_type="COSINE", index_type="HNSW", params={"M": 16, "efConstruction": 200})
    # FIX: Every vector field MUST have an index before collection creation
    index_params.add_index(field_name="hyp_tangent_vec", metric_type="L2", index_type="HNSW", params={"M": 16, "efConstruction": 200})
    index_params.add_index(field_name="country_code", index_type="INVERTED")
    index_params.add_index(field_name="s2_token_l10", index_type="INVERTED")

    client.create_collection(collection_name="world_locations", schema=schema, index_params=index_params)
    return client

# --- Pipeline Components ---
download_queue = queue.Queue(maxsize=MAX_BUFFERED_ZIPS)
upload_queue = queue.Queue(maxsize=10)

class CheckpointManager:
    def __init__(self, filepath=CHECKPOINT_FILE):
        self.filepath = filepath
        self.processed_zips = self._load()
    def _load(self):
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f: return set(json.load(f))
        return set()
    def save(self, zip_name):
        self.processed_zips.add(zip_name)
        with open(self.filepath, "w") as f: json.dump(list(self.processed_zips), f)
    def is_processed(self, zip_name):
        return zip_name in self.processed_zips

    def clear(self):
        self.processed_zips = set()
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

def _milvus_row_count(client):
    try:
        return int(client.get_collection_stats("world_locations").get("row_count", 0))
    except Exception:
        return 0

def milvus_worker(milvus_client):
    while True:
        records = upload_queue.get()
        if records is None:
            upload_queue.task_done()
            break
        for attempt in range(1, MAX_INSERT_RETRIES + 1):
            try:
                milvus_client.insert(collection_name="world_locations", data=records)
                break
            except Exception as e:
                if attempt == MAX_INSERT_RETRIES: print(f" [!!] Milvus Drop: {e}")
                else: time.sleep(2 ** attempt)
        upload_queue.task_done()

def download_worker(zip_names):
    for name in zip_names:
        try:
            path = hf_hub_download(repo_id="osv5m/osv5m", filename=f"images/train/{name}", repo_type="dataset", token=HF_TOKEN)
            download_queue.put(path)
        except Exception as e: print(f" [!] HF Download Error: {name} -> {e}")
    download_queue.put(None)

def _start_batch(batch_images, batch_coords, pool):
    """Run CLIP on the GPU, then immediately submit GEE + reverse-geocode
    to the thread pool so they run concurrently with the *next* CLIP call.
    Returns a pending tuple consumed by _collect_batch."""
    clip_vecs = get_streetclip_embedding(batch_images, processor, vision_model, device)
    coords_copy = list(batch_coords)
    ae_future = pool.submit(alpha_earth_features, coords_copy)
    geo_future = pool.submit(rg.search, coords_copy)
    return (clip_vecs, ae_future, geo_future, coords_copy)


def _collect_batch(pending):
    """Block on the GEE / reverse-geocode futures and build Milvus records.
    Call this while the *next* batch's CLIP is running to hide GEE latency."""
    clip_vecs, ae_future, geo_future, batch_coords = pending
    ae_vecs = ae_future.result()
    geo_results = geo_future.result()
    records = []
    for i, (lat, lon) in enumerate(batch_coords):
        cc = geo_results[i]["cc"]
        records.append({
            "streetclip_vec": clip_vecs[i].tolist(),
            "alphaearth_vec": ae_vecs[i],
            "gps": {"lat": float(lat), "lon": float(lon)},
            "country_code": cc,
            "continent": CC_TO_CONTINENT.get(cc, "Unknown"),
            "s2_token_l10": get_s2_token(lat, lon, 10),
            "s2sphere_boundary": get_s2_token(lat, lon, 12),
            "hyp_tangent_vec": [0.0] * 128,
        })
    return records

def run_pipeline():
    sample_frac = _train_sample_fraction()
    _init_runtime()
    ckpt_path = _checkpoint_path_for_sample_fraction(sample_frac)
    milvus_client = init_milvus()
    checkpoint = CheckpointManager(filepath=ckpt_path)

    print(">>> Loading OSV-5M Metadata...")
    csv_path = hf_hub_download(repo_id="osv5m/osv5m", filename="train.csv", repo_type="dataset", token=HF_TOKEN)
    df = pd.read_csv(csv_path, low_memory=False)
    n_rows_full = len(df)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    coords_lookup = df.set_index("id")[["latitude", "longitude"]].to_dict("index")
    valid_ids = set(coords_lookup.keys())
    n_rows_used = len(valid_ids)

    print(
        f">>> train.csv rows: {n_rows_full:,}; using {n_rows_used:,} IDs "
        f"(TRAIN_SAMPLE_FRACTION={sample_frac:g})"
    )
    if sample_frac < 1.0:
        print(f">>> checkpoint: {ckpt_path} (override with PIPELINE_CHECKPOINT_FILE=...)")
    _print_ingest_estimate(n_rows_used, BATCH_SIZE, alpha_earth=ALPHA_EARTH_ON)

    all_zips = [f"{str(i).zfill(2)}.zip" for i in range(98)]
    zips_to_process = [z for z in all_zips if not checkpoint.is_processed(z)]

    rows = _milvus_row_count(milvus_client)
    if rows == 0 and checkpoint.processed_zips:
        print(
            f"Milvus has 0 rows but checkpoint lists {len(checkpoint.processed_zips)} zips "
            "(DB was cleared or never loaded). Clearing checkpoint and re-ingesting."
        )
        checkpoint.clear()
        zips_to_process = list(all_zips)

    if not zips_to_process:
        print(f"All zips checkpointed; Milvus has {rows} rows. Nothing to do.")
        return

    print(f">>> {len(zips_to_process)} zips to process")
    Thread(target=download_worker, args=(zips_to_process,), daemon=True).start()
    Thread(target=milvus_worker, args=(milvus_client,), daemon=True).start()

    db_buffer = []
    batch_imgs, batch_coords = [], []
    pool = ThreadPoolExecutor(max_workers=4)
    # Double-buffer state: holds (clip_vecs, ae_future, geo_future, coords) for
    # the previous batch so its GEE results are collected while the *current*
    # batch's CLIP inference runs, hiding GEE latency behind GPU work.
    pending_batch = None

    def _flush_pending():
        nonlocal pending_batch, db_buffer
        if pending_batch is not None:
            db_buffer.extend(_collect_batch(pending_batch))
            pending_batch = None

    def _submit_and_collect(imgs, coords):
        """Start a new batch (CLIP + async GEE), collect the previous one."""
        nonlocal pending_batch, db_buffer
        new_pending = _start_batch(imgs, coords, pool)
        _flush_pending()           # collect previous batch's GEE results
        pending_batch = new_pending

    try:
        while True:
            zip_path = download_queue.get()
            if zip_path is None: break

            z_name = os.path.basename(zip_path)
            print(f"\n>>> Extracting & Processing {z_name}")

            with zipfile.ZipFile(zip_path, "r") as zf:
                jpg_names = [f for f in zf.namelist() if f.lower().endswith(".jpg")]
                batch_imgs, batch_coords = [], []

                for name in tqdm(jpg_names, desc=f"Processing {z_name}", leave=False):
                    img_id = int(os.path.basename(name).split(".")[0])
                    if img_id not in valid_ids: continue

                    try:
                        with zf.open(name) as f:
                            batch_imgs.append(Image.open(f).convert("RGB"))
                            row = coords_lookup[img_id]
                            batch_coords.append((row["latitude"], row["longitude"]))

                        if len(batch_imgs) >= BATCH_SIZE:
                            _submit_and_collect(batch_imgs, batch_coords)
                            batch_imgs, batch_coords = [], []

                        if len(db_buffer) >= DB_FLUSH_SIZE:
                            upload_queue.put(db_buffer)
                            db_buffer = []
                    except Exception: continue

            # Flush before checkpointing so all records from this ZIP are queued.
            _flush_pending()
            if batch_imgs:
                pending_batch = _start_batch(batch_imgs, batch_coords, pool)
                _flush_pending()
                batch_imgs, batch_coords = [], []
            if db_buffer:
                upload_queue.put(db_buffer)
                db_buffer = []

            if os.path.exists(zip_path): os.remove(zip_path)
            checkpoint.save(z_name)
            print(f"  Saved checkpoint for {z_name}.")

        # Final flush (shouldn't be needed after the per-ZIP flush above, but be safe)
        _flush_pending()
        if batch_imgs:
            pending_batch = _start_batch(batch_imgs, batch_coords, pool)
            _flush_pending()
        if db_buffer:
            print(f"  Queuing last {len(db_buffer)} records to Milvus...")
            upload_queue.put(db_buffer)
        print("  Waiting for Milvus worker to finish all inserts (can be several minutes)...")
        upload_queue.put(None)
        upload_queue.join()

    finally:
        pool.shutdown()
        print("\n[SUCCESS] Pipeline Finished.")

if __name__ == "__main__":
    run_pipeline()