from datasets import load_dataset
from pymilvus import MilvusClient, DataType
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# 1. Setup Milvus & Model
client = MilvusClient("localhost:19530")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to("cuda")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 2. Create Collection Schema
schema = client.create_schema(auto_id=False, enable_dynamic_field=True)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=768)
schema.add_field(field_name="lat", datatype=DataType.FLOAT)
schema.add_field(field_name="lon", datatype=DataType.FLOAT)

client.create_collection(collection_name="geolocation_clip", schema=schema)

# 3. Stream & Process
ds = load_dataset("osv5m/osv5m", split="train", streaming=True)
batch_size = 64
data_batch = []

for i, example in enumerate(tqdm(ds)):
    # Encode Image
    with torch.no_grad():
        inputs = processor(images=example['image'], return_tensors="pt").to("cuda")
        emb = model.get_image_features(**inputs).cpu().numpy().tolist()[0]
    
    data_batch.append({
        "id": i,
        "vector": emb,
        "lat": example['lat'],
        "lon": example['lon']
    })
    
    # Bulk insert every batch_size
    if len(data_batch) >= batch_size:
        client.insert(collection_name="geolocation_clip", data=data_batch)
        data_batch = []