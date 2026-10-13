#!/usr/bin/env python3
"""
Ingest: stream OSV5M zips -> frozen StreetCLIP embeddings (fp16 shards), and load the serving gallery into Milvus.

Usage:
  python ingest.py encode --split train --zip-start 0 --zip-end 40
  python ingest.py encode --split test  --zip-end 5
  python ingest.py encode --split train --zip-end 4 --backbone siglip2   # alternate encoder, own shards
  python ingest.py milvus     # gallery named in results/best_config.json -> Milvus + data/serve/leaf_counts.npy

Encoding progress is saved to data/ingest_checkpoint.json so re-runs skip completed zips.
"""

import argparse
import json
import logging
import os
import shutil
import time
import zipfile
from queue import Queue
from threading import Thread

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import bins
import config

logger = logging.getLogger(__name__)


# ── Checkpoint ────────────────────────────────────────────────────────────────

def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed": {}}


def save_checkpoint(path: str, state: dict) -> None:
    with open(path, 'w') as f:
        json.dump(state, f, indent=2)


# ── Download / extract ────────────────────────────────────────────────────────

def download_and_extract(split: str, zip_name: str) -> str | None:
    try:
        zip_path = hf_hub_download(
            repo_id="osv5m/osv5m",
            filename=f"images/{split}/{zip_name}",
            repo_type='dataset',
            local_dir=config.TEMP_DIR,
            token=config.HF_TOKEN,
        )
        extract_to = os.path.join(config.TEMP_DIR, f"{split}_" + zip_name.replace(".zip", ""))
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)
        os.remove(zip_path)
        return extract_to
    except Exception as e:
        logger.error("Failed to download/extract %s/%s: %s", split, zip_name, e)
        return None


def producer_task(split: str, zip_names: list[str], queue: Queue) -> None:
    for name in zip_names:
        path = download_and_extract(split, name)
        if path:
            queue.put(path)
        else:
            logger.warning("Skipping %s: download failed (not marked complete; re-run to retry)", name)
    queue.put(None)  # sentinel


def get_all_images(directory: str) -> list[str]:
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(directory)
        for f in files
        if f.lower().endswith('.jpg')
    ]


# ── Encoding ──────────────────────────────────────────────────────────────────

class ImageFiles(Dataset):
    """CLIPImageProcessor runs in DataLoader workers, so preprocessing parity with predict.py is exact."""

    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor
        size = getattr(processor, "crop_size", None) or processor.size
        self.shape = (3, size["height"], size["width"])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.processor(images=img, return_tensors="pt")["pixel_values"][0], True
        except Exception as e:
            logger.warning("unreadable image %s: %s", self.paths[i], e)
            return torch.zeros(self.shape), False


def encode_dir(folder, processor, vision, dim):
    from model import encode_pixels
    paths = sorted(get_all_images(folder))
    ids = np.array([int(os.path.basename(p).split(".")[0]) for p in paths], np.int64)
    feats, ok, i = np.empty((len(paths), dim), np.float16), np.zeros(len(paths), bool), 0
    loader = DataLoader(ImageFiles(paths, processor), batch_size=config.ENCODE_BATCH,
                        num_workers=config.ENCODE_WORKERS, pin_memory=True)
    for px, good in tqdm(loader, desc=os.path.basename(folder)):
        feats[i:i + len(px)] = encode_pixels(vision, px).half().cpu().numpy()
        ok[i:i + len(px)] = good.numpy()
        i += len(px)
    return feats[ok], ids[ok], int((~ok).sum())


def run_encode(split: str, zip_names: list[str], state: dict, backbone: str) -> None:
    from model import encode_pixels, load_backbone
    processor, vision = load_backbone(backbone)
    dim = encode_pixels(vision, torch.zeros(1, *ImageFiles([], processor).shape)).shape[-1]
    out_dir = bins.shard_dir(split, backbone)
    os.makedirs(out_dir, exist_ok=True)

    queue: Queue = Queue(maxsize=config.MAX_BUFFERED_ZIPS)
    Thread(target=producer_task, args=(split, zip_names, queue), daemon=True).start()
    while (folder := queue.get()) is not None:
        name = os.path.basename(folder).split("_", 1)[1]
        t0 = time.time()
        feats, ids, bad = encode_dir(folder, processor, vision, dim)
        np.save(os.path.join(out_dir, f"{name}.npy"), feats)
        np.save(os.path.join(out_dir, f"{name}.ids.npy"), ids)
        shutil.rmtree(folder)
        state["completed"].setdefault(bins.ckpt_key(split, backbone), []).append(f"{name}.zip")
        save_checkpoint(config.INGEST_CHECKPOINT, state)
        logger.info("%s/%s: %d images (%d unreadable) at %.1f img/s",
                    split, name, len(ids), bad, len(ids) / (time.time() - t0))


# ── Milvus serving gallery ────────────────────────────────────────────────────

def setup_collection(client, reset: bool = True) -> None:
    from pymilvus import DataType
    if client.has_collection(config.MILVUS_COLLECTION):
        if not reset:
            return
        logger.warning("Dropping collection '%s'", config.MILVUS_COLLECTION)
        client.drop_collection(config.MILVUS_COLLECTION)

    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field("id",           DataType.INT64, is_primary=True)
    schema.add_field("ae_vec",       DataType.FLOAT_VECTOR, dim=64)
    schema.add_field("lat",          DataType.DOUBLE)
    schema.add_field("lon",          DataType.DOUBLE)
    schema.add_field("leaf",         DataType.INT32)
    schema.add_field("src",          DataType.INT8)
    schema.add_field("country_code", DataType.VARCHAR, max_length=8)

    idx = client.prepare_index_params()
    idx.add_index("ae_vec", "HNSW", "COSINE", params={"M": 16, "efConstruction": 200})
    idx.add_index("leaf", "INVERTED")
    client.create_collection(config.MILVUS_COLLECTION, schema=schema, index_params=idx)
    logger.info("Collection '%s' created", config.MILVUS_COLLECTION)


def insert_with_retry(client, records: list[dict]) -> None:
    for attempt in range(1, config.MAX_INSERT_RETRIES + 1):
        try:
            client.insert(collection_name=config.MILVUS_COLLECTION, data=records)
            return
        except Exception as e:
            if attempt == config.MAX_INSERT_RETRIES:
                raise
            logger.warning("Milvus insert attempt %d/%d failed: %s", attempt, config.MAX_INSERT_RETRIES, e)
            time.sleep(2 ** attempt)


def load_milvus() -> None:
    """Loads exactly eval.Env.gallery(cfg['gallery']) so serving and eval share one gallery definition."""
    import reverse_geocoder as rg
    from pymilvus import MilvusClient
    import eval as ev

    cfg = json.load(open(os.path.join(config.RESULTS_DIR, "best_config.json")))
    if cfg["field"] != "ae_vec":
        raise SystemExit(f"serving field {cfg['field']} not supported by the v2 schema (Gate B fallback)")
    radius = torch.load(cfg["ckpt"], map_location="cpu", weights_only=False)["args"]["ae_radius"]
    env = ev.Env(radius)
    gal = env.gallery(cfg["gallery"])
    ae, leaf = gal["vec"]["ae"].float().cpu().numpy(), gal["leaf"].cpu().numpy()
    cc = np.array([r["cc"] for r in rg.search(list(zip(gal["lat"], gal["lon"])), verbose=False)])

    client = MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN)
    if config.MILVUS_DB not in client.list_databases():
        client.create_database(config.MILVUS_DB)
    client = MilvusClient(uri=config.MILVUS_URI, token=config.MILVUS_TOKEN, db_name=config.MILVUS_DB)
    setup_collection(client, reset=True)
    for s in tqdm(range(0, len(leaf), config.MILVUS_BATCH), desc="milvus"):
        sl = slice(s, s + config.MILVUS_BATCH)
        insert_with_retry(client, [
            {"ae_vec": v, "lat": float(la), "lon": float(lo), "leaf": int(lf), "src": int(sr), "country_code": c}
            for v, la, lo, lf, sr, c in zip(ae[sl].tolist(), gal["lat"][sl], gal["lon"][sl], leaf[sl], gal["src"][sl], cc[sl])])
    client.flush(config.MILVUS_COLLECTION)
    client.load_collection(config.MILVUS_COLLECTION)

    os.makedirs(config.SERVE_DIR, exist_ok=True)
    np.save(os.path.join(config.SERVE_DIR, "leaf_counts.npy"), np.bincount(leaf, minlength=env.n_leaf))
    logger.info("Milvus: %d gallery points loaded into %s", len(leaf), config.MILVUS_COLLECTION)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cmd", choices=["encode", "milvus"])
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--zip-start", type=int, default=0, help="First zip index, inclusive")
    parser.add_argument("--zip-end",   type=int, default=1, help="Last zip index, exclusive")
    parser.add_argument("--backbone", choices=["streetclip", "siglip2"], default="streetclip")
    args = parser.parse_args()

    os.makedirs(config.LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(config.LOG_DIR, "ingest.log")),
        ],
    )

    if args.cmd == "milvus":
        return load_milvus()

    os.makedirs(config.TEMP_DIR, exist_ok=True)
    state = load_checkpoint(config.INGEST_CHECKPOINT)
    done = set(state["completed"].get(bins.ckpt_key(args.split, args.backbone), []))
    zip_names = [f"{i:02d}.zip" for i in range(args.zip_start, args.zip_end) if f"{i:02d}.zip" not in done]
    if not zip_names:
        logger.info("All %s zips already completed per checkpoint", args.split)
        return
    logger.info("Encoding %d %s zip(s): %s", len(zip_names), args.split, zip_names)
    run_encode(args.split, zip_names, state, args.backbone)


if __name__ == "__main__":
    main()
