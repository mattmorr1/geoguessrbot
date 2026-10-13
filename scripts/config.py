"""Central configuration — all tunable values live here, overridable via env vars."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Infrastructure ──────────────────────────────────────────────────────────

MILVUS_URI        = os.getenv("MILVUS_URI",   "http://localhost:19530")
MILVUS_TOKEN      = os.getenv("MILVUS_TOKEN", "root:Milvus")
MILVUS_DB         = os.getenv("MILVUS_DB",    "geoguessr")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "world_locations_v2")

# ── Paths ────────────────────────────────────────────────────────────────────

DATA_DIR          = os.getenv("DATA_DIR",    os.path.join(ROOT, "data"))
DATASET_DIR       = os.path.join(DATA_DIR, "osv5m")
TEMP_DIR          = os.path.join(DATA_DIR, "tmp")
# SPLIT=blocks: novel-location protocol (held-out spatial blocks) in its own meta/runs/results namespace
SPLIT             = os.getenv("SPLIT", "official")
_NS               = "" if SPLIT == "official" else f"_{SPLIT}"
META_DIR          = os.path.join(DATA_DIR, "meta" + _NS)
FEAT_DIR          = os.path.join(DATA_DIR, "feats")
SERVE_DIR         = os.path.join(DATA_DIR, "serve")
RUNS_DIR          = os.getenv("RUNS_DIR",    os.path.join(ROOT, "runs" + _NS))
RESULTS_DIR       = os.getenv("RESULTS_DIR", os.path.join(ROOT, "results" + _NS))
LOG_DIR           = os.path.join(ROOT, "logs")
INGEST_CHECKPOINT = os.path.join(DATA_DIR, "ingest_checkpoint.json")

HF_TOKEN = os.getenv("HF_TOKEN")

# ── Bins ─────────────────────────────────────────────────────────────────────

QT_LEVELS = (50000, 25000, 12500, 5000, 2500, 1000, 500)  # coarse -> fine, strictly nested
TEST_LEVEL = 1000                                          # only level labelled in test.csv
AE_YEARS  = (2017, 2024)                                   # AlphaEarth annual coverage
GRID_YEAR = 2020

# ── Ingest ───────────────────────────────────────────────────────────────────

ENCODE_BATCH       = int(os.getenv("ENCODE_BATCH",       "64"))
ENCODE_WORKERS     = int(os.getenv("ENCODE_WORKERS",     "12"))
MAX_BUFFERED_ZIPS  = int(os.getenv("MAX_BUFFERED_ZIPS",  "2"))
MAX_INSERT_RETRIES = int(os.getenv("MAX_INSERT_RETRIES", "3"))
MILVUS_BATCH       = int(os.getenv("MILVUS_BATCH",       "5000"))

# ── AlphaEarth (public COG mirror; Earth Engine point sampling costs ~2.3 EECU-s/pt) ─────────

AEF_URL     = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/tge-labs/aef/v1/annual"
AE_YEAR     = int(os.getenv("AE_YEAR", "2020"))  # one year for every point keeps the download ~1x
AE_SCALES   = (80, 160, 320)                     # metres: 80 m COG overview, then 2x2 / 4x4 pooled
AEF_THREADS = int(os.getenv("AEF_THREADS", "4"))  # ~1.3 GB transient RAM per tile; one stream saturates ~23 MB/s
AEF_RETRIES = int(os.getenv("AEF_RETRIES", "5"))
PYR_DEPTHS  = (8, 9, 10, 11, 12)                 # lat/lon quadtree depths of the AE cell pyramid (~78x156 .. 5x10 km)
PYR_K       = 64                                 # bag-of-environments codebook size

# ── Training ─────────────────────────────────────────────────────────────────

TRAIN_BATCH_SIZE     = int(os.getenv("TRAIN_BATCH_SIZE",    "2048"))
EPOCHS               = int(os.getenv("EPOCHS",              "20"))
LR                   = float(os.getenv("LR",                "1e-3"))
WEIGHT_DECAY         = float(os.getenv("WEIGHT_DECAY",      "1e-2"))
WARMUP_EPOCHS        = int(os.getenv("WARMUP_EPOCHS",       "1"))
EMA_DECAY            = float(os.getenv("EMA_DECAY",         "0.999"))
QUEUE_SIZE           = int(os.getenv("QUEUE_SIZE",          "8192"))
EARLY_STOP_PATIENCE  = int(os.getenv("EARLY_STOP_PATIENCE",  "3"))
EARLY_STOP_MIN_DELTA = float(os.getenv("EARLY_STOP_MIN_DELTA", "1.0"))  # GeoScore points
