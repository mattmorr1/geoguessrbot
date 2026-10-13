# AlphaEarth Scout

Street-view geolocalization as a product of experts: a hierarchical geocell posterior over the nested OSV5M
quadtree, times a von Mises-Fisher likelihood in Google AlphaEarth's satellite embedding space.

## How it works

```
Image -> StreetCLIP (frozen, 768d) -> AlphaBridge (trainable)
                                        ├── 7 cell heads: quadtree_10_{50000 .. 500}  (haversine-smoothed CE)
                                        ├── 64d AlphaEarth prediction                  (geo-aware soft InfoNCE)
                                        └── aux: drive_side, climate, land_cover, soil, road_index, dist_sea

score(location j) = log p(leaf_j | img) - log n_leaf_j + kappa * cos(ae_hat, ae_j)
```

The gallery of candidate locations is keyed by AlphaEarth vectors, which exist at every 10 m of land, so it can
include places nobody photographed (G1: photo-free train locations, G2: per-cell grid). See
[architecture.md](architecture.md) for the method and evaluation protocol.

## Prerequisites

- Python 3.11, CUDA GPU (developed on an RTX 2080 8 GB), ~150 GB free disk while encoding
- Rust stable, Docker + Docker Compose (serving only)
- ~320 GB of network transfer for AlphaEarth (read from the public Source Cooperative COG mirror; no Earth Engine
  account or quota needed). Optional `.env` with `HF_TOKEN`; export it before running: `set -a; . ./.env; set +a`

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -p .venv -r requirements-dev.txt
```

## Pipeline

All commands run from `scripts/`. Outputs land in `data/` (features, metadata), `runs/` (checkpoints, logs) and
`results/` (reports); every stage is resumable.

```bash
# 1. Bins: nested quadtree, centroids, sequence-hashed 10/90 test tune/final split, aux labels (~20 s)
python bins.py build

# 2. Frozen StreetCLIP features (~70 img/s on a 2080; zips are uniform-random samples)
python ingest.py encode --split train --zip-start 0 --zip-end 40
python ingest.py encode --split test  --zip-end 5
python bins.py align              # consolidated fp16 memmaps aligned to metadata rows
python bins.py grid               # per-cell grid points for gallery G2

# 3. AlphaEarth targets and galleries at 80/160/320 m for train, test and grid: one pass over ~11k COG tiles,
#    ~4 h at 23 MB/s (resumable, threaded) -> data/feats/ae/{set}_r{scale}.npy
python geo_lookup.py

# 4. Train (flags select ablations: --head linear|euclid|lorentz, --w-ae 0, --w-aux 0, --target-tau 0, --batch cell)
python train.py --name r4 --ae-radius 50

# 5. Evaluate: tune on val (10% of test), one pass on the other 90% of the official test split, then aggregate
python eval.py tune runs/r4/best.pt
python eval.py test runs/r4/best.pt
python eval.py report             # results/report.md + results/best_config.json (selected on val)
```

## Serving

```bash
docker compose up -d                              # Milvus standalone
python ingest.py milvus                           # loads the gallery named in best_config.json
cd inference && cargo build --release && cd ..
python predict.py photo.jpg | inference/target/release/inference
python eval.py parity runs/r4/best.pt -n 200      # serving path vs eval's serve-mode predictions
```

`predict.py` emits one JSON payload per image (top-32 cells with priors, predicted AlphaEarth vector, tuned
`kappa`/radius, and a cell-centroid fallback). The Rust engine runs a leaf-filtered Milvus search, rescores
with the same product-of-experts formula as `eval.py`, and prints:

```json
{"lat": 48.85, "lon": 2.35, "cluster_size": 7, "cluster_radius_km": 3.2, "fallback_used": false, "top": [...]}
```

## Configuration

Defaults live in `scripts/config.py`; all are overridable by environment variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` / `RUNS_DIR` / `RESULTS_DIR` | `<repo>/data`, `runs`, `results` | Output locations |
| `ENCODE_BATCH` / `ENCODE_WORKERS` | `64` / `12` | StreetCLIP batch size and JPEG decode workers |
| `AE_YEAR` / `AEF_THREADS` | `2020` / `4` | AlphaEarth year used for every point, concurrent COG tile reads |
| `TRAIN_BATCH_SIZE` / `EPOCHS` / `LR` | `2048` / `20` / `1e-3` | Training (GPU-resident data, no accumulation) |
| `EARLY_STOP_PATIENCE` | `3` | Epochs without val GeoScore gain before stopping |
| `QUEUE_SIZE` | `8192` | AlphaEarth negatives queue depth |
| `MILVUS_URI` / `MILVUS_COLLECTION` | `http://localhost:19530` / `world_locations_v2` | Serving store |

## Tests

```bash
pytest tests/ -v                       # from repo root
cd scripts/inference && cargo test
```
