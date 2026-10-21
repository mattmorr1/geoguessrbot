# AlphaEarth Scout

Street-view geolocalization without an LLM: frozen image encoders, a hierarchical geocell prior, photo retrieval,
and a location tower over Google AlphaEarth satellite embeddings, combined by a learned candidate reranker.

## Result (OSV-5M official test)

| method | GeoScore (95% CI) | median km | acc@25 km | @200 km | @750 km | @2500 km |
|---|---|---|---|---|---|---|
| Pinpoint (reported, full test) | 4114 | — | **35.6** | 67.5 | 83.7 | 93.2 |
| **AlphaEarth Scout** | **4124.7 [4118.8, 4130.4]** | 73.9 | 29.6 | **67.7** | **84.7** | **93.5** |
| B2: cell prior + photo kNN | 4055.6 [4049.5, 4061.7] | 77.0 | 28.4 | 65.6 | 83.0 | 92.5 |

Inductive: every learned and tuned component is fit on train or on the validation slice; test is scored once.
GeoScore is `5000·exp(-d/1492.7)`. See [architecture.md](architecture.md#results) for the protocol and disclosures.

## How it works

```
photo -> StreetCLIP + SigLIP 2 (frozen) -> PCA-768 fusion -> trunk -> 7 quadtree cell heads   (prior)
                                                                  -> image embedding          (two-tower)
candidates (75 per photo):
  32 photo-kNN hits  (cell prior + cos to 2M train photos)                       B2
  32 map hits        (cell prior + cos to tower(AlphaEarth) over photos + grid)   T2
   8 prior cells + the B0 / B2 / T2 point predictions
LightGBM lambdarank scores each candidate (37 features) -> minimum-Bayes-risk pick under expected GeoScore
```

The map gallery is keyed by AlphaEarth vectors, which exist for every 10 m of land, so it reaches places nobody
photographed; 39% of final picks come from it.

## Setup

Python 3.11 and a CUDA GPU (developed on an RTX 2080 8 GB). AlphaEarth is read from the public Source Cooperative
COG mirror (~320 GB transfer, no Earth Engine quota). Optional `.env` with `HF_TOKEN`: `set -a; . ./.env; set +a`.

```bash
uv venv -p 3.11 .venv
uv pip install -p .venv torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -p .venv -r requirements-dev.txt
```

## Pipeline

All commands run from `scripts/`; outputs go to `data/`, `runs/` and `results/`, and every stage resumes.

```bash
# 1. Bins: nested quadtree, centroids, sequence-hashed 10/90 test tune/final split, aux labels
python bins.py build

# 2. Frozen image features (zips are uniform-random; the headline uses train zips 00-39 = 2.0M photos)
python ingest.py encode --split train --zip-end 40
python ingest.py encode --split train --zip-end 40 --backbone siglip2
python ingest.py encode --split test  --zip-end 5
python ingest.py encode --split test  --zip-end 5 --backbone siglip2
python bins.py align && python bins.py align --backbone siglip2
python bins.py fuse               # PCA-768 of [StreetCLIP | SigLIP 2], fit on train
python bins.py grid               # per-cell grid points for the map gallery

# 3. AlphaEarth context for train, test and grid (~11k COG tiles, threaded), then the cell pyramid
python geo_lookup.py
python bins.py pyramid

# 4. Train: f1 = prior + photo retrieval, ft = adds the AlphaEarth location tower
python train.py --name f1 --feats fusion768 --w-ae 0 --w-aux 0
python train.py --name ft --feats fusion768 --loc ae+pyr --w-ae 0 --w-aux 0

# 5. Tune retrieval on val, then the reranker: fit on val, one pass on test
python eval.py tune ../runs/f1/best.pt --methods B1,B2
python eval.py tune ../runs/ft/best.pt --methods B2,T2,BT2
python rerank.py ../runs/ft/best.pt --prior ../runs/f1/best.pt    # -> runs/ft@rerank2_lgb/rerank.json
```

`eval.py test` gives the single-pass test metrics of each retrieval method; `eval.py report` aggregates runs.

### Reproducing the headline

The reranker caches its candidates and features per split (`data/rerank/official_{val,test}_v2_f1.pt`); with those
present, step 5's `rerank.py` refits and rescores in minutes on CPU. Encoding more train zips changes the photo
gallery, so the 4124.7 requires the 2.0M-photo feature files (zips 00-39) in `data/feats/`.

## Demo

```bash
SPLIT=blocks python eval.py gate ../runs_blocks/ft/best.pt --tag gate   # once
SPLIT=blocks python serve.py --run ft                                   # http://127.0.0.1:8000
```

Drop a photo and the page maps the B0 / B2 / T2 predictions and the top cells (with error if the photo has EXIF
GPS). The demo uses the spatial-block split so the held-out regions were never seen in training; it shows the
retrieval methods, not the reranker.

## Configuration

Defaults live in `scripts/config.py`; all are overridable by environment variable.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` / `RUNS_DIR` / `RESULTS_DIR` | `<repo>/data`, `runs`, `results` | Output locations |
| `SPLIT` | `official` | `official` OSV-5M test, or `blocks` (spatially held-out regions) |
| `ENCODE_BATCH` / `ENCODE_WORKERS` | `64` / `12` | Encoder batch size and JPEG decode workers |
| `AE_YEAR` / `AEF_THREADS` | `2020` / `4` | AlphaEarth year used for every point, concurrent COG tile reads |
| `TRAIN_BATCH_SIZE` / `EPOCHS` / `LR` | `2048` / `20` / `1e-3` | Training |
| `EARLY_STOP_PATIENCE` | `3` | Epochs without val GeoScore gain before stopping |

## Tests

```bash
pytest tests/ -q          # from repo root
```
