# AlphaEarth Scout — Architecture

## Thesis

Global street-view geolocation as a **product of experts**:

```
p(location j | image) ∝ p(leaf_j | image) / n_leaf_j  ×  exp(kappa · cos(ae_hat(image), ae_j))
```

- **Cell prior**: a hierarchical posterior over the nested OSV5M quadtree (the binned dataset).
- **Satellite likelihood**: a vMF term in AlphaEarth space, a *fixed, physically grounded* 64d embedding field
  available at every 10 m land pixel. The image encoder learns to predict it (privileged-information distillation).

Because the location key is a satellite field rather than a learned GPS encoder (GeoCLIP) or a photo gallery
(PIGEON / OSV5M hybrid), the gallery can contain locations that were never photographed.

## Data: the binned dataset

OSV5M `train.csv` ships seven quadtree bin levels, `quadtree_10_{50000,25000,12500,5000,2500,1000,500}`.
`bins.py build` asserts they are **strictly nested** (every fine cell has exactly one parent) on all 4.89M rows:

| level | 50000 | 25000 | 12500 | 5000 | 2500 | 1000 | 500 |
|---|---|---|---|---|---|---|---|
| cells | 268 | 498 | 1,065 | 2,788 | 5,304 | 11,399 | 18,201 |

`test.csv` (210,122 rows, the official benchmark) carries only level 1000; coarser labels come from the parent map.
Cell centroids are spherical means over all train rows. Validation = 10% of official test **sequences** (hashed);
the disjoint 90% is the reported test. Train-side holdouts were rejected: OSV5M test sits much farther from train
(median nearest-train 5.6 km vs 1.4 km for a 1 km-buffered train val). Zip membership is uniform-random over
geography, so zips 00–39 (2.0M images) are an unbiased subsample.

Per-image OSV5M attributes (`drive_side`, Köppen `climate`, `land_cover`, `soil`, `road_index`, `dist_sea`)
are auxiliary targets.

## Features

- **StreetCLIP** (ViT-L/14-336, frozen): `image_embeds` stored as fp16 memmaps (`data/feats/clip_{split}.npy`),
  aligned to metadata rows. The same `load_streetclip` / `CLIPImageProcessor` path is used by `predict.py`.
- **AlphaEarth V1** (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), read from the public COG mirror on Source
  Cooperative instead of Earth Engine: point sampling cost ~2.3 EECU-s/point, so the ~8M points would need
  ~5,100 EECU-hours against a 150 EECU-hour/month noncommercial quota. One year (2020) for every point keeps the
  transfer at ~320 GB. Targets come from each tile's 80 m overview (Google's dequantize-sum-normalize pooling,
  verified at cos 0.9999 against pooling the 10 m pixels; cos 0.94 to the 10 m point itself) and 2x2 / 4x4
  pools of it (160 / 320 m); the pilot picks the scale.
  Missing pixels are NaN and masked out of every loss.

## Model (`model.py`)

```
StreetCLIP 768d -> L2 norm -> EmbeddingAugmentor -> SharedEncoder (768 -> 1024 -> 512, residual)
  ├── CellHead   (7 levels)   linear | euclid | lorentz
  ├── PhysicalHead            512 -> 256 -> 64, L2-normalized (predicted AlphaEarth)
  └── AuxHeads                4 categorical + 2 regression
```

**Cell head arms** share one logits interface:
- `linear`: one linear layer per level (baseline).
- `euclid`: prototypes with a tree parametrization `e_child = e_parent + delta_child`, logits `-s·||h - e||²`.
- `lorentz`: the same parametrization mapped to the Lorentz hyperboloid by `expmap0`, logits `-s·d_L(h, e)`,
  computed in FP32 with norm clipping. `lorentz` vs `euclid` isolates curvature from the tree structure.

**Losses**
- Cells: per-level CE against haversine-smoothed labels `q_c = softmax(-d(x, centroid_c) / tau_l)`, with
  `tau_l` = median member-to-centroid distance at that level (PIGEON-style).
- AlphaEarth: InfoNCE with a queue of recent targets; `--target-tau > 0` replaces one-hot labels with
  `softmax(t_i·t_j / tau_t)` so near-duplicate targets on the same road are not treated as negatives.
- Aux: CE (ignore missing) + smooth-L1 on standardized regression targets.

**Training**: the whole train set lives on the GPU (2M × 768 fp16 = 3.1 GB), batches of 2048 by random index,
fp16 autocast (Turing has no bf16), AdamW, warmup + cosine, EMA weights, early stopping on validation GeoScore.

## Inference and evaluation (`eval.py`)

1. **Leaf posterior**: `log p(leaf) = Σ_l log p_l(anc_l(leaf))`, renormalized over leaves; keep the top 256.
2. **Retrieval**: exact chunked GEMM over a GPU-resident gallery, adding the cell prior `log p(leaf) - log n_leaf`
   to `kappa · cos` for every gallery point. No ANN, so ANN recall never confounds a result.
3. **Refinement**: score-weighted spherical mean of hits within R km of the best hit.

| method | description |
|---|---|
| B0 / B0-flat | argmax leaf centroid (hierarchical / finest head only) |
| B1 | training-free StreetCLIP kNN over the train photos |
| B2 | cell prior + StreetCLIP kNN (the strongest classical baseline) |
| G0 / G1 / G2 | cell prior + AlphaEarth vMF over photo locations / + photo-free train locations / + per-cell grid |
| Fused | cell prior + AlphaEarth vMF + StreetCLIP vMF |
| *-serve32 | Milvus-equivalent path: rank by similarity inside the top-32 leaves, then rescore |

`kappa`, `lambda` and R are tuned on validation only; the 90% test slice is evaluated once per method. Reported:
median/mean km, accuracy at 1/25/200/750/2500 km, GeoScore `5000·exp(-d/1492.7)`, country/admin1/city accuracy
(same reverse geocoder for prediction and ground truth), per-level cell accuracy, calibration (ECE, error by
confidence decile), AlphaEarth diagnostics (cosine, within-cell MRR vs random), strata (continent, land cover,
cell density decile, distance to nearest train image), and 1000× bootstrap 95% CIs with **paired** deltas vs B2.

## Serving

- Milvus collection `world_locations_v2`: `ae_vec FLOAT_VECTOR(64)` (HNSW, COSINE), `lat`/`lon` DOUBLE,
  `leaf INT32` (INVERTED), `src INT8` (photo / photo-free / grid), `country_code`.
- `ingest.py milvus` loads exactly `eval.Env.gallery(...)` for the configuration selected on validation, so
  serving and evaluation share one gallery definition, and writes the per-leaf counts used by the prior.
- `predict.py | inference`: leaf-filtered search (`leaf in [...]`), rescoring with the same formula, the same
  cluster rule, and a cell-centroid fallback. `eval.py parity` checks the live path against eval's serve mode.

## Infrastructure

- Milvus 2.6.7 standalone + etcd + MinIO via Docker Compose (serving only; training never reads Milvus).
- Python: PyTorch + transformers + rasterio; Rust: reqwest + tokio + clap.
