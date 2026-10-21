# AlphaEarth Scout — Architecture

## Thesis

Street-view geolocation as **retrieve, then rerank**, with no language model anywhere:

1. A **cell prior**: a hierarchical posterior over the nested OSV5M quadtree.
2. Two **retrieval experts** under that prior: kNN over train photos (B2), and a two-tower match between the image
   and a learned embedding of AlphaEarth satellite context at candidate locations (T2).
3. A **reranker** that picks one location from the pooled candidates, maximizing expected GeoScore.

AlphaEarth (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`) is a fixed, physically grounded 64d field available at every
10 m land pixel, so the T2 gallery includes locations nobody photographed. That is where it earns its keep: on test
photos more than 25 km from any train photo the reranker gains 95-116 GeoScore over B2.

## Data: the binned dataset

OSV5M `train.csv` ships seven quadtree bin levels, `quadtree_10_{50000,25000,12500,5000,2500,1000,500}`.
`bins.py build` asserts they are **strictly nested** (every fine cell has exactly one parent) on all 4.89M rows:

| level | 50000 | 25000 | 12500 | 5000 | 2500 | 1000 | 500 |
|---|---|---|---|---|---|---|---|
| cells | 268 | 498 | 1,065 | 2,788 | 5,304 | 11,399 | 18,201 |

`test.csv` (210,122 rows, the official benchmark) carries only level 1000; coarser labels come from the parent map.
Cell centroids are spherical means over all train rows. Validation = 10% of official test **sequences** (hashed,
fold 0); the disjoint folds 1-9 are the reported test. Train-side holdouts were rejected: OSV5M test sits much
farther from train (median nearest-train 4.4 km for val queries vs 0.3 km for a sequence-held-out train slice), so
a scorer fit on train-side queries learns to over-trust near-duplicate neighbours. Zip membership is uniform-random
over geography, so zips 00-39 (2.0M images) are an unbiased subsample.

`SPLIT=blocks` builds an alternative split whose test regions are spatially held out from train (used by the demo).

## Features

- **Images**: StreetCLIP (ViT-L/14-336) and SigLIP 2 (so400m-patch14-384), both frozen, stored as fp16 memmaps
  aligned to metadata rows. `fusion768` is a PCA-768 of the concatenated unit-normalized embeddings, fit on train.
  `serve.py` encodes uploads through the same `encode_pixels` path as `ingest.py`.
- **AlphaEarth**, read from the public COG mirror on Source Cooperative instead of Earth Engine (point sampling the
  ~8M points would need ~5,100 EECU-hours against a 150/month noncommercial quota). One year (2020) for every point
  keeps transfer at ~320 GB. Per point: pooled context at 80/160/320 m (Google's dequantize-sum-normalize pooling,
  cos 0.9999 against pooling the 10 m pixels) and a coarse cell pyramid. Missing pixels are NaN and masked.

## Model (`model.py`)

```
fusion768 -> L2 norm -> SharedEncoder (768 -> 1024 -> 512, residual)
  ├── CellHead   (7 levels, linear)              haversine-smoothed CE per level
  └── loc_head   512 -> tower dim                 InfoNCE against LocationTower(AlphaEarth context + pyramid)
```

- Cells: per-level CE against `q_c = softmax(-d(x, centroid_c) / tau_l)`, `tau_l` = median member-to-centroid
  distance at that level (PIGEON-style).
- Two-tower: the image embedding is matched to `LocationTower(ae context, cell pyramid)` of the photo's own location
  with geodesic soft targets (`--loc-tau-km`), so near-duplicate locations are not hard negatives.
- `f1` trains the prior only (`--loc none`); `ft` adds the tower (`--loc ae+pyr`). The AlphaEarth regression and
  auxiliary-attribute heads exist (`--w-ae`, `--w-aux`) but are off in the headline runs.

Training streams host-resident features to the GPU per batch (batches of 2048), fp16 autocast, AdamW, warmup +
cosine, EMA weights, early stopping on validation GeoScore.

## Retrieval (`eval.py`)

1. **Leaf posterior**: `log p(leaf) = Σ_l log p_l(anc_l(leaf))`, renormalized over leaves; keep the top 256.
2. **Retrieval**: exact chunked GEMM, adding `log p(leaf) - log n_leaf` to the weighted similarity of every gallery
   point. Galleries above 4 GB stay on the host and stream through the GPU chunk by chunk. No ANN, so ANN recall
   never confounds a result.
3. **Refinement**: score-weighted spherical mean of hits within R km of the best hit.

| method | score per gallery point |
|---|---|
| B0 | argmax leaf centroid |
| B1 | training-free photo kNN |
| B2 | cell prior + `lambda · cos(img, photo_j)` over train photos |
| T2 | cell prior + `kappa · cos(img_emb, tower(location_j))` over photo locations + per-cell grid points |
| BT2 | union: photos keep the photo term and gain the tower term; grid points get the tower term + an offset |

`kappa`, `lambda`, R and the BT2 offset are tuned on validation only (`tuned.json`).

## Reranker (`rerank.py`)

Per photo, 75 candidates: the top 32 B2 hits, the top 32 T2 hits, the top 8 prior cells, and the B0/B2/T2 point
predictions. Each candidate gets 23 location features: B2 / T2 hit mass within 1-500 km and prior-cell mass within
25-500 km, log km to the B0/B2/T2 outputs, its own leaf and coarse-level prior log-probabilities, gallery density,
its raw score and gap to the best of its source, and log km to the nearest other candidate. Plus 7 static features
(source one-hot, within-source rank) and 7 per-photo features (top-cell confidence, prior entropy, B2/T2 top-1 score
gaps, B2-T2 disagreement in km, peak B2/T2 hit weight).

- **Scorer**: LightGBM lambdarank over the flattened candidates, labels = GeoScore binned into 31 grades, early
  stopping on a held-out 10% of validation.
- **Pick**: minimum Bayes risk, `argmax_j Σ_i softmax(s/T)_i · exp(-d(c_j, c_i)/1492.7)`, i.e. the candidate with the
  highest expected GeoScore under the scorer's distribution. `T = 0.25`, chosen leave-one-fold-out on validation.
- The scorer and `T` were selected on validation (5-fold sequence cross-fit inside val) against an MLP with a
  listwise expected-GeoScore loss: LightGBM + MBR beat it by +24.6 [+16.8, +32.2] GeoScore. That bake-off, the MLP,
  and the rejected train-holdout variants live on the `experiments` branch.

## Results

Official test folds 1-9 (n = 188,249), one pass, 1000× bootstrap 95% CIs, paired deltas.

| | GeoScore | median km | acc@1 | @25 | @200 | @750 | @2500 km |
|---|---|---|---|---|---|---|---|
| Reranker | 4124.7 [4118.8, 4130.4] | 73.9 | 1.3 | 29.6 | 67.7 | 84.7 | 93.5 |
| B2 | 4055.6 [4049.5, 4061.7] | 77.0 | 0.8 | 28.4 | 65.6 | 83.0 | 92.5 |
| Oracle (best of 75) | 4604.8 | 15.2 | 11.8 | 57.7 | 84.4 | 94.2 | 97.9 |
| Pinpoint (reported) | 4114 | — | — | 35.6 | 67.5 | 83.7 | 93.2 |

- Reranker vs B2: +65.6 to +72.3 GeoScore; on the 25,674 photos > 25 km from any train photo, 3205.5 vs 3100.0.
- 39% of picks come from AlphaEarth candidates (T2 hits and the T2 output).
- The gain is at regional and continental scale; Pinpoint is clearly better at street/city scale (acc@25 km).
- The oracle gap (480 GeoScore) says retrieval already surfaces the right place; selection is the bottleneck.

**Disclosures**

- Test subset: folds 1-9 of the official test; fold 0 is validation. Pinpoint reports on the full test set, and our
  CI is compared against its point estimate.
- Single seed for every model; CIs cover sampling noise only.
- The photo gallery is the 2.0M-image subsample (zips 00-39); the full 4.9M train set is not yet used.
- External data: AlphaEarth satellite embeddings (public, unlabelled with respect to photos). No extra
  photo-coordinate pairs beyond OSV-5M train; no test photos in any gallery.

## Infrastructure

Python only: PyTorch, transformers, LightGBM, rasterio. `serve.py` is a stdlib HTTP demo; there is no serving stack.
