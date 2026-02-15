# AlphaEarth Scout

Multi-Modal Hierarchical Geolocalization System

## System Overview

```
Image(s) -> StreetCLIP (frozen 768d) -> Multi-View Pooling -> Shared Encoder (512d)
  |                                                               |
  +--- Administrative Head -> Country logits (cross-entropy)      |
  +--- Physical Head -> 64d DNA (InfoNCE)                         |
  |         +--- Auxiliary Decoder -> Climate Zone + Elevation     |
  +--- Hierarchical Head -> 128d tangent -> 129d Lorentz manifold |
```

## Frozen Backbones

- **StreetCLIP** (ViT-L/14): 768d visual encoder, 1.1M Street View images. Frozen.
- **AlphaEarth V1**: 64-band satellite embedding (GEE). Training target.

## Multi-View Patching (PIGEON-style)

When N views are available (cardinal directions from panoramic data), they're
aggregated via learned cross-attention with a CLS token. Falls through to
identity for single-view input, so the same weights work either way.

## Multi-Head Bridge

```
Shared Encoder: Linear(768->1024) -> LayerNorm -> GELU -> Linear(1024->512) -> LayerNorm -> GELU

Administrative Head: Dropout(0.1) -> Linear(512 -> num_countries)
Physical Head:       Linear(512->256) -> GELU -> Linear(256->64) -> L2Norm
Hierarchical Head:   Linear(512->128) -> ExpMap0 -> Lorentz Hyperboloid (129d)
Auxiliary Decoder:   climate = Linear(64->32->6), elevation = Linear(64->32->8)
```

## Hyperbolic Geometry: Lorentz Model

Replaced Poincare ball with Lorentz hyperboloid. Poincare gradients explode
near ||x||->1. Lorentz is unbounded in ambient space so gradients stay healthy.

- ExpMap0: tangent -> `[cosh(||v||), sinh(||v||) * v/||v||]`
- Distance: `arccosh(-<x,y>_L)` where `<x,y>_L = -x0*y0 + x1..xn dot`
- Milvus stores 128d tangent vectors indexed with L2 (local proxy for geodesic distance)

## Auxiliary Losses

Force the 64d DNA vector to encode interpretable environmental semantics:
- **Climate Zone** (6 classes): tropical, arid, temperate, continental, polar, unknown
- **Elevation Bin** (8 classes): <0m, 0-100, 100-500, 500-1k, 1k-2k, 2k-3k, 3k-5k, >5k

Targets derived from GPS coordinates (latitude-based climate, heuristic elevation).
Replace with SRTM DEM + Koppen map for production accuracy.

## Curriculum Learning

Dynamic loss weights shift focus from easy to hard over training:

| Epoch Phase | alpha (Country) | beta (DNA) | gamma (Hierarchy) | aux |
|------------|-----------------|------------|-------------------|-----|
| Warmup     | 2.0             | 0.3        | 0.1               | 0.1 |
| Early      | ~1.5            | ~1.0       | ~0.5              | ~0.3|
| Late       | 0.5             | 2.0        | 1.0               | 0.5 |

## Loss Function

```
L = alpha * L_CE(country)
  + beta  * L_InfoNCE(DNA, temperature=0.07)
  + gamma * [d_Lorentz(img,country) + 0.3*d(img,continent) + 0.2*d(country,continent)]
  + aux   * [L_CE(climate) + L_CE(elevation)]
```

## Inference Pipeline

1. Image(s) -> StreetCLIP -> MultiView pooling -> Bridge -> {country, DNA, tangent}
2. Constrained Milvus search: `country_code == 'US'` on alphaearth_vec (COSINE)
3. Optional dual search: L2 on hyp_tangent_vec, fuse with DNA hits
4. Geodesic refinement: weighted Haversine center-of-mass in Cartesian

## Milvus Schema

Collection: `world_locations` (DB: `geoguessr`)

| Field | Type | Index |
|---|---|---|
| id | INT64 (auto) | Primary Key |
| streetclip_vec | FLOAT_VECTOR(768) | HNSW (COSINE, M=16) |
| alphaearth_vec | FLOAT_VECTOR(64) | HNSW (COSINE, M=8) |
| hyp_tangent_vec | FLOAT_VECTOR(128) | HNSW (L2, M=8) |
| gps | JSON | - |
| s2sphere_boundary | VARCHAR(512) | - |
| country_code | VARCHAR(8) | TRIE |
| continent | VARCHAR(32) | - |
| s2_token_l10 | VARCHAR(32) | TRIE |

## Training

- Optimizer: AdamW (lr=1e-4, weight_decay=1e-2)
- Scheduler: CosineAnnealing
- Gradient Accumulation: 32 steps (effective batch 1024)
- Mixed Precision: FP16 on CUDA, FP32 on MPS/CPU
- Gradient Clipping: max_norm=1.0
- Lorentz ops forced to FP32

## Infrastructure

- Milvus 2.6.7 (standalone) + etcd + MinIO via Docker Compose
- Rust inference (reqwest + tokio + clap): dual search + geodesic refinement
- Python training (PyTorch + transformers)
