#!/usr/bin/env python3
"""
OSV5M quadtree bins: nested cell hierarchy, centroids, test tune/final split, aux labels, gallery grid.

Usage:
  python bins.py build   # train/test CSV -> data/meta/{train,test}.npz + cells.npz
  python bins.py align   # encoded CLIP shards -> consolidated fp16 memmaps aligned to meta rows
  python bins.py grid    # uniform per-cell points for the photo-free gallery (G2)
  SPLIT=blocks python bins.py blocks   # novel-location split -> data/meta_blocks (needs geo_lookup.py --cache)
"""

import argparse
import functools
import glob
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import os

import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)

R_EARTH_KM = 6371.0088
QT_COLS = [f"quadtree_10_{n}" for n in config.QT_LEVELS]
AUX_CAT = ("drive_side", "climate", "land_cover", "soil")
AUX_REG = ("road_index", "dist_sea")
BASE_COLS = ["id", "latitude", "longitude", "country", "sequence", "captured_at", *AUX_CAT, *AUX_REG]
N_LEVELS = len(config.QT_LEVELS)
LEAF = N_LEVELS - 1
TEST_L = config.QT_LEVELS.index(config.TEST_LEVEL)
TUNE_MOD = 10  # 1/TUNE_MOD of test sequences tune + early-stop; the disjoint rest is the reported test


# ── Geometry ──────────────────────────────────────────────────────────────────

def to_xyz(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def to_latlon(xyz):
    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    return np.degrees(np.arcsin(np.clip(xyz[..., 2], -1, 1))), np.degrees(np.arctan2(xyz[..., 1], xyz[..., 0]))


def haversine(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(np.asarray(lon2) - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ── Hierarchy ─────────────────────────────────────────────────────────────────

def parent_map(fine, coarse, n_fine):
    """parent[f] per fine cell; raises if any fine cell maps to two coarse cells (bins not nested)."""
    par = np.full(n_fine, -1, np.int64)
    par[fine] = coarse
    bad = np.unique(fine[par[fine] != coarse])
    if bad.size:
        raise ValueError(f"quadtree not nested: {bad.size} fine cells with multiple parents")
    return par


def ancestors(parents, n_leaf):
    """[n_leaf, N_LEVELS] ancestor id per level; parents[l] maps level l -> l-1."""
    anc = np.empty((n_leaf, len(parents) + 1), np.int64)
    anc[:, -1] = np.arange(n_leaf)
    for l in range(len(parents), 0, -1):
        anc[:, l - 1] = parents[l - 1][anc[:, l]]
    return anc


def cell_stats(xyz, cells, n):
    """Spherical-mean centroid, member count and mean member->centroid km for one level."""
    cnt = np.bincount(cells, minlength=n)
    cent = np.stack([np.bincount(cells, weights=xyz[:, k], minlength=n) for k in range(3)], 1)
    cent /= np.linalg.norm(cent, axis=1, keepdims=True)
    d = R_EARTH_KM * np.arccos(np.clip((xyz * cent[cells]).sum(1), -1, 1))
    return cent, cnt, np.bincount(cells, weights=d, minlength=n) / np.maximum(cnt, 1)


# ── Build ─────────────────────────────────────────────────────────────────────

def read_csv(split):
    path = os.path.join(config.DATASET_DIR, f"{split}.csv")
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        hf_hub_download("osv5m/osv5m", f"{split}.csv", repo_type="dataset",
                        local_dir=config.DATASET_DIR, token=config.HF_TOKEN)
    qt = QT_COLS if split == "train" else [f"quadtree_10_{config.TEST_LEVEL}"]
    # keep_default_na=False: country "NA" is Namibia, not missing
    return pd.read_csv(path, usecols=BASE_COLS + qt, keep_default_na=False, na_values=[""],
                       dtype={"id": np.int64, "country": str, "sequence": str})


def base_arrays(df, reg_mean, reg_std):
    year = pd.to_datetime(df["captured_at"], unit="ms", errors="coerce").dt.year
    return {
        "id":      df["id"].to_numpy(np.int64),
        "lat":     df["latitude"].to_numpy(np.float64),
        "lon":     df["longitude"].to_numpy(np.float64),
        "country": df["country"].fillna("").to_numpy("U3"),
        "year":    np.clip(year.fillna(config.GRID_YEAR).to_numpy(np.int16), *config.AE_YEARS),
        "aux_cat": np.stack([df[c].fillna(-1).to_numpy(np.int16) for c in AUX_CAT], 1),
        "aux_reg": ((df[list(AUX_REG)].to_numpy(np.float32) - reg_mean) / reg_std).astype(np.float32),
    }


def build():
    os.makedirs(config.META_DIR, exist_ok=True)
    tr = read_csv("train")
    logger.info("train rows: %d", len(tr))
    reg = tr[list(AUX_REG)].to_numpy(np.float64)
    reg_mean, reg_std = np.nanmean(reg, 0), np.nanstd(reg, 0)

    raw, cells = [], np.empty((len(tr), N_LEVELS), np.int64)
    for l, col in enumerate(QT_COLS):
        vals, cells[:, l] = np.unique(tr[col].to_numpy(np.int64), return_inverse=True)
        raw.append(vals)
    n_cells = [len(v) for v in raw]
    parents = [parent_map(cells[:, l], cells[:, l - 1], n_cells[l]) for l in range(1, N_LEVELS)]
    anc = ancestors(parents, n_cells[LEAF])
    logger.info("nested levels OK: %s", dict(zip(config.QT_LEVELS, n_cells)))

    train = base_arrays(tr, reg_mean, reg_std)
    xyz = to_xyz(train["lat"], train["lon"])
    stats = {}
    for l in range(N_LEVELS):
        cent, cnt, spread = cell_stats(xyz, cells[:, l], n_cells[l])
        clat, clon = to_latlon(cent)
        stats |= {f"lat_{l}": clat, f"lon_{l}": clon, f"count_{l}": cnt, f"raw_{l}": raw[l],
                  f"tau_{l}": np.float64(np.median(spread[cnt > 0]))}
    np.savez(os.path.join(config.META_DIR, "train.npz"), cells=cells.astype(np.int32), **train)
    n_aux = (train["aux_cat"].max(0) + 1).astype(np.int64)
    del tr

    te = read_csv("test")
    raw_t = te[f"quadtree_10_{config.TEST_LEVEL}"].to_numpy(np.int64)
    j = np.minimum(np.searchsorted(raw[TEST_L], raw_t), n_cells[TEST_L] - 1)
    tcells = np.full((len(te), N_LEVELS), -1, np.int64)
    tcells[:, TEST_L] = np.where(raw[TEST_L][j] == raw_t, j, -1)
    ok = tcells[:, TEST_L] >= 0
    for l in range(TEST_L, 0, -1):
        tcells[ok, l - 1] = parents[l - 1][tcells[ok, l]]
    # val = tune slice of the official test split: same spatial sparsity vs train as the reported test
    tune = pd.util.hash_array(te["sequence"].fillna("").to_numpy(object)) % TUNE_MOD == 0
    logger.info("test rows: %d (%d with unknown %d-cell); tune %d", len(te), (~ok).sum(), config.TEST_LEVEL, tune.sum())
    np.savez(os.path.join(config.META_DIR, "test.npz"), cells=tcells.astype(np.int32), tune=tune,
             **base_arrays(te, reg_mean, reg_std))

    np.savez(os.path.join(config.META_DIR, "cells.npz"), anc=anc, n_cells=np.array(n_cells),
             n_aux=n_aux, reg_mean=reg_mean, reg_std=reg_std,
             **{f"parent_{l}": p for l, p in enumerate(parents, 1)}, **stats)
    logger.info("wrote %s", config.META_DIR)


# ── Loading helpers ───────────────────────────────────────────────────────────

def test_folds():
    """Official test rows -> sequence-hash fold in [0, TUNE_MOD); fold 0 is the tune slice. Cached next to test.npz."""
    path = os.path.join(config.DATA_DIR, "meta", "test_folds.npy")
    if not os.path.exists(path):
        te, meta = read_csv("test"), np.load(os.path.join(config.DATA_DIR, "meta", "test.npz"))
        fold = (pd.util.hash_array(te["sequence"].fillna("").to_numpy(object)) % TUNE_MOD).astype(np.int8)
        if not (np.array_equal(te["id"].to_numpy(), meta["id"]) and np.array_equal(fold == 0, meta["tune"])):
            raise RuntimeError("test.csv no longer matches meta/test.npz; rebuild with bins.py build")
        np.save(path, fold)
    return np.load(path)


def load(name):
    with np.load(os.path.join(config.META_DIR, f"{name}.npz")) as z:
        return {k: z[k] for k in z.files}


def _tag(split, backbone):
    return f"clip_{split}" if backbone == "streetclip" else f"{backbone}_{split}"


def shard_dir(split, backbone="streetclip"):
    return os.path.join(config.FEAT_DIR, "clip" if backbone == "streetclip" else backbone, split)


def ckpt_key(split, backbone="streetclip"):
    return split if backbone == "streetclip" else f"{split}:{backbone}"


def _root(split):
    """Train features/AE are shared; a non-official SPLIT gathers its own query (test) and grid arrays."""
    return config.FEAT_DIR if config.SPLIT == "official" or split == "train" else config.META_DIR


def clip_path(split, backbone="streetclip"):
    return os.path.join(_root(split), f"{_tag(split, backbone)}.npy")


def encoded_rows(split, backbone="streetclip"):
    """Meta row index for each row of the consolidated feature memmap."""
    return np.load(os.path.join(_root(split), f"{_tag(split, backbone)}_rows.npy"))


def ae_path(name, scale):
    return os.path.join(_root(name), "ae", f"{name}_r{scale}.npy")


def ctx_path(name):
    return os.path.join(_root(name), "ae", f"{name}_ctx.npy")


def zip_of(split, backbone="streetclip"):
    """Per meta row: source zip index, -1 if the image was not encoded (raw-data property, split-independent)."""
    suffix = "" if backbone == "streetclip" else f"_{backbone}"
    return np.load(os.path.join(config.DATA_DIR, "meta", f"{split}_zip{suffix}.npy"))


def load_feats(split, rows, backbones):
    """Image features aligned to meta `rows`; fusion concatenates unit-normalized parts. A whole single backbone stays a memmap."""
    parts = []
    for b in backbones:
        mem, enc = np.load(clip_path(split, b), mmap_mode="c"), encoded_rows(split, b)
        if len(backbones) == 1 and np.array_equal(enc, rows):
            return mem
        inv = np.full(enc.max() + 1, -1)
        inv[enc] = np.arange(len(enc))
        x = np.asarray(mem[inv[rows]], np.float32)
        parts.append(x / np.linalg.norm(x, axis=1, keepdims=True) if len(backbones) > 1 else x)
    return np.concatenate(parts, 1).astype(np.float16)


def fuse(splits, dim=768, n_fit=300_000, chunk=1 << 17, seed=0):
    """Pseudo-backbone fusion{dim}: PCA (fit on train rows only) of [unit StreetCLIP | unit SigLIP2], StreetCLIP-sized."""
    bbs, name = ("streetclip", "siglip2"), f"fusion{dim}"
    rows = {s: functools.reduce(np.intersect1d, [encoded_rows(s, b) for b in bbs]) for s in splits}
    pca = os.path.join(config.FEAT_DIR, f"{name}_pca.npz")
    if not os.path.exists(pca):
        tr = functools.reduce(np.intersect1d, [encoded_rows("train", b) for b in bbs])
        fit = np.sort(np.random.default_rng(seed).choice(tr, min(n_fit, len(tr)), replace=False))
        X = load_feats("train", fit, bbs).astype(np.float64)
        mu = X.mean(0)
        w, V = np.linalg.eigh(np.cov(X - mu, rowvar=False))
        logger.info("%s: PCA %d -> %d keeps %.1f%% of variance (fit on %d train rows)", name, X.shape[1], dim,
                    100 * w[::-1][:dim].sum() / w.sum(), len(fit))
        np.savez(pca, mean=mu.astype(np.float32), W=np.ascontiguousarray(V[:, ::-1][:, :dim]).astype(np.float32))
    z = np.load(pca)
    for s, r in rows.items():
        out = np.lib.format.open_memmap(clip_path(s, name), mode="w+", dtype=np.float16, shape=(len(r), dim))
        for i in range(0, len(r), chunk):
            out[i:i + chunk] = (load_feats(s, r[i:i + chunk], bbs).astype(np.float32) - z["mean"]) @ z["W"]
        out.flush()
        np.save(os.path.join(_root(s), f"{_tag(s, name)}_rows.npy"), r)
        logger.info("%s/%s: %d rows -> %s", s, name, len(r), clip_path(s, name))


# ── Align encoded shards ──────────────────────────────────────────────────────

def align(split, backbone="streetclip"):
    ids = load(split)["id"]
    order = np.argsort(ids)
    sid = ids[order]
    with open(config.INGEST_CHECKPOINT) as f:
        done = set(json.load(f)["completed"].get(ckpt_key(split, backbone), []))
    # only checkpointed zips: a shard still being written by a concurrent encode is never read
    shards = sorted(s for s in glob.glob(os.path.join(shard_dir(split, backbone), "*.ids.npy"))
                    if os.path.basename(s).replace(".ids.npy", ".zip") in done)
    if not shards:
        raise FileNotFoundError(f"no completed encoded shards for {split}")

    matches = []
    for s in shards:
        q = np.load(s)
        j = np.minimum(np.searchsorted(sid, q), len(sid) - 1)
        ok = sid[j] == q
        if not ok.all():
            logger.warning("%s: %d ids missing from %s metadata", s, (~ok).sum(), split)
        matches.append((s, ok, order[j[ok]]))

    total = sum(len(r) for _, _, r in matches)
    dim = np.load(matches[0][0].replace(".ids.npy", ".npy"), mmap_mode="r").shape[1]
    out = np.lib.format.open_memmap(clip_path(split, backbone), mode="w+", dtype=np.float16, shape=(total, dim))
    rows, zips, pos = np.empty(total, np.int64), np.full(len(ids), -1, np.int8), 0
    for s, ok, r in matches:
        out[pos:pos + len(r)] = np.load(s.replace(".ids.npy", ".npy"), mmap_mode="r")[ok]
        rows[pos:pos + len(r)] = r
        zips[r] = int(os.path.basename(s).split(".")[0])
        pos += len(r)
    out.flush()
    np.save(os.path.join(config.FEAT_DIR, f"{_tag(split, backbone)}_rows.npy"), rows)
    np.save(os.path.join(config.DATA_DIR, "meta", f"{split}_zip{'' if backbone == 'streetclip' else '_' + backbone}.npy"), zips)
    logger.info("%s/%s: %d encoded rows (dim %d) from %d shards", split, backbone, total, dim, len(shards))


# ── Photo-free gallery grid ───────────────────────────────────────────────────

def grid(per=256, max_nn_km=50.0, seed=0):
    """Uniform points in each TEST_LEVEL cell's member bbox; leaf = leaf of nearest train row (Voronoi)."""
    from scipy.spatial import cKDTree

    tr = load("train")
    c = tr["cells"][:, TEST_L]
    order = np.argsort(c, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(c[order])) + 1]
    lat, lon = tr["lat"][order], tr["lon"][order]
    box = [f(a, starts) for f in (np.minimum.reduceat, np.maximum.reduceat) for a in (lat, lon)]
    (la0, lo0, la1, lo1), n = box, len(starts)
    pla, plo = 0.1 * (la1 - la0) + 0.005, 0.1 * (lo1 - lo0) + 0.005
    u = np.random.default_rng(seed).random((n, per, 2))
    glat = np.clip((la0 - pla)[:, None] + u[..., 0] * (la1 - la0 + 2 * pla)[:, None], -89.9, 89.9).ravel()
    glon = ((lo0 - plo)[:, None] + u[..., 1] * (lo1 - lo0 + 2 * plo)[:, None] + 180) % 360 - 180
    glon = glon.ravel()

    chord, nn = cKDTree(to_xyz(tr["lat"], tr["lon"])).query(to_xyz(glat, glon), workers=-1)
    keep = 2 * R_EARTH_KM * np.arcsin(np.clip(chord / 2, 0, 1)) <= max_nn_km
    np.savez(os.path.join(config.META_DIR, "grid.npz"), lat=glat[keep], lon=glon[keep],
             leaf=tr["cells"][nn[keep], LEAF], year=np.full(keep.sum(), config.GRID_YEAR, np.int16))
    logger.info("grid: %d cells x %d -> %d points within %.0f km of a train row", n, per, keep.sum(), max_nn_km)


# ── Novel-location split (SPLIT=blocks) ───────────────────────────────────────

BLOCK_KM, HOLD_FRAC, TUNE_FRAC, QT_DEPTH, GRID_REACH_KM = 100.0, 0.10, 0.10, 10, 120.0


def block_ids(lat, lon, km=BLOCK_KM):
    """Equal-area ~km x km blocks: fixed latitude bands, longitude cells per band scaled by cos(latitude)."""
    dlat = km / 111.195
    band = np.floor((np.asarray(lat) + 90) / dlat).astype(np.int64)
    n = np.maximum(1, np.round(360 * np.cos(np.radians(-90 + (band + 0.5) * dlat)) / dlat)).astype(np.int64)
    return band * 4096 + np.floor((np.asarray(lon) + 180) / 360 * n).astype(np.int64) % n


def pick_blocks(blk, cont, seed=0):
    """HOLD_FRAC of blocks per majority continent are held out, TUNE_FRAC of those tune; returns (held, tune) ids."""
    major = pd.DataFrame({"b": blk, "c": cont}).value_counts().reset_index().sort_values("count").drop_duplicates("b", keep="last")
    rng = np.random.default_rng(seed)
    held = np.concatenate([rng.permutation(g.b.to_numpy())[: int(np.ceil(HOLD_FRAC * len(g)))] for _, g in major.groupby("c")])
    return np.sort(held), np.sort(rng.permutation(held)[: int(np.ceil(TUNE_FRAC * len(held)))])


def qt_key(lat, lon, d):
    n = 1 << d
    i = np.minimum(((np.asarray(lat) + 90) / 180 * n).astype(np.int64), n - 1)
    return i * n + np.minimum(((np.asarray(lon) + 180) / 360 * n).astype(np.int64), n - 1)


def qt_counts(lat, lon):
    return [np.unique(qt_key(lat, lon, d), return_counts=True) for d in range(QT_DEPTH + 1)]


def qt_leaf(lat, lon, counts, n_max):
    """Leaf code d * 4^QT_DEPTH + key: the shallowest lat/lon quadrant holding <= n_max counted points."""
    code, done = np.zeros(len(lat), np.int64), np.zeros(len(lat), bool)
    for d, (uk, cnt) in enumerate(counts):
        k = qt_key(lat, lon, d)
        p = np.minimum(np.searchsorted(uk, k), len(uk) - 1)
        hit = ~done & ((np.where(uk[p] == k, cnt[p], 0) <= n_max) | (d == QT_DEPTH))
        code[hit], done = d * 4 ** QT_DEPTH + k[hit], done | hit
    return code


def qt_center(code):
    d, k = code // 4 ** QT_DEPTH, code % 4 ** QT_DEPTH
    n = np.left_shift(1, d)
    return -90 + (k // n + 0.5) * 180 / n, -180 + (k % n + 0.5) * 360 / n


def _lookup(raw, codes):
    p = np.minimum(np.searchsorted(raw, codes), len(raw) - 1)
    return np.where(raw[p] == codes, p, -1)


def land_grid():
    """Cached 2.56 km AE land grid: lat, lon and unit AE vectors at 80 m and 320 m."""
    from rasterio.warp import transform
    from geo_lookup import aef_index, dequantize
    ix, out = aef_index(config.AE_YEAR), [[], [], [], []]
    for p in sorted(glob.glob(os.path.join(config.DATA_DIR, "aef", f"c320_{config.AE_YEAR}", "*.npz"))):
        z, r = np.load(p), ix.iloc[int(os.path.basename(p)[:6])]
        if len(z["lat"]):
            x, y = (np.asarray(v) for v in transform("EPSG:4326", r.crs, z["lon"], z["lat"]))
            q320 = z["r320"][:, ((y - r.utm_south) // 320).astype(int), ((x - r.utm_west) // 320).astype(int)].T
            for o, v in zip(out, (z["lat"], z["lon"], z["q"], q320)):
                o.append(v)
    unit = lambda q: (lambda d: d / np.linalg.norm(d, axis=1, keepdims=True))(dequantize(np.concatenate(q)))
    return np.concatenate(out[0]), np.concatenate(out[1]), unit(out[2]), unit(out[3])


# ── AE cell pyramid ───────────────────────────────────────────────────────────

def pyr_path():
    return os.path.join(config.DATA_DIR, "aef", f"pyramid_{config.AE_YEAR}.npz")


def _cache_paths():
    return sorted(glob.glob(os.path.join(config.DATA_DIR, "aef", f"c320_{config.AE_YEAR}", "*.npz")))


def codebook(k=config.PYR_K, per_tile=256, stride=8, iters=25, seed=0):
    """Spherical k-means over 320 m AE pixels sampled from every stride-th cached tile."""
    from geo_lookup import dequantize
    rng, X = np.random.default_rng(seed), []
    for p in _cache_paths()[::stride]:
        a = np.load(p)["r320"].reshape(64, -1).T
        a = a[a[:, 0] != -128]
        if len(a):
            X.append(a[rng.choice(len(a), min(per_tile, len(a)), replace=False)])
    X = dequantize(np.concatenate(X))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        a = np.argmax(X @ C.T, 1)
        S = np.stack([np.bincount(a, X[:, j], minlength=k) for j in range(X.shape[1])], 1)
        empty = ~S.any(1)
        S[empty] = X[rng.choice(len(X), empty.sum(), replace=False)]
        C = S / np.linalg.norm(S, axis=1, keepdims=True)
    return C.astype(np.float32)


def cell_partials(lat, lon, V, codes, k):
    """Per pyramid depth: (sorted cell keys, summed unit vectors, code counts) over a batch of pixels."""
    out = []
    for d in config.PYR_DEPTHS:
        key = qt_key(lat, lon, d)
        o = np.argsort(key, kind="stable")
        ks = key[o]
        st = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
        out.append((ks[st], np.add.reduceat(V[o], st, axis=0),
                    np.add.reduceat(np.eye(k, dtype=np.int32)[codes[o]], st, axis=0)))
    return out


def _pyr_tile(job):
    p, C, crs, west, south = job
    from rasterio.warp import transform
    from geo_lookup import dequantize
    a = np.load(p)["r320"]
    rr, cc = np.nonzero(a[0] != -128)
    if not len(rr):
        return None
    lon, lat = (np.asarray(v) for v in transform(crs, "EPSG:4326", west + (cc + 0.5) * 320, south + (rr + 0.5) * 320))
    epsg = int(crs.split(":")[1])
    # own UTM zone only, so overlapping zone tiles do not count a pixel twice
    own = ((np.floor((lon + 180) / 6).astype(int) % 60) + 1 == epsg % 100) & ((lat >= 0) == (epsg < 32700))
    V = dequantize(a[:, rr[own], cc[own]].T)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return cell_partials(lat[own], lon[own], V, np.argmax(V @ C.T, 1), len(C))


def pyramid(workers=8):
    """Per land cell at each PYR_DEPTHS depth: mean AE (int8) + sqrt bag-of-environments histogram (uint8)."""
    from tqdm import tqdm
    from geo_lookup import aef_index, quantize
    C, ix = codebook(), aef_index(config.AE_YEAR)
    jobs = [(p, C, r.crs, float(r.utm_west), float(r.utm_south))
            for p, r in ((p, ix.iloc[int(os.path.basename(p)[:6])]) for p in _cache_paths())]
    parts = [[] for _ in config.PYR_DEPTHS]
    with ProcessPoolExecutor(workers) as ex:
        for res in tqdm(ex.map(_pyr_tile, jobs, chunksize=4), total=len(jobs), desc="pyramid"):
            for acc, r in zip(parts, res or ()):
                acc.append(r)
    out = {"depths": np.array(config.PYR_DEPTHS), "codebook": C}
    for d, acc in zip(config.PYR_DEPTHS, parts):
        u, s, h = (np.concatenate([a[i] for a in acc]) for i in range(3))
        o = np.argsort(u, kind="stable")
        st = np.flatnonzero(np.r_[True, u[o][1:] != u[o][:-1]])
        s, h = np.add.reduceat(s[o], st, axis=0), np.add.reduceat(h[o], st, axis=0)
        n = h.sum(1)
        out |= {f"k{d}": u[o][st], f"m{d}": quantize(s / np.linalg.norm(s, axis=1, keepdims=True)),
                f"h{d}": np.round(np.sqrt(h / n[:, None]) * 255).astype(np.uint8), f"n{d}": n}
        logger.info("pyramid depth %d: %d cells, median %d px", d, len(st), np.median(n))
    np.savez(pyr_path(), **out)


class Pyramid:
    """The AE pyramid as one table (depths stacked, then an all-masked row for cells without AE); rows via index()."""

    def __init__(self, path=None):
        z = np.load(path or pyr_path())
        self.depths = [int(d) for d in z["depths"]]
        self.keys = [z[f"k{d}"] for d in self.depths]
        self.off = np.cumsum([0] + [len(k) for k in self.keys])
        self.M = np.concatenate([*(z[f"m{d}"] for d in self.depths), np.full((1, 64), -128, np.int8)])
        h = [z[f"h{d}"] for d in self.depths]
        self.H = np.concatenate([*h, np.zeros((1, h[0].shape[1]), np.uint8)])

    def index(self, lat, lon):
        out = np.full((len(lat), len(self.depths)), self.off[-1], np.int64)
        for i, (d, k) in enumerate(zip(self.depths, self.keys)):
            q = qt_key(lat, lon, d)
            p = np.minimum(np.searchsorted(k, q), len(k) - 1)
            hit = k[p] == q
            out[hit, i] = self.off[i] + p[hit]
        return out


def blocks(seed=0):
    """Novel-location split: held-out blocks, train-only quadtree, gathered queries, leakage-free land-grid gallery."""
    from scipy.spatial import cKDTree
    from geo_lookup import CC_TO_CONTINENT_IDX
    off = os.path.join(config.DATA_DIR, "meta")
    tr, te, oc = (dict(np.load(os.path.join(off, f"{n}.npz"))) for n in ("train", "test", "cells"))
    cont = lambda c: np.array([CC_TO_CONTINENT_IDX.get(x, 7) for x in c])
    btr, bte = block_ids(tr["lat"], tr["lon"]), block_ids(te["lat"], te["lon"])
    held, tune = pick_blocks(np.r_[btr, bte], np.r_[cont(tr["country"]), cont(te["country"])], seed)
    hold = np.isin(btr, held)
    keep = ~hold

    glat, glon, g80, g320 = land_grid()
    reach = 2 * np.sin(GRID_REACH_KM / (2 * R_EARTH_KM))
    near = np.isfinite(cKDTree(to_xyz(tr["lat"][keep], tr["lon"][keep])).query(
        to_xyz(glat, glon), distance_upper_bound=reach, workers=-1)[0])
    glat, glon, g80, g320 = glat[near], glon[near], g80[near], g320[near]

    counts = qt_counts(tr["lat"][keep], tr["lon"][keep])
    vlat, vlon = np.r_[tr["lat"][keep], glat], np.r_[tr["lon"][keep], glon]
    raw = [np.unique(qt_leaf(vlat, vlon, counts, n)) for n in config.QT_LEVELS]
    assign = lambda la, lo: np.stack([_lookup(raw[l], qt_leaf(la, lo, counts, n)) for l, n in enumerate(config.QT_LEVELS)], 1)
    vc = assign(vlat, vlon)
    n_cells = [len(r) for r in raw]
    parents = [parent_map(vc[:, l], vc[:, l - 1], n_cells[l]) for l in range(1, N_LEVELS)]
    trc = assign(tr["lat"], tr["lon"])

    xyz, stats = to_xyz(tr["lat"][keep], tr["lon"][keep]), {}
    for l in range(N_LEVELS):
        cent, cnt, spread = cell_stats(xyz, trc[keep, l], n_cells[l])
        clat, clon = to_latlon(np.nan_to_num(cent))
        blat, blon = qt_center(raw[l])
        empty = cnt == 0
        stats |= {f"lat_{l}": np.where(empty, blat, clat), f"lon_{l}": np.where(empty, blon, clon), f"count_{l}": cnt,
                  f"raw_{l}": raw[l], f"tau_{l}": np.float64(np.median(spread[~empty]))}

    zt = zip_of("train")
    qtr = np.flatnonzero(hold & (zt >= 0) & (zt < 4))
    qte = np.flatnonzero(np.isin(bte, held))
    q = {k: np.r_[tr[k][qtr], te[k][qte]] for k in ("id", "lat", "lon", "country", "year", "aux_cat", "aux_reg")}
    q |= {"cells": assign(q["lat"], q["lon"]).astype(np.int32), "src": np.r_[np.zeros(len(qtr), np.int8), np.ones(len(qte), np.int8)],
          "src_row": np.r_[qtr, qte], "tune": np.isin(np.r_[btr[qtr], bte[qte]], tune)}

    os.makedirs(os.path.join(config.META_DIR, "ae"), exist_ok=True)
    np.savez(os.path.join(config.META_DIR, "train.npz"), **(tr | {"cells": trc.astype(np.int32), "holdout": hold}))
    np.savez(os.path.join(config.META_DIR, "test.npz"), **q)
    np.savez(os.path.join(config.META_DIR, "cells.npz"), anc=ancestors(parents, n_cells[LEAF]), n_cells=np.array(n_cells),
             n_aux=oc["n_aux"], reg_mean=oc["reg_mean"], reg_std=oc["reg_std"],
             **{f"parent_{l}": p for l, p in enumerate(parents, 1)}, **stats)
    np.savez(os.path.join(config.META_DIR, "grid.npz"), lat=glat, lon=glon, leaf=assign(glat, glon)[:, LEAF],
             year=np.full(len(glat), config.AE_YEAR, np.int16))
    np.save(os.path.join(config.META_DIR, "ae", "grid_r80.npy"), g80.astype(np.float16))
    np.save(os.path.join(config.META_DIR, "ae", "grid_r320.npy"), g320.astype(np.float16))

    src = [(0, qtr), (1, qte)]
    for bb in ("streetclip", "siglip2"):
        if not os.path.exists(os.path.join(config.FEAT_DIR, f"{_tag('test', bb)}.npy")):
            continue
        parts = []
        for split, (s, rows) in zip(("train", "test"), src):
            mem = np.load(os.path.join(config.FEAT_DIR, f"{_tag(split, bb)}.npy"), mmap_mode="r")
            inv = np.full(len(tr["lat"]) if s == 0 else len(te["lat"]), -1)
            inv[np.load(os.path.join(config.FEAT_DIR, f"{_tag(split, bb)}_rows.npy"))] = np.arange(len(mem))
            if (inv[rows] < 0).any():
                raise RuntimeError(f"{bb}: {(inv[rows] < 0).sum()} {split} query rows lack features")
            parts.append(np.asarray(mem[inv[rows]]))
        np.save(clip_path("test", bb), np.concatenate(parts))
        np.save(os.path.join(config.META_DIR, f"{_tag('test', bb)}_rows.npy"), np.arange(len(q["lat"])))
    for s in config.AE_SCALES:
        paths = [os.path.join(config.FEAT_DIR, "ae", f"{n}_r{s}.npy") for n in ("train", "test")]
        if all(map(os.path.exists, paths)):
            np.save(ae_path("test", s), np.concatenate([np.load(p, mmap_mode="r")[rows] for p, (_, rows) in zip(paths, src)]))
    logger.info("blocks: %d held-out of %d blocks (%d tune); train holdout %d rows (%.1f%%); queries %d (%d tune) "
                "[train %d + test %d]; grid %d pts; cells %s", len(held), len(np.unique(np.r_[btr, bte])), len(tune),
                hold.sum(), 100 * hold.mean(), len(q["lat"]), q["tune"].sum(), len(qtr), len(qte), len(glat),
                dict(zip(config.QT_LEVELS, n_cells)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "align", "grid", "blocks", "pyramid", "fuse"])
    ap.add_argument("--split", choices=["train", "test", "both"], default="both")
    ap.add_argument("--per", type=int, default=256)
    ap.add_argument("--backbone", choices=["streetclip", "siglip2"], default="streetclip")
    a = ap.parse_args()
    if a.cmd == "build":
        build()
    elif a.cmd == "align":
        for s in (["train", "test"] if a.split == "both" else [a.split]):
            align(s, a.backbone)
    elif a.cmd == "fuse":
        fuse(["train", "test"] if a.split == "both" else [a.split])
    elif a.cmd == "pyramid":
        pyramid()
    elif a.cmd == "blocks":
        if config.SPLIT == "official":
            raise SystemExit("run with SPLIT=blocks so outputs land in the blocks namespace")
        blocks()
    else:
        grid(per=a.per)
