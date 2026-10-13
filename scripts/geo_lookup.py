#!/usr/bin/env python3
"""
Geographic data and AlphaEarth embeddings from the public COG mirror (no Earth Engine quota).

Provides:
  - CC_TO_CONTINENT_NAME / CC_TO_CONTINENT_IDX — shared country->continent maps
  - extract() — AlphaEarth 64d at every point of train/test/grid, scales 80/160/320 m (resumable, threaded)
  - cache() / contexts() — per-tile 320 m rasters + land grid, then 3x3 @ 320 m + 3x3 @ 1280 m context per point

Usage:
  python geo_lookup.py                       # all sets, one pass over every touched tile (~11k tiles)
  python geo_lookup.py --limit-tiles 20      # smoke test
  python geo_lookup.py --cache               # 320 m rasters + 2.56 km land grid
  python geo_lookup.py --contexts --sets train grid   # location-tower inputs from the cache (no network)
"""

import argparse
import hashlib
import logging
import os
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)

# ── Continent mappings ────────────────────────────────────────────────────────

CONTINENT_NAMES = [
    'Africa', 'Asia', 'Europe', 'North America',
    'South America', 'Oceania', 'Antarctica',
]

CC_TO_CONTINENT_NAME: dict[str, str] = {}
CC_TO_CONTINENT_IDX:  dict[str, int] = {}

_CC_DATA = [
    (['DZ','AO','BJ','BW','BF','BI','CM','CV','CF','TD','KM','CG','CD','CI',
      'DJ','EG','GQ','ER','SZ','ET','GA','GM','GH','GN','GW','KE','LS','LR',
      'LY','MG','MW','ML','MR','MU','MA','MZ','NA','NE','NG','RW','ST','SN',
      'SC','SL','SO','ZA','SS','SD','TZ','TG','TN','UG','ZM','ZW','RE','YT'],
     'Africa', 0),
    (['AF','AM','AZ','BH','BD','BT','BN','KH','CN','CY','GE','IN','ID','IR',
      'IQ','IL','JP','JO','KZ','KW','KG','LA','LB','MY','MV','MN','MM','NP',
      'KP','OM','PK','PS','PH','QA','SA','SG','KR','LK','SY','TW','TJ','TH',
      'TL','TR','TM','AE','UZ','VN','YE','HK','MO'],
     'Asia', 1),
    (['AL','AD','AT','BY','BE','BA','BG','HR','CZ','DK','EE','FI','FR','DE',
      'GR','HU','IS','IE','IT','XK','LV','LI','LT','LU','MT','MD','MC','ME',
      'NL','MK','NO','PL','PT','RO','RU','SM','RS','SK','SI','ES','SE','CH',
      'UA','GB','VA','FO','GI','JE','GG','IM','AX'],
     'Europe', 2),
    (['AG','BS','BB','BZ','CA','CR','CU','DM','DO','SV','GD','GT','HT','HN',
      'JM','MX','NI','PA','KN','LC','VC','TT','US','PR','VI','GL','BM','AW',
      'CW','SX','MQ','GP','TC','KY','MS','AI'],
     'North America', 3),
    (['AR','BO','BR','CL','CO','EC','GY','PY','PE','SR','UY','VE','GF','FK'],
     'South America', 4),
    (['AU','FJ','KI','MH','FM','NR','NZ','PW','PG','WS','SB','TO','TV','VU',
      'NC','PF','GU','AS','CK','NU','TK','WF','MP'],
     'Oceania', 5),
    (['AQ'], 'Antarctica', 6),
]

for _ccs, _name, _idx in _CC_DATA:
    for _cc in _ccs:
        CC_TO_CONTINENT_NAME[_cc] = _name
        CC_TO_CONTINENT_IDX[_cc]  = _idx

NUM_CONTINENTS = 7

# ── AlphaEarth from the public COG mirror ─────────────────────────────────────
# Source Cooperative copy of GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL: per-UTM-zone 8192^2 x 64 int8 tiles stored
# bottom-up, overview k = 2^(k+1) x 10 m, built by Google's dequantize-sum-normalize (verified cos 0.9999).

OVERVIEW_80M = 2


def dequantize(q):
    """AEF int8 -> float32 (q/127.5)^2 * sign(q); -128 (masked) -> NaN."""
    q = np.asarray(q, np.float32)
    return np.where(q == -128, np.nan, np.square(q / 127.5) * np.sign(q))


def pooled(d, k):
    """Unit vectors [64, H/k, W/k] of non-overlapping k x k blocks of dequantized d; NaN where fully masked."""
    v = np.nansum(d.reshape(d.shape[0], d.shape[1] // k, k, d.shape[2] // k, k), axis=(2, 4))
    n = np.linalg.norm(v, axis=0, keepdims=True)
    return np.where(n > 0, v / np.where(n > 0, n, 1), np.nan)


def quantize(v):
    """Inverse of dequantize for pooled vectors: sign(v) sqrt(|v|) 127.5 -> int8; NaN -> -128 (masked)."""
    q = np.clip(np.round(np.sign(v) * np.sqrt(np.abs(v)) * 127.5), -127, 127)
    return np.where(np.isnan(v), -128, q).astype(np.int8)


def window(a, r, c):
    """3x3 neighbourhoods [n, 9, 64] of raster a [64, H, W] around pixels (r, c); outside the raster = masked."""
    ap = np.pad(a, ((0, 0), (1, 1), (1, 1)), constant_values=-128)
    return ap[:, r[:, None] + np.repeat(np.arange(3), 3), c[:, None] + np.tile(np.arange(3), 3)].transpose(1, 2, 0)


def aef_index(year):
    path = os.path.join(config.DATA_DIR, "aef", "aef_index.csv")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(f"{config.AEF_URL}/aef_index.csv", path + ".tmp")
        os.replace(path + ".tmp", path)
    ix = pd.read_csv(path, usecols=["path", "year", "crs", "utm_west", "utm_south", "utm_east", "utm_north"])
    return ix[ix.year == year].reset_index(drop=True)


def assign(lat, lon, ix):
    """Index row of the tile holding each point (-1 if none) and the point's UTM x, y in that tile's CRS."""
    from rasterio.warp import transform
    zone = (np.floor((np.asarray(lon) + 180) / 6).astype(int) % 60) + 1
    epsg = np.where(np.asarray(lat) >= 0, 32600, 32700) + zone
    f, x, y = np.full(len(lat), -1), np.full(len(lat), np.nan), np.full(len(lat), np.nan)
    for e in np.unique(epsg):
        m = np.flatnonzero(epsg == e)
        x[m], y[m] = transform("EPSG:4326", f"EPSG:{e}", lon[m], lat[m])
        t = ix[ix.crs == f"EPSG:{e}"]
        for j, w, s, ea, n in zip(t.index, t.utm_west, t.utm_south, t.utm_east, t.utm_north):
            f[m[(x[m] >= w) & (x[m] < ea) & (y[m] >= s) & (y[m] < n)]] = j
    return f, x, y


def sample_tile(url, x, y):
    """{scale_m: [n, 64]} from one tile's 80 m overview; coarser scales pool 2x2 / 4x4 overview pixels."""
    import rasterio
    with rasterio.open(url, overview_level=OVERVIEW_80M) as ds:
        d = dequantize(ds.read())
        r, c = (np.asarray(v) for v in rasterio.transform.rowcol(ds.transform, x, y))
    return {s: pooled(d, s // 80)[:, r // (s // 80), c // (s // 80)].T for s in config.AE_SCALES}


def cache_tile(url, crs, west, south, step=32):
    """320 m raster (int8, bottom-up like the source) + land grid every step x 80 m with its 80 m vector."""
    import rasterio
    from rasterio.warp import transform
    with rasterio.open(url, overview_level=OVERVIEW_80M) as ds:
        a = ds.read()
    r320 = quantize(pooled(dequantize(a), 4))
    rr, cc = (v.ravel() for v in np.meshgrid(*[np.arange(step // 2, n, step) for n in a.shape[1:]], indexing="ij"))
    q = a[:, rr, cc].T
    lon, lat = (np.asarray(v) for v in transform(crs, "EPSG:4326", west + (cc + 0.5) * 80, south + (rr + 0.5) * 80))
    epsg = int(crs.split(":")[1])
    # keep land points inside this tile's own UTM zone so overlapping zone tiles do not duplicate the grid
    own = ((np.floor((lon + 180) / 6).astype(int) % 60) + 1 == epsg % 100) & ((lat >= 0) == (epsg < 32700))
    keep = (q[:, 0] != -128) & own
    return r320, lat[keep], lon[keep], q[keep]


def _with_backoff(fn, *args):
    for attempt in range(config.AEF_RETRIES):
        try:
            return fn(*args)
        except Exception as e:
            if attempt == config.AEF_RETRIES - 1:
                raise
            delay = min(60.0, 2.0 ** attempt) * random.uniform(0.5, 1.5)
            logger.warning("AEF read attempt %d failed (%s); retry in %.1fs", attempt + 1, str(e)[:120], delay)
            time.sleep(delay)


def claim(out_dir, sig):
    """Bind out_dir's per-tile files to one point set; refuse to mix results from a different one."""
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "SIGNATURE")
    if os.path.exists(p) and open(p).read() != sig:
        raise RuntimeError(f"{out_dir} was sampled for different points; per-tile files would map to wrong rows. "
                           "Delete the directory to resample.")
    with open(p, "w") as fh:
        fh.write(sig)


def extract(sets, limit_tiles=0):
    """One pass over every tile touched by the union of point sets -> feats/ae/{set}_r{scale}.npy (fp16, NaN = none)."""
    import bins
    metas = {s: bins.load(s) for s in sets}
    lat, lon = (np.concatenate([metas[s][k] for s in sets]) for k in ("lat", "lon"))
    off = np.cumsum([0] + [len(metas[s]["lat"]) for s in sets])
    out_dir = os.path.join(config.FEAT_DIR, "ae", f"aef{config.AE_YEAR}")
    claim(out_dir, hashlib.sha1(np.ascontiguousarray(np.c_[lat, lon], np.float64).tobytes()
                                + str(config.AE_YEAR).encode()).hexdigest())

    ix = aef_index(config.AE_YEAR)
    f, x, y = assign(lat, lon, ix)
    order = np.argsort(f, kind="stable")
    cut = np.flatnonzero(np.diff(f[order])) + 1
    groups = {int(f[g[0]]): g for g in np.split(order, cut) if f[g[0]] >= 0}
    path = lambda j: os.path.join(out_dir, f"{j:06d}.npz")
    todo = [j for j in groups if not os.path.exists(path(j))][: limit_tiles or None]
    logger.info("%d points, %.2f%% inside a %d tile; %d/%d tiles to read", len(lat), 100 * (f >= 0).mean(),
                config.AE_YEAR, len(todo), len(groups))

    def run(j):
        m = groups[j]
        v = _with_backoff(sample_tile, "/vsicurl/" + config.AEF_URL + "/" + ix.path[j].split("/annual/", 1)[1], x[m], y[m])
        with open(path(j) + ".tmp", "wb") as fh:
            np.savez(fh, idx=m, **{f"s{s}": a.astype(np.float16) for s, a in v.items()})
        os.replace(path(j) + ".tmp", path(j))
        return len(m)

    os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff",
                      GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_TIMEOUT="120")
    t0, done = time.time(), 0
    with ThreadPoolExecutor(config.AEF_THREADS) as ex:
        futs = [ex.submit(run, j) for j in todo]
        for fu in (bar := tqdm(as_completed(futs), total=len(futs), desc="tiles")):
            done += fu.result()
            bar.set_postfix(pts_s=f"{done / (time.time() - t0):.0f}")

    full = {s: np.full((len(lat), 64), np.nan, np.float16) for s in config.AE_SCALES}
    for j in groups:
        if os.path.exists(path(j)):
            with np.load(path(j)) as z:
                for s in config.AE_SCALES:
                    full[s][z["idx"]] = z[f"s{s}"]
    for i, name in enumerate(sets):
        for s in config.AE_SCALES:
            part = full[s][off[i]:off[i + 1]]
            np.save(os.path.join(config.FEAT_DIR, "ae", f"{name}_r{s}.npy"), part)
            logger.info("%s_r%d: %d rows, valid %.2f%%", name, s, len(part), 100 * (~np.isnan(part[:, 0])).mean())


def cache(sets, limit_tiles=0):
    """Per touched tile: 320 m int8 raster (location-tower context) + 2.56 km land grid (photo-free gallery)."""
    import bins
    lat, lon = (np.concatenate([bins.load(s)[k] for s in sets]) for k in ("lat", "lon"))
    ix = aef_index(config.AE_YEAR)
    tiles = np.unique(assign(lat, lon, ix)[0])
    tiles = tiles[tiles >= 0]
    out_dir = os.path.join(config.DATA_DIR, "aef", f"c320_{config.AE_YEAR}")
    os.makedirs(out_dir, exist_ok=True)
    path = lambda j: os.path.join(out_dir, f"{j:06d}.npz")
    todo = [int(j) for j in tiles if not os.path.exists(path(j))][: limit_tiles or None]
    logger.info("cache: %d/%d tiles to read -> %s", len(todo), len(tiles), out_dir)

    def run(j):
        r = ix.iloc[j]
        out = _with_backoff(cache_tile, "/vsicurl/" + config.AEF_URL + "/" + r.path.split("/annual/", 1)[1],
                            r.crs, float(r.utm_west), float(r.utm_south))
        with open(path(j) + ".tmp", "wb") as fh:
            np.savez(fh, r320=out[0], lat=out[1], lon=out[2], q=out[3])
        os.replace(path(j) + ".tmp", path(j))
        return len(out[1])

    os.environ.update(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff",
                      GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2", GDAL_HTTP_TIMEOUT="120")
    n = 0
    with ThreadPoolExecutor(config.AEF_THREADS) as ex:
        for fu in tqdm(as_completed([ex.submit(run, j) for j in todo]), total=len(todo), desc="cache"):
            n += fu.result()
    logger.info("cache: %d land grid points in %d new tiles", n, len(todo))


def contexts(sets):
    """Location-tower input per point: 3x3 @ 320 m + 3x3 @ 1280 m (block-aligned) from the cache -> int8 [N, 18, 64]."""
    import bins
    ix, cdir = aef_index(config.AE_YEAR), os.path.join(config.DATA_DIR, "aef", f"c320_{config.AE_YEAR}")
    for name in sets:
        m = bins.load(name)
        f, x, y = assign(m["lat"], m["lon"], ix)
        out_path = bins.ctx_path(name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out = np.lib.format.open_memmap(out_path + ".tmp.npy", mode="w+", dtype=np.int8, shape=(len(f), 18, 64))
        out[:] = -128
        order = np.argsort(f, kind="stable")
        missing = 0
        for g in tqdm(np.split(order, np.flatnonzero(np.diff(f[order])) + 1), desc=f"ctx {name}"):
            j, p = int(f[g[0]]), os.path.join(cdir, f"{int(f[g[0]]):06d}.npz")
            if j < 0 or not os.path.exists(p):
                missing += len(g)
                continue
            r320 = np.load(p)["r320"]
            r, c = ((y[g] - ix.utm_south[j]) // 320).astype(int), ((x[g] - ix.utm_west[j]) // 320).astype(int)
            out[g] = np.concatenate([window(r320, r, c), window(quantize(pooled(dequantize(r320), 4)), r // 4, c // 4)], 1)
        out.flush()
        del out
        os.replace(out_path + ".tmp.npy", out_path)
        logger.info("ctx %s: %d points, %d (%.2f%%) without a cached tile -> %s", name, len(f), missing,
                    100 * missing / max(len(f), 1), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=["train", "test", "grid"], choices=["train", "test", "grid"])
    ap.add_argument("--limit-tiles", type=int, default=0, help="read only N tiles (smoke test)")
    ap.add_argument("--cache", action="store_true", help="cache 320 m rasters + 2.56 km land grid instead")
    ap.add_argument("--contexts", action="store_true", help="build location-tower contexts from the cache")
    a = ap.parse_args()
    if a.contexts:
        contexts(a.sets)
    else:
        (cache if a.cache else extract)(a.sets, a.limit_tiles)
