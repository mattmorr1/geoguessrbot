#!/usr/bin/env python3
"""
Evaluation: product-of-experts scoring, geodesic metrics, bootstrap CIs, stratified breakdowns.

  score(j) = log p(leaf_j | img) - log n_leaf_j + kappa * cos(ae_hat, ae_j) [+ lambda * cos(clip, clip_j)]
  T0/T2 swap the AE term for the learned two-tower score kappa * cos(img_emb, tower(location_j)), same galleries as G0/G2.
  HT2/HG2 mix a coarse-level prior (spread over its gallery points) with the leaf prior at weight beta (beta=1 ~ T2/G2);
  level and beta are tuned on val. UT2 drops the prior entirely (photo-free map).
  BT2 is the union: photos keep lambda * cos(clip) and gain the tower term; map points get the tower term + a tuned offset.

Usage:
  python eval.py tune runs/r4/best.pt          # grid for kappa/lambda/R on val (10% test tune slice) -> tuned.json
  python eval.py test runs/r4/best.pt          # single pass on the disjoint 90% of the official test split
  python eval.py tune runs/r4/best.pt --prior-ckpt runs/r1/best.pt   # decoupled -> runs/r4_prior-r1/
  python eval.py tune runs/t_ae/best.pt --methods B2,HT2,UT2 --tag coarse   # subset -> runs/t_ae@coarse/
  python eval.py regions runs/t_ae/best.pt     # localization inside the true cell (no tuning) -> regions_val.json
  python eval.py report                        # aggregate runs -> results/report.md
"""

import argparse
import functools
import glob
import json
import logging
import math
import os

import numpy as np
import torch

import bins
import config
from geo_lookup import CC_TO_CONTINENT_NAME
from model import device, load_bridge, pyramid_feats

logger = logging.getLogger(__name__)

THRESH_KM = (1, 5, 10, 25, 200, 750, 2500)
TOP_M = 256          # leaves kept per query for the soft prior (mass beyond is negligible)
HITS = 64
GPU_GALLERY_BYTES = 4 << 30   # larger gallery vectors stay in host RAM and stream through retrieve chunk by chunk
KAPPAS = (2, 5, 10, 20, 40, 80, 160, 320, 640, 1280)
ALPHAS = (0.0, 0.25, 0.5, 1.0)   # coarse-level weight in the leaf prior
RADII = (0, 5, 25, 100)
COARSE_LEVELS = (1, 2)           # candidate prior levels for H* methods (25000 / 12500)
BETAS = (0.0, 0.25, 0.5, 0.75)   # leaf-prior weight in H* methods
USE_AMP = device == "cuda"
FEATS = {"streetclip": ("streetclip",), "siglip2": ("siglip2",), "fusion": ("streetclip", "siglip2"),
         "fusion768": ("fusion768",)}   # fusion768: PCA of the fusion (bins.py fuse), StreetCLIP-sized


# ── Metrics ───────────────────────────────────────────────────────────────────

def geoscore(d_km):
    return 5000 * np.exp(-np.asarray(d_km) / 1492.7)


def distance_metrics(d):
    return {"median_km": float(np.median(d)), "mean_km": float(d.mean()), "geoscore": float(geoscore(d).mean()),
            **{f"acc_{t}km": float((d <= t).mean()) for t in THRESH_KM}}


def bootstrap(stat, *arrays, n=1000, seed=0):
    rng, N = np.random.default_rng(seed), len(arrays[0])
    vals = [stat(*(a[i] for a in arrays)) for i in (rng.integers(0, N, N) for _ in range(n))]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


OBJECTIVES = {"geoscore": lambda d: geoscore(d).mean(), "acc25": lambda d: (d <= 25).mean()}
CO_PRIMARY = {"median_km": np.median, "geoscore": lambda x: geoscore(x).mean(), "acc_25km": lambda x: (x <= 25).mean()}


def with_cis(d, base=None):
    out = {f"{k}_ci": bootstrap(f, d) for k, f in CO_PRIMARY.items()}
    if base is not None:
        out |= {f"delta_{k}_ci": bootstrap(lambda a, b, f=f: f(a) - f(b), d, base) for k, f in CO_PRIMARY.items()}
    return out


def admin_labels(lat, lon):
    import reverse_geocoder as rg
    res = rg.search(list(zip(np.asarray(lat, float), np.asarray(lon, float))), verbose=False)
    return (np.array([r["cc"] for r in res]), np.array([r["cc"] + "/" + r["admin1"] for r in res]),
            np.array([r["cc"] + "/" + r["admin1"] + "/" + r["name"] for r in res]))


def ece(conf, correct, bins_=15):
    idx = np.minimum((conf * bins_).astype(int), bins_ - 1)
    w = np.bincount(idx, minlength=bins_) / len(conf)
    acc = np.bincount(idx, weights=correct, minlength=bins_) / np.maximum(np.bincount(idx, minlength=bins_), 1)
    cf = np.bincount(idx, weights=conf, minlength=bins_) / np.maximum(np.bincount(idx, minlength=bins_), 1)
    return float((w * np.abs(acc - cf)).sum())


# ── Stage 1: per-query model outputs ──────────────────────────────────────────

def leaf_logp(cell_logits, anc, alpha=1.0):
    """Normalized leaf log posterior: log p_leaf + alpha * sum_{coarser l} log p_l(anc_l(leaf)).
    alpha=1 is the plain product of experts (overconfident), alpha=0 the finest head alone; tuned on val."""
    lp = torch.log_softmax(cell_logits[-1].float(), -1) + alpha * sum(
        torch.log_softmax(lg.float(), -1)[:, anc[:, l]] for l, lg in enumerate(cell_logits[:-1]))
    return lp - torch.logsumexp(lp, -1, keepdim=True)


@torch.no_grad()
def encode_queries(model, X, anc, m=TOP_M, bs=2048, alpha=1.0):
    anc_t = torch.as_tensor(anc, device=device)
    parts = {k: [] for k in ("ae", "loc", "top_idx", "top_lp", "flat", "head_top5", "p1000", "c1000",
                             *(f"lp{l}" for l in COARSE_LEVELS))}
    for i in range(0, len(X), bs):
        x = X[i:i + bs]
        x = (x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
            o = model(x)
        lp = leaf_logp(o["cells"], anc_t, alpha)
        v, ix = lp.topk(m, dim=1)
        marg = torch.zeros(len(x), int(anc_t[:, bins.TEST_L].max()) + 1, device=device)
        marg.scatter_add_(1, anc_t[:, bins.TEST_L].expand(len(x), -1), lp.exp())
        p1, c1 = marg.max(1)
        for k, t in (("ae", o["ae"].half()), ("loc", o["loc"].half() if "loc" in o else torch.zeros(len(x), 0)),
                     ("top_idx", ix.int()), ("top_lp", v), ("flat", o["cells"][-1].argmax(1)),
                     ("head_top5", torch.stack([lg.topk(5, 1).indices for lg in o["cells"][:bins.TEST_L + 1]], 1)),
                     ("p1000", p1), ("c1000", c1),
                     *((f"lp{l}", torch.log_softmax(o["cells"][l].float(), -1).half()) for l in COARSE_LEVELS)):
            parts[k].append(t.cpu())
    q = {k: torch.cat(v) for k, v in parts.items()}
    q["b0"] = q["top_idx"][:, 0].long()
    return q


def quick_val(model, X, gt_lat, gt_lon, anc, cells, ae_true=None, ae_ok=None):
    """Per-epoch validation used by train.py: GeoScore of B0 and mean AE cosine."""
    q = encode_queries(model, X, anc, m=1)
    b0 = q["b0"].numpy()
    d = bins.haversine(gt_lat, gt_lon, cells[f"lat_{bins.LEAF}"][b0], cells[f"lon_{bins.LEAF}"][b0])
    out = {"val_geoscore": float(geoscore(d).mean()), "val_median_km": float(np.median(d))}
    if ae_true is not None and ae_ok.any():
        out["val_ae_cos"] = float((q["ae"].float().numpy()[ae_ok] * ae_true[ae_ok]).sum(1).mean())
    return out


class Decoupled(torch.nn.Module):
    """Cell logits from a prior model, AE prediction (and tower embedding) from another: neither can degrade the prior."""

    def __init__(self, prior, ae):
        super().__init__()
        self.prior, self.ae, self.tower = prior, ae, ae.tower

    def forward(self, x):
        return {**self.prior(x), **{k: v for k, v in self.ae(x).items() if k in ("ae", "loc")}}


# ── Stage 2: exact retrieval over a GPU-resident gallery ──────────────────────

def to_dev(a, dtype=None):
    return torch.as_tensor(np.asarray(a), device=device, dtype=dtype)


def normalize16(v, idx=None, dev=device, chunk=1 << 18):
    """L2-normalized fp16 copy on dev, streamed in chunks so a 2M x 768 memmap never materializes in fp32."""
    n = len(v) if idx is None else len(idx)
    out = torch.empty((n, v.shape[1]), dtype=torch.float16, device=dev)
    for i in range(0, n, chunk):
        part = v[i:i + chunk] if idx is None else v[idx[i:i + chunk]]
        out[i:i + chunk] = torch.nn.functional.normalize(to_dev(part).float(), dim=-1).half()
    return out


def gallery_dev(n, dim):
    return device if n * dim * 2 <= GPU_GALLERY_BYTES else "cpu"


@torch.no_grad()
def _sim(qf, v, lo, hi):
    """qf @ v[lo:hi].T; gallery rows past len(v) (points without this feature, e.g. map points in BT2) score 0."""
    if len(v) >= hi:
        return (qf @ v[lo:hi].T).float()
    s = torch.zeros(len(qf), hi - lo, device=qf.device)
    if lo < len(v):
        s[:, :len(v) - lo] = (qf @ v[lo:].T).float()
    return s


def retrieve(qv, prior, gal, w, n_leaf, k=HITS, qb=512, gb=1 << 18):
    """
    Top-k hits per query. qv/gal['vec'] share feature keys; w maps key -> weight.
    prior: (top_idx, top_lp), a callable qsl -> dense [qb, n_leaf] additive term (count-normalized), or None.
    Gallery chunks are the outer loop, so a host-resident gallery crosses to the GPU once per call.
    """
    leaf, G = gal["leaf"], len(gal["leaf"])
    logn = torch.log(torch.bincount(leaf, minlength=n_leaf).clamp(min=1).float())
    N = len(next(iter(qv.values())))

    def leaf_prior(qsl):
        if callable(prior):
            return prior(qsl)
        if prior is None:
            return None
        idx = prior[0][qsl].to(device).long()
        P = torch.full((len(idx), n_leaf), -math.inf, device=device)
        return P.scatter_(1, idx, prior[1][qsl].to(device) - logn[idx])

    hit_s = torch.full((N, k), -math.inf, device=device)
    hit_i = torch.zeros((N, k), dtype=torch.long, device=device)
    for gs in range(0, G, gb):
        lf, chunk = leaf[gs:gs + gb], {f: gal["vec"][f][gs:gs + gb].to(device) for f in w}
        for qs in range(0, N, qb):
            qsl = slice(qs, qs + qb)
            s = sum(wt * _sim(qv[f][qsl].to(device), chunk[f], 0, len(lf)) for f, wt in w.items())
            P = leaf_prior(qsl)
            if P is not None:
                s = s + P.index_select(1, lf)
            cs, ci = s.topk(min(k, s.shape[1]), dim=1)
            hit_s[qsl], sel = torch.cat([hit_s[qsl], cs], 1).topk(k, dim=1)
            hit_i[qsl] = torch.cat([hit_i[qsl], ci + gs], 1).gather(1, sel)
    return hit_s.cpu(), hit_i.cpu()


def refine(hit_s, hit_i, gal_xyz, radius_km, fallback):
    """Score-weighted spherical mean of hits within radius_km of the best hit (0 -> best hit only)."""
    xyz = gal_xyz[hit_i]
    best = hit_s.argmax(1)
    bx = xyz[torch.arange(len(xyz)), best]
    near = (xyz * bx[:, None]).sum(-1) >= math.cos(radius_km / bins.R_EARTH_KM) - 1e-6
    wts = torch.exp(hit_s - hit_s.max(1, keepdim=True).values).nan_to_num(0) * near
    ok = torch.isfinite(hit_s).any(1).numpy()
    lat, lon = bins.to_latlon((wts[..., None] * xyz).sum(1).double().numpy() + ~ok[:, None])
    return np.where(ok, lat, fallback[0]), np.where(ok, lon, fallback[1])


# ── Data assembly ─────────────────────────────────────────────────────────────

class Env:
    """Lazily loaded meta, features, and galleries; queries come only from the test split.
    feats picks the backbone(s); rows='common' restricts train/gallery/queries to rows every backbone encoded (bake-off)."""

    def __init__(self, ae_radius, feats="streetclip", rows="all"):
        self.r, self.bbs = ae_radius, FEATS[feats]
        self.common = FEATS["fusion"] if rows == "common" else self.bbs
        self.cells = bins.load("cells")
        self.anc = self.cells["anc"]
        self.n_leaf = len(self.anc)
        self.train = bins.load("train")
        self.rows = self._rows("train")
        self.clip = bins.load_feats("train", self.rows, self.bbs)
        self.ae_train = self._ae("train")
        self.pyramid = None
        self.min_km = 0.0

    @functools.cached_property
    def _seen_tree(self):
        from scipy.spatial import cKDTree
        seen = ~self.train["holdout"] if "holdout" in self.train else slice(None)
        return cKDTree(bins.to_xyz(self.train["lat"][seen], self.train["lon"][seen]))

    def nn_km(self, lat, lon):
        """Great-circle km to the nearest non-held-out train row (photo or not)."""
        chord, _ = self._seen_tree.query(bins.to_xyz(lat, lon), workers=-1)
        return 2 * bins.R_EARTH_KM * np.arcsin(np.clip(chord / 2, 0, 1))

    def _rows(self, split):
        enc = [bins.encoded_rows(split, b) for b in self.common]
        return enc[0] if len(enc) == 1 else functools.reduce(np.intersect1d, enc)

    def _ae(self, name):
        p = bins.ae_path(name, self.r)
        return np.load(p, mmap_mode="c") if os.path.exists(p) else None

    def ctx(self, name):
        p = bins.ctx_path(name)
        return np.load(p, mmap_mode="r") if os.path.exists(p) else None

    @torch.no_grad()
    def embed(self, tower, name, idx, lat, lon, bs=1 << 16):
        """Location-tower embeddings (fp16, host) for rows idx of point set `name`."""
        ctx = self.ctx(name) if tower.ctx else None
        if tower.ctx and ctx is None:
            raise FileNotFoundError(f"no AE contexts for {name}")
        if tower.pyr and self.pyramid is None:
            self.pyramid = bins.Pyramid()
        pt = (to_dev(self.pyramid.M), to_dev(self.pyramid.H), self.pyramid.index(lat, lon)) if tower.pyr else None
        xyz = bins.to_xyz(lat, lon)
        out = [tower(None if ctx is None else to_dev(ctx[idx[i:i + bs]]), to_dev(xyz[i:i + bs]),
                     None if pt is None else pyramid_feats(pt[0], pt[1], to_dev(pt[2][i:i + bs]))).half().cpu()
               for i in range(0, len(idx), bs)]
        return torch.cat(out).numpy() if out else np.zeros((0, tower.out_dim), np.float16)

    def queries(self, split):
        """val = test tune slice, test = the disjoint rest (split by sequence in bins.build); foldK = K-th hash fold."""
        meta, rows = bins.load("test"), self._rows("test")
        if split.startswith("fold"):
            rows = rows[bins.test_folds()[rows] == int(split[4:])]
        else:
            rows = rows[meta["tune"][rows] == (split == "val")]
        if self.min_km:
            rows = rows[self.nn_km(meta["lat"][rows], meta["lon"][rows]) > self.min_km]
            logger.info("queries %s: %d beyond %.0f km of train", split, len(rows), self.min_km)
        X, ae = bins.load_feats("test", rows, self.bbs), self._ae("test")
        ae_true = None if ae is None else np.asarray(ae[rows], np.float32)
        return {"X": X, "lat": meta["lat"][rows], "lon": meta["lon"][rows], "cells": meta["cells"][rows],
                "country": meta["country"][rows], "land_cover": meta["aux_cat"][rows, 2], "ae_true": ae_true}

    def gallery(self, name, tower=None):
        """photos | fused | G0-G2 (AE vectors) | T0, T2 (tower embeddings of the G0 / G2 point sets)."""
        tr, rows = self.train, self.rows
        free = ~tr["holdout"] if "holdout" in tr else np.ones(len(tr["lat"]), bool)
        if name == "photos":
            keep = np.flatnonzero(free[rows])
            return self._gal(tr["lat"][rows[keep]], tr["lon"][rows[keep]], tr["cells"][rows[keep], bins.LEAF],
                             np.zeros(len(keep), np.int8), clip=(self.clip, keep))
        if self.ae_train is None:
            raise FileNotFoundError(f"no train AE at r={self.r}")
        ok = ~np.isnan(np.asarray(self.ae_train[:, 0], np.float32)) & free
        g0 = rows[ok[rows]]
        if name == "fused":
            return self._gal(tr["lat"][g0], tr["lon"][g0], tr["cells"][g0, bins.LEAF], np.zeros(len(g0), np.int8),
                             ae=(self.ae_train, g0), clip=(self.clip, np.flatnonzero(ok[rows])))
        kind, lvl = name[-2], name[-1]
        if kind == "T" and tower is None:
            raise FileNotFoundError(f"{name} needs a checkpoint with a location tower")
        sel, src = g0, np.zeros(len(g0), np.int8)
        if lvl in ("1", "2"):
            enc = np.zeros(len(ok), bool)
            enc[rows] = True
            extra = np.flatnonzero(~enc & ok)  # empty once every train row is encoded
            sel, src = np.concatenate([g0, extra]), np.r_[src, np.ones(len(extra), np.int8)]
        lat, lon, leaf = tr["lat"][sel], tr["lon"][sel], tr["cells"][sel, bins.LEAF]
        vec = (np.asarray(self.ae_train[sel]) if kind == "G" else self.embed(tower, "train", sel, lat, lon))
        if lvl == "2":
            g, gae = bins.load("grid"), self._ae("grid")
            if gae is None:
                raise FileNotFoundError("grid AE not sampled")
            gsel = np.flatnonzero(~np.isnan(np.asarray(gae[:, 0], np.float32)))
            glat, glon = g["lat"][gsel], g["lon"][gsel]
            gvec = np.asarray(gae)[gsel] if kind == "G" else self.embed(tower, "grid", gsel, glat, glon)
            lat, lon, leaf, vec = np.r_[lat, glat], np.r_[lon, glon], np.r_[leaf, g["leaf"][gsel]], np.concatenate([vec, gvec])
            src = np.r_[src, np.full(len(gsel), 2, np.int8)]
        vecs = {"ae" if kind == "G" else "loc": (vec, None)}
        if name == "BT2":   # photos (first len(g0) points) also carry clip; map points carry the offset flag
            vecs |= {"clip": (self.clip, np.flatnonzero(ok[rows])), "flag": ((src > 0).astype(np.float16)[:, None], None)}
        return self._gal(lat, lon, leaf, src, **vecs)

    def _gal(self, lat, lon, leaf, src, **vec):
        """src: 0 = encoded photo location, 1 = photo-free train location, 2 = grid point."""
        logger.info("gallery: %d points (%s) src counts %s", len(lat), ",".join(vec), np.bincount(src, minlength=3))
        return {"xyz": torch.as_tensor(bins.to_xyz(lat, lon), dtype=torch.float32), "lat": lat, "lon": lon, "src": src,
                "leaf": to_dev(leaf, torch.long),
                "vec": {k: normalize16(v, idx, gallery_dev(len(v) if idx is None else len(idx), v.shape[1]))
                        for k, (v, idx) in vec.items()}}


# ── Methods ───────────────────────────────────────────────────────────────────

GALLERY_OF = {"B1": "photos", "B2": "photos", "G0": "G0", "G1": "G1", "G2": "G2", "Fused": "fused", "T0": "T0", "T2": "T2",
              "HT2": "T2", "UT2": "T2", "HG2": "G2", "BT2": "BT2"}
DIAG = {"G0": "ae", "G2": "ae", "T0": "loc", "T2": "loc"}   # within-cell MRR; *2 adds land-grid candidates (blocks)


def method_hits(name, q, gal, qv, p, env):
    prior = (q["top_idx"], q["top_lp"])
    if name == "B1":
        return retrieve(qv, None, gal, {"clip": 1.0}, env.n_leaf)
    if name == "B2":
        return retrieve(qv, prior, gal, {"clip": p["lambda"]}, env.n_leaf)
    if name == "Fused":
        return retrieve(qv, prior, gal, {"ae": p["kappa"], "clip": p["lambda"]}, env.n_leaf)
    if name == "BT2":
        return retrieve(qv, prior, gal, {"clip": p["lambda"], "loc": p["kappa"], "flag": p["offset"]}, env.n_leaf)
    if name[0] in "HU":
        w = {"ae" if name[1] == "G" else "loc": p["kappa"]}
        if name[0] == "U":
            return retrieve(qv, None, gal, w, env.n_leaf)
        return retrieve(qv, mixed_prior(q, p, env, gal), gal, w, env.n_leaf)
    return retrieve(qv, prior, gal, {"ae" if name[0] == "G" else "loc": p["kappa"]}, env.n_leaf)


def query_vecs(q, X):
    return {"ae": q["ae"], "loc": q["loc"], "clip": normalize16(X), "flag": torch.ones(len(X), 1, dtype=torch.float16)}


def mixed_prior(q, p, env, gal):
    """(1-beta) [log p_level(cell) - log n_cell] + beta [log p_leaf - log n_leaf] per leaf, n = gallery points.
    Leaves outside the query's top-M leaf list take its M-th log prob."""
    anc = to_dev(env.anc[:, p["level"]], torch.long)
    n_leaf = torch.bincount(gal["leaf"], minlength=env.n_leaf).clamp(min=1).float().log()
    n_cell = torch.bincount(anc[gal["leaf"]], minlength=int(env.cells["n_cells"][p["level"]])).clamp(min=1).float().log()
    beta = p.get("beta", 0.0)

    def prior(qsl):
        coarse = q[f"lp{p['level']}"][qsl].to(device).float()[:, anc] - n_cell[anc]
        top_lp = q["top_lp"][qsl].to(device).float()
        leaf = top_lp[:, -1:].repeat(1, env.n_leaf).scatter_(1, q["top_idx"][qsl].to(device).long(), top_lp)
        return (1 - beta) * coarse + beta * (leaf - n_leaf)
    return prior


def cell_centroid(env, leaf):
    return env.cells[f"lat_{bins.LEAF}"][leaf], env.cells[f"lon_{bins.LEAF}"][leaf]


def tune(model, ck, env, run_dir, only=None, val="val", objective="geoscore"):
    """Val-only grid: prior alpha by B0 objective, then each method keeps the (weight, R) maximizing the val objective."""
    Q, obj = env.queries(val), OBJECTIVES[objective]
    tag = {"objective": objective, "min_km": env.min_km}
    b0_d = lambda q: bins.haversine(Q["lat"], Q["lon"], *cell_centroid(env, q["b0"].numpy()))
    d_alpha = {a: b0_d(encode_queries(model, Q["X"], env.anc, m=1, alpha=a)) for a in ALPHAS}
    by_alpha = {a: float(obj(d)) for a, d in d_alpha.items()}
    alpha = max(by_alpha, key=by_alpha.get)
    logger.info("prior alpha -> B0 val %s %s; using alpha=%s", objective, by_alpha, alpha)
    q = encode_queries(model, Q["X"], env.anc, alpha=alpha)
    qv = query_vecs(q, Q["X"])
    methods = (["B1", "B2"] + (["G0", "G1", "G2", "Fused", "HG2"] if ck["args"]["w_ae"] > 0 else [])
               + (["T0", "T2", "HT2", "UT2", "BT2"] if model.tower is not None else []))
    methods = [m for m in methods if not only or m in only]
    fb = cell_centroid(env, q["b0"].numpy())
    best = {}
    for m in methods:
        if (m == "Fused" and "G0" not in best) or (m == "BT2" and "B2" not in best):
            continue
        try:
            gal = env.gallery(GALLERY_OF[m], model.tower)
        except FileNotFoundError as e:
            logger.warning("skip %s: %s", m, e)
            continue
        key = "lambda" if m in ("B2", "Fused") else "kappa"
        grid = [{}] if m == "B1" else [{key: v} for v in KAPPAS]
        if m[0] == "H":
            grid = [{"kappa": v, "level": l, "beta": b} for l in COARSE_LEVELS for b in BETAS for v in KAPPAS]
        if m == "Fused":
            grid = [{"kappa": best["G0"]["kappa"], "lambda": v} for v in KAPPAS]
        if m == "BT2":   # offset in units of the photo weight: map points compete with photos of clip cos ~ c
            lam = best["B2"]["lambda"]
            grid = [{"lambda": lam, "kappa": v, "offset": lam * c} for c in (0.2, 0.4, 0.6, 0.8) for v in KAPPAS]
        for p in grid:
            hs, hi = method_hits(m, q, gal, qv, p, env)
            for r in RADII:
                lat, lon = refine(hs, hi, gal["xyz"], r, fb)
                d = bins.haversine(Q["lat"], Q["lon"], lat, lon)
                s = float(obj(d))
                if s > best.get(m, {}).get("val_obj", -1):
                    best[m] = {**p, "radius_km": r, "val_obj": s, "val_geoscore": float(geoscore(d).mean()), **tag}
        if m in DIAG:
            best[m]["ae_diag"] = ae_diag(q, Q, env, gal, DIAG[m])
        logger.info("tuned %s: %s", m, best[m])
        if best[m].get(key) in (KAPPAS[0], KAPPAS[-1]) and m != "B1":
            logger.warning("%s: %s=%s is on the grid edge; the optimum may lie outside", m, key, best[m][key])
        del gal
        torch.cuda.empty_cache()
    best["B0"] = {"val_obj": by_alpha[alpha], "val_geoscore": float(geoscore(d_alpha[alpha]).mean()), "alpha": alpha,
                  "by_alpha": by_alpha, **tag}
    json.dump(best, open(os.path.join(run_dir, "tuned.json"), "w"), indent=2)
    return best


def ae_within_cell_mrr(q_ae, true_lat, true_lon, true_cell, env, gal, max_q=20000, key="ae"):
    """Rank of the gallery point nearest the true location among its TEST_LEVEL cell, by cosine on gal vec `key`."""
    anc_l = env.anc[:, bins.TEST_L]
    gcell = anc_l[gal["leaf"].cpu().numpy()]
    order = np.argsort(gcell, kind="stable")
    starts = np.searchsorted(gcell[order], np.arange(anc_l.max() + 2))
    gae, gxyz = gal["vec"][key].cpu().numpy(), gal["xyz"].numpy()
    txyz = bins.to_xyz(true_lat, true_lon)
    rr, rr_rand = [], []
    for i in np.random.default_rng(0).permutation(np.flatnonzero(true_cell >= 0))[:max_q]:
        members = order[starts[true_cell[i]]:starts[true_cell[i] + 1]]
        if len(members) < 2:
            continue
        target = members[np.argmax(gxyz[members] @ txyz[i])]
        sims = gae[members].astype(np.float32) @ q_ae[i]
        rr.append(1.0 / (1 + (sims > sims[members == target][0]).sum()))
        rr_rand.append((np.log(len(members)) + 0.5772156649 + 0.5 / len(members)) / len(members))
    return {"mrr": float(np.mean(rr)), "mrr_random": float(np.mean(rr_rand)),
            "mrr_ratio": float(np.mean(rr) / np.mean(rr_rand)), "n": len(rr)}


def ae_diag(q, Q, env, gal, key="ae"):
    """Gate B: within-cell MRR of predicted AE (or tower embedding); oracle_* ranks with the true AE, same queries."""
    q_ae, at = q[key].float().numpy(), Q["ae_true"] if key == "ae" else None
    ok = np.ones(len(q_ae), bool) if at is None else ~np.isnan(at[:, 0])
    args = (Q["lat"][ok], Q["lon"][ok], Q["cells"][ok, bins.TEST_L], env, gal)
    out = ae_within_cell_mrr(q_ae[ok], *args, key=key)
    if at is not None:
        out["ae_cos"] = float((q_ae[ok] * at[ok]).sum(1).mean())
        out |= {f"oracle_{k}": v for k, v in ae_within_cell_mrr(at[ok], *args).items()}
    return out


REGION_KM = (1, 5, 10, 25)


def regions(model, ck, env, run_dir, split="val", levels=(2, bins.TEST_L)):
    """Localization given the right region: each query searches only its true level-L cell, best hit (R=0), no tuning.
    Rows: tower over the T2 map (if any), photo kNN, and the cell centroid; acc@REGION_KM + median km."""
    Q = env.queries(split)
    q = encode_queries(model, Q["X"], env.anc, m=1)
    qv = {"loc": q["loc"], "clip": normalize16(Q["X"])}
    gals = {"photos": ("clip", env.gallery("photos"))}
    if model.tower is not None:
        gals["tower_map"] = ("loc", env.gallery("T2", model.tower))
    res = {"n": len(Q["lat"]), "split": split}
    for L in levels:
        true = np.maximum(Q["cells"][:, L], 0)
        anc = to_dev(env.anc[:, L], torch.long)
        tc = torch.as_tensor(true)
        oracle = lambda qsl: torch.where(anc[None, :] == tc[qsl].to(device)[:, None], 0.0, -math.inf)
        lat_c, lon_c = env.cells[f"lat_{L}"][true], env.cells[f"lon_{L}"][true]
        rows = {"centroid": bins.haversine(Q["lat"], Q["lon"], lat_c, lon_c)}
        for name, (key, gal) in gals.items():
            hs, hi = retrieve(qv, oracle, gal, {key: 1.0}, env.n_leaf)
            rows[name] = bins.haversine(Q["lat"], Q["lon"], *refine(hs, hi, gal["xyz"], 0, (lat_c, lon_c)))
        res[str(config.QT_LEVELS[L])] = {m: {"median_km": float(np.median(d)), **{f"acc_{t}km": float((d <= t).mean())
                                             for t in REGION_KM}} for m, d in rows.items()}
        logger.info("regions L%s: %s", config.QT_LEVELS[L], {m: round(v["median_km"], 1) for m, v in res[str(config.QT_LEVELS[L])].items()})
    json.dump(res, open(os.path.join(run_dir, f"regions_{split}.json"), "w"), indent=2)
    return res


def stratify(d, keys):
    return {name: {str(k): {"n": int((v == k).sum()), "median_km": float(np.median(d[v == k])),
                            "acc_25km": float((d[v == k] <= 25).mean()),
                            "geoscore": float(geoscore(d[v == k]).mean())}
                   for k in np.unique(v)} for name, v in keys.items()}


def predict(model, env, Q, tuned):
    """(lat, lon) per tuned method on queries Q, plus B0."""
    q = encode_queries(model, Q["X"], env.anc, alpha=tuned["B0"]["alpha"])
    qv = query_vecs(q, Q["X"])
    preds = {"B0": cell_centroid(env, q["b0"].numpy())}
    for m, p in tuned.items():
        if m in GALLERY_OF:
            gal = env.gallery(GALLERY_OF[m], model.tower)
            preds[m] = refine(*method_hits(m, q, gal, qv, p, env), gal["xyz"], p["radius_km"], preds["B0"])
            del gal
            torch.cuda.empty_cache()
    return preds


GATE_TAUS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01)   # 1.01 disables the confidence arm
GATE_DELTAS = (0, 5, 10, 25, 50, 100)                                        # 0 disables the agreement arm


def gate(b2, t2, p1000, tau, delta):
    """T2 where the prior is confident in the TEST_LEVEL cell or B2 and T2 agree within delta km, else B2."""
    use = (p1000 >= tau) | (bins.haversine(*b2, *t2) <= delta)
    return np.where(use, t2[0], b2[0]), np.where(use, t2[1], b2[1])


def gate_cfg(run_dir):
    """B2 as tuned (GeoScore); T2 keeps its tuned kappa but R=0, so it stays sharp."""
    tuned = json.load(open(os.path.join(run_dir.split("@")[0], "tuned.json")))
    return {"B0": tuned["B0"], "B2": tuned["B2"], "T2": {**tuned["T2"], "radius_km": 0}}


def gate_eval(model, ck, env, run_dir):
    """Tune (tau, delta) on val: max acc@25km subject to GeoScore >= B2; then one test pass incl. the >25 km subset."""
    cfg, out = gate_cfg(run_dir), {}
    for split in ("val", "test"):
        Q = env.queries(split)
        p1000 = encode_queries(model, Q["X"], env.anc, alpha=cfg["B0"]["alpha"])["p1000"].numpy()
        pr = predict(model, env, Q, cfg)
        dist = lambda ll: bins.haversine(Q["lat"], Q["lon"], *ll)
        if split == "val":
            floor = geoscore(dist(pr["B2"])).mean()
            grid = [(t, dl, dist(gate(pr["B2"], pr["T2"], p1000, t, dl))) for t in GATE_TAUS for dl in GATE_DELTAS]
            ok = [g for g in grid if geoscore(g[2]).mean() >= floor]
            tau, delta, dv = max(ok or grid, key=lambda g: ((g[2] <= 25).mean(), geoscore(g[2]).mean()))
            out["gate"] = {"tau": tau, "delta": delta, "feasible": bool(ok), "val_b2_geoscore": float(floor),
                           "val_geoscore": float(geoscore(dv).mean()), "val_acc25": float((dv <= 25).mean())}
            logger.info("gate tuned on val: %s", out["gate"])
            continue
        d = {m: dist(pr[m]) for m in ("B0", "B2", "T2")} | {"GATE": dist(gate(pr["B2"], pr["T2"], p1000, tau, delta))}
        nn = env.nn_km(Q["lat"], Q["lon"])
        beat = lambda a, b, n: (b < n).mean() - (a < n).mean()
        sub_res = {}
        for name, m in (("all", np.ones(len(nn), bool)), ("far25", nn > 25)):
            sub_res[name] = {"n": int(m.sum()),
                             "methods": {k: {**distance_metrics(v[m]), "beat_oracle": float((v[m] < nn[m]).mean())} for k, v in d.items()},
                             "paired_vs_B2": {k: {**{c: bootstrap(lambda x, y, f=f: f(y) - f(x), d["B2"][m], v[m])
                                                     for c, f in CO_PRIMARY.items()}, "beat_oracle": bootstrap(beat, d["B2"][m], v[m], nn[m])}
                                              for k, v in d.items() if k != "B2"},
                             "gate_use_rate": float(((p1000 >= tau) | (bins.haversine(*pr["B2"], *pr["T2"]) <= delta))[m].mean())}
        out["test"] = sub_res
    g, f = out["test"]["all"]["paired_vs_B2"]["GATE"], out["test"]["far25"]["paired_vs_B2"]
    out["criteria"] = {"geoscore_not_worse": g["geoscore"][1] >= 0,
                       "keeps_half_t2_oracle_gain_far": out["test"]["far25"]["methods"]["GATE"]["beat_oracle"] - out["test"]["far25"]["methods"]["B2"]["beat_oracle"]
                       >= 0.5 * (out["test"]["far25"]["methods"]["T2"]["beat_oracle"] - out["test"]["far25"]["methods"]["B2"]["beat_oracle"]),
                       "far_gate_vs_b2": f["GATE"]}
    json.dump(out, open(os.path.join(run_dir, "gate.json"), "w"), indent=2, default=float)
    logger.info("gate result: %s", json.dumps({k: out[k] for k in ("gate", "criteria")}, default=float))
    return out



def test(model, ck, env, run_dir):
    tuned = json.load(open(os.path.join(run_dir, "tuned.json")))
    Q = env.queries("test")
    q = encode_queries(model, Q["X"], env.anc, alpha=tuned["B0"]["alpha"])
    qv = query_vecs(q, Q["X"])
    fb = cell_centroid(env, q["b0"].numpy())
    hier = encode_queries(model, Q["X"], env.anc, m=1, alpha=1.0)["b0"].numpy()
    preds, diag = {"B0": fb, "B0-hier": cell_centroid(env, hier), "B0-flat": cell_centroid(env, q["flat"].numpy())}, {}
    for m, p in tuned.items():
        if m not in GALLERY_OF:
            continue
        gal = env.gallery(GALLERY_OF[m], model.tower)
        preds[m] = refine(*method_hits(m, q, gal, qv, p, env), gal["xyz"], p["radius_km"], fb)
        if m in DIAG:
            diag[m] = ae_diag(q, Q, env, gal, DIAG[m])
        del gal
        torch.cuda.empty_cache()

    nn_km = env.nn_km(Q["lat"], Q["lon"])
    dens = env.cells[f"count_{bins.TEST_L}"][np.maximum(Q["cells"][:, bins.TEST_L], 0)]
    strata = {"continent": np.array([CC_TO_CONTINENT_NAME.get(c, "Unknown") for c in Q["country"]]),
              "land_cover": Q["land_cover"],
              "density_decile": np.searchsorted(np.quantile(dens, np.linspace(0, 1, 11)[1:-1]), dens),
              "nn_train_km": np.digitize(nn_km, [1, 5, 10, 25, 50, 100]),
              "region_correct": env.anc[q["b0"].numpy(), bins.TEST_L] == Q["cells"][:, bins.TEST_L]}
    gt_admin = admin_labels(Q["lat"], Q["lon"])
    dists = {m: bins.haversine(Q["lat"], Q["lon"], *ll) for m, ll in preds.items()}
    base = dists.get("B2")

    res = {"n_test": len(Q["lat"]), "nn_train_km_pct": {p: float(np.percentile(nn_km, p)) for p in (5, 25, 50, 75, 95)},
           "methods": {}}
    for m, d in dists.items():
        pa = admin_labels(*preds[m])
        res["methods"][m] = {**distance_metrics(d), **with_cis(d, None if m == "B2" else base),
                             **{f"acc_{n}": float((a == b).mean()) for n, a, b in zip(("country", "admin1", "city"), pa, gt_admin)},
                             "strata": stratify(d, strata)}
    tc = Q["cells"][:, :bins.TEST_L + 1]
    ok = tc[:, 0] >= 0
    res["cells"] = {str(config.QT_LEVELS[l]): {"top1": float((q["head_top5"][ok, l, 0].numpy() == tc[ok, l]).mean()),
                                               "top5": float((q["head_top5"][ok, l].numpy() == tc[ok, l, None]).any(1).mean()),
                                               "hier_top1": float((env.anc[q["b0"].numpy()[ok], l] == tc[ok, l]).mean())}
                    for l in range(bins.TEST_L + 1)}
    conf, corr = q["p1000"].numpy()[ok], (q["c1000"].numpy()[ok] == tc[ok, bins.TEST_L]).astype(float)
    dec = np.searchsorted(np.quantile(conf, np.linspace(0, 1, 11)[1:-1]), conf)
    res["calibration"] = {"ece_1000": ece(conf, corr),
                          "median_km_by_conf_decile": [float(np.median(dists["B0"][ok][dec == i])) for i in range(10)]}
    if diag:
        res["ae_diagnostics"] = diag
    json.dump(res, open(os.path.join(run_dir, "test_metrics.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(run_dir, "test_dists.npz"), **{m: d.astype(np.float32) for m, d in dists.items()})
    open(os.path.join(run_dir, "summary.md"), "w").write(summary_md(os.path.basename(run_dir), res))
    logger.info("\n%s", summary_md(os.path.basename(run_dir), res))
    return res


def summary_md(name, res):
    cols = ["median_km", "geoscore", "acc_1km", "acc_5km", "acc_10km", "acc_25km", "acc_200km", "acc_750km", "acc_2500km", "acc_country", "acc_admin1"]
    lines = [f"## {name} (test n={res['n_test']})", "", "| method | " + " | ".join(cols)
             + " | GeoScore 95% CI | vs B2: dGeoScore | d acc@25km (pt) | d median km |", "|---" * (len(cols) + 5) + "|"]
    for m, r in res["methods"].items():
        vals = [f"{r[c]:.1f}" if c in ("median_km", "geoscore") else f"{100 * r[c]:.2f}" for c in cols]
        ci = r["geoscore_ci"]
        lines.append(f"| {m} | " + " | ".join(vals) + f" | [{ci[0]:.0f}, {ci[1]:.0f}] | " + " | ".join(
            fmt_ci(r.get(f"delta_{k}_ci"), 100 if k == "acc_25km" else 1) for k in ("geoscore", "acc_25km", "median_km")) + " |")
    return "\n".join(lines) + "\n"


def fmt_ci(ci, scale=1):
    return f"[{scale * ci[0]:+.1f}, {scale * ci[1]:+.1f}]" if ci else "—"



# ── Report across runs ────────────────────────────────────────────────────────

CLAIMS = [("C1 AE distillation improves the prior (B0)", ("r1", "B0"), ("r4", "B0")),
          ("C1 ... and the CLIP-kNN hybrid (B2)", ("r1", "B2"), ("r4", "B2")),
          ("C2 AE re-ranking vs CLIP re-ranking", ("r4", "B2"), ("r4", "G0")),
          ("C3 + photo-free train locations", ("r4", "G0"), ("r4", "G1")),
          ("C3 + per-cell grid", ("r4", "G1"), ("r4", "G2")),
          ("C4 curvature (lorentz vs euclid prototypes, B0)", ("r6", "B0"), ("r7", "B0")),
          ("Decoupled: R1 prior + R4 AE vs coupled R4 (G0)", ("r4", "G0"), ("r4_prior-r1", "G0")),
          ("C5 AE location tower vs GPS tower (land grid)", ("t_gps", "T2"), ("t_ae", "T2")),
          ("C5 ... AE+GPS tower vs GPS tower", ("t_gps", "T2"), ("t_aegps", "T2")),
          ("C6 learned AE tower vs AE regression (land grid)", ("r4", "G2"), ("t_ae", "T2")),
          ("C3 tower: + land grid", ("t_ae", "T0"), ("t_ae", "T2"))]
GATE_C = ("Gate C (pre-registered): an AE method beats the strongest baseline iff the paired 95% CI excludes 0 in its "
          "favour on GeoScore or acc@25km, and no co-primary metric (GeoScore, acc@25km, median km) is significantly worse.")


def paired(a, b):
    """Paired bootstrap CIs of (b - a) on the co-primary metrics; a, b = (run, method) with saved test distances."""
    da, db = (np.load(os.path.join(config.RUNS_DIR, r, "test_dists.npz"))[m] for r, m in (a, b))
    return {k: bootstrap(lambda x, y, f=f: f(y) - f(x), da, db) for k, f in CO_PRIMARY.items()}


def claims_md(rows):
    """Pre-registered comparisons, plus Gate C: val-selected best AE method vs val-selected best B2, on test."""
    have = lambda r, m: os.path.exists(os.path.join(config.RUNS_DIR, r, "test_dists.npz")) and m in np.load(
        os.path.join(config.RUNS_DIR, r, "test_dists.npz")).files
    pick = lambda ms: max(((r, m, g) for r, m, g in rows if m in ms and not r.startswith("pilot") and have(r, m)),
                          key=lambda t: t[2], default=None)
    ours, base = pick({"G0", "G1", "G2", "Fused", "T0", "T2", "HT2", "UT2", "HG2", "BT2"}), pick({"B2"})
    todo = CLAIMS + ([("Gate C best AE vs best baseline", base[:2], ours[:2])] if ours and base else [])
    md = ["", "## Paired comparisons on test (b - a; lower median is better)", "", GATE_C, "",
          "| comparison | a | b | d GeoScore | d acc@25km (pt) | d median km |", "|---|---|---|---|---|---|"]
    for name, a, b in todo:
        if have(*a) and have(*b):
            ci = paired(a, b)
            md.append(f"| {name} | {'/'.join(a)} | {'/'.join(b)} | {fmt_ci(ci['geoscore'])} | "
                      f"{fmt_ci(ci['acc_25km'], 100)} | {fmt_ci(ci['median_km'])} |")
    return md


def report():
    runs = sorted(glob.glob(os.path.join(config.RUNS_DIR, "*", "tuned.json")))
    rows = []
    for t in runs:
        run = os.path.basename(os.path.dirname(t))
        tuned = json.load(open(t))
        if any(p.get("objective", "geoscore") != "geoscore" or p.get("min_km") for p in tuned.values()):
            continue
        for m, p in tuned.items():
            rows.append((run, m, p["val_geoscore"]))
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    md = ["# Validation GeoScore (selection only; test numbers in each run's summary.md)", "",
          "| run | method | val GeoScore |", "|---|---|---|"] + [f"| {r} | {m} | {g:.1f} |" for r, m, g in rows]
    md += claims_md(rows)
    md += ["", *[open(s).read() for s in sorted(glob.glob(os.path.join(config.RUNS_DIR, "*", "summary.md")))]]
    open(os.path.join(config.RESULTS_DIR, "report.md"), "w").write("\n".join(md))
    logger.info("report: %d rows", len(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["tune", "test", "report", "regions", "gate"])
    ap.add_argument("--split", choices=["val", "test"], default="val", help="regions only")
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--prior-ckpt", help="cell logits from this model, AE from ckpt (decoupled scoring)")
    ap.add_argument("--methods", help="comma list: tune only these (B0 always); pair with --tag to keep the full run intact")
    ap.add_argument("--objective", choices=list(OBJECTIVES), default="geoscore", help="tune selection metric")
    ap.add_argument("--min-km", type=float, default=0.0, help="keep queries farther than this from any train row")
    ap.add_argument("--tag", help="write to <run_dir>@<tag> instead of the checkpoint's run dir")
    a = ap.parse_args()
    if a.cmd == "report":
        return report()
    if not a.ckpt:
        ap.error(f"{a.cmd} needs a checkpoint path")
    model, ck = load_bridge(a.ckpt)
    data = lambda c: (c["args"].get("feats", "streetclip"), c["args"].get("rows", "all"))
    env = Env(ck["args"]["ae_radius"], *data(ck))
    run_dir = os.path.dirname(os.path.abspath(a.ckpt))
    if a.prior_ckpt:
        prior, pck = load_bridge(a.prior_ckpt)
        if data(pck) != data(ck):
            raise SystemExit(f"--prior-ckpt was trained on {data(pck)}, ckpt on {data(ck)}: features differ")
        model = Decoupled(prior, model)
        run_dir += "_prior-" + os.path.basename(os.path.dirname(os.path.abspath(a.prior_ckpt)))
    env.min_km = a.min_km
    if a.tag:
        run_dir += "@" + a.tag
    os.makedirs(run_dir, exist_ok=True)
    only = set(a.methods.split(",")) if a.methods else None
    {"tune": lambda *x: tune(*x, only, objective=a.objective), "test": test, "regions": lambda *x: regions(*x, a.split), "gate": gate_eval}[a.cmd](model, ck, env, run_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
