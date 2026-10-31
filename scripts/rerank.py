#!/usr/bin/env python3
"""
Learned candidate reranker: one location per query, picked from photo-kNN hits, AlphaEarth-map hits, prior cells
and the B0/B2/T2 outputs. A LightGBM lambdarank model fit on val scores the candidates and the pick is the
minimum-Bayes-risk candidate under expected GeoScore. Inductive: test is never used for fitting; one pass on test.

Usage:
  python rerank.py ../runs/ft/best.pt --prior ../runs/f1/best.pt          # val candidate cache only (valcv.py compares)
  python rerank.py ../runs/ft/best.pt --prior ../runs/f1/best.pt --test   # fit on val, the one pass on test
RHEAD=<name> retrieves B2 on that retrieval head, with B2 tuned in runs/<prior>@rhead_<name>/tuned.json.
"""

import argparse
import json
import logging
import math
import os

import lightgbm as lgb
import numpy as np
import torch
from torch import nn

import bins
import config
from eval import (CO_PRIMARY, COARSE_LEVELS, GALLERY_OF, Env, bootstrap, cell_centroid, distance_metrics, encode_queries,
                  write_manifest,
                  gate_cfg, method_hits, query_vecs, refine, to_dev)
from model import device, load_bridge

logger = logging.getLogger(__name__)
K, TOPC, PRIOR_TOP = 32, 8, 64          # hits kept per method, prior cells as candidates, cells for prior mass
HIT_KM, PRIOR_KM = (1, 5, 25, 100, 500), (25, 100, 500)
SRC_N = (K, K, TOPC, 1, 1, 1)           # b2 hits | t2 hits | cells | B0 | B2 | T2
I_B2 = sum(SRC_N[:4])                   # column of the B2 output among candidates
GS_KM = 1492.7
MBR_T = 0.25                            # MBR softmax temperature, chosen leave-one-fold-out on val


def km(a, b):
    """Great-circle km between unit vectors a [..., n, 3] and b [..., m, 3] -> [..., n, m]."""
    return bins.R_EARTH_KM * torch.arccos((a @ b.transpose(-1, -2)).clamp(-1, 1))


def static_feats():
    """[C, 7]: source one-hot and within-source rank / K."""
    src = torch.repeat_interleave(torch.arange(len(SRC_N)), torch.tensor(SRC_N))
    rank = torch.cat([torch.arange(n) for n in SRC_N]).float() / K
    return torch.cat([nn.functional.one_hot(src, len(SRC_N)).float(), rank[:, None]], 1)


@torch.no_grad()
def raw_sim(qf, v, hi, bs=4096):
    """Own-method similarity of each kept hit: qf [N,D] host, v [G,D] device, hi [N,K] -> [N,K]."""
    return torch.cat([(qf[s:s + bs].to(device)[:, None] * v[hi[s:s + bs].to(v.device)].to(device)).float().sum(-1).cpu()
                      for s in range(0, len(hi), bs)])


@torch.no_grad()
def candidate_feats(cand, hits, cells, preds, p1000, own, leafc, pq, bs=2048):
    """cand [N,C,3]; hits {m: (scores [N,K], xyz [N,K,3])}; cells (xyz [N,P,3], logp [N,P]); preds {m: xyz [N,3]};
    own [N,C] raw score in its source; leafc [N,C] leaf ids; pq: top_idx/top_lp, lp{l}, anc [L,levels], logn [L].
    Returns per-candidate features [N,C,23] (hit / prior mass within radii, log km to each output, own leaf and
    coarse prior, gallery density, raw score and gap to source best, log km to nearest other candidate) and [N,7]."""
    loc, qf = [], []
    anc, logn = (to_dev(pq["anc"], torch.long), pq["logn"].to(device)) if len(cand) else (None, None)
    for s in range(0, len(cand), bs):
        c, f = cand[s:s + bs].to(device), []
        w = {m: torch.softmax(h[0][s:s + bs].to(device).float(), 1).nan_to_num(0) for m, h in hits.items()}
        for m in ("B2", "T2"):
            d = km(c, hits[m][1][s:s + bs].to(device))
            f += [((d <= r) * w[m][:, None]).sum(-1) for r in HIT_KM]
        pw = torch.softmax(cells[1][s:s + bs].to(device).float(), 1)
        d = km(c, cells[0][s:s + bs].to(device))
        f += [((d <= r) * pw[:, None]).sum(-1) for r in PRIOR_KM]
        px = {m: v[s:s + bs, None].to(device) for m, v in preds.items()}
        f += [torch.log1p(km(c, px[m])[..., 0]) for m in ("B0", "B2", "T2")]
        lc, tl = leafc[s:s + bs].to(device), pq["top_lp"][s:s + bs].to(device).float()
        eq = lc[..., None] == pq["top_idx"][s:s + bs].to(device)[:, None]
        f += [torch.where(eq.any(-1), (eq * tl[:, None]).sum(-1), tl[:, -1:]), logn[lc],
              *(pq[f"lp{l}"][s:s + bs].to(device).float().gather(1, anc[lc, l]) for l in COARSE_LEVELS)]
        o = own[s:s + bs].to(device).float()
        dc = km(c, c).diagonal_scatter(torch.full(c.shape[:2], math.inf, device=device), dim1=1, dim2=2)
        f += [o, torch.cat([b - b.max(1, keepdim=True).values for b in o.split(SRC_N, 1)], 1), torch.log1p(dc.min(-1).values)]
        loc.append(torch.stack(f, -1).nan_to_num(0, 0, 0).half().cpu())
        gap = lambda m, s=s: (hits[m][0][s:s + bs, 0] - hits[m][0][s:s + bs, 1]).float().nan_to_num(0, 0, 0).to(device)
        qf.append(torch.stack([p1000[s:s + bs].to(device), -(pw * pw.clamp(min=1e-12).log()).sum(1), gap("B2"), gap("T2"),
                               torch.log1p(km(px["B2"], px["T2"])[:, 0, 0]), w["B2"].max(1).values, w["T2"].max(1).values],
                              1).cpu())
    return torch.cat(loc), torch.cat(qf)


def xyz(ll):
    return torch.as_tensor(bins.to_xyz(*ll), dtype=torch.float32)


def candidates(pm, tm, env, cfg, X, gallery):
    """Candidates and features for query features X; gallery(m) -> the B2 / T2 gallery. pm supplies the prior, B0 and
    B2; tm the T2 tower. Returns ({cand, loc, qf}, {B0, B2, T2: (lat, lon)}, the prior model's query outputs)."""
    qp = encode_queries(pm, X, env.anc, alpha=cfg["B0"]["alpha"])
    qt = qp if tm is pm else encode_queries(tm, X, env.anc, alpha=cfg["T2"]["alpha"])
    fb = cell_centroid(env, qp["b0"].numpy())
    hits, preds, raw, leaves = {}, {"B0": fb}, {}, {}
    for m, q in (("B2", qp), ("T2", qt)):
        gal, qv = gallery(m), query_vecs(q, X, env.head)
        hs, hi = method_hits(m, q, gal, qv, cfg[m], env)
        preds[m] = refine(hs, hi, gal["xyz"], cfg[m]["radius_km"], fb)
        hi = hi[:, :K]
        f = "clip" if m == "B2" else "loc"
        raw[m], leaves[m] = raw_sim(qv[f], gal["vec"][f], hi), gal["leaf"].cpu()[hi]
        if m == "B2":
            logn = torch.log1p(torch.bincount(gal["leaf"], minlength=env.n_leaf).float()).cpu()
        hits[m] = (hs[:, :K], gal["xyz"][hi])
        del gal
        torch.cuda.empty_cache()
    top = qp["top_idx"].long()
    cells = (xyz(cell_centroid(env, top[:, :PRIOR_TOP].numpy())), qp["top_lp"][:, :PRIOR_TOP].float())
    px = {m: xyz(v) for m, v in preds.items()}
    cand = torch.cat([hits["B2"][1], hits["T2"][1], cells[0][:, :TOPC], px["B0"][:, None], px["B2"][:, None],
                      px["T2"][:, None]], 1)
    leafc = torch.cat([leaves["B2"], leaves["T2"], top[:, :TOPC + 1], leaves["B2"][:, :1], leaves["T2"][:, :1]], 1)
    own = torch.cat([raw["B2"], raw["T2"], cells[1][:, :TOPC], torch.zeros(len(cand), 3)], 1)
    pq = {"top_idx": top, "top_lp": qp["top_lp"], "anc": env.anc, "logn": logn,
          **{f"lp{l}": qp[f"lp{l}"] for l in COARSE_LEVELS}}
    loc, qf = candidate_feats(cand, hits, cells, px, qp["p1000"].float(), own, leafc, pq)
    return {"cand": cand, "loc": loc, "qf": qf}, preds, qp


def extract(pm, tm, env, cfg, split, tag):
    """candidates() for one query split plus per-candidate km to the truth, cached under data/rerank."""
    path = os.path.join(config.DATA_DIR, "rerank", f"{config.SPLIT}_{split}_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path)
    Q = env.queries(split)
    out = candidates(pm, tm, env, cfg, Q["X"], lambda m: env.gallery(GALLERY_OF[m], tm.tower))[0]
    out |= {"dist": km(out["cand"], xyz((Q["lat"], Q["lon"]))[:, None])[..., 0],
            "nn_km": torch.as_tensor(env.nn_km(Q["lat"], Q["lon"]), dtype=torch.float32)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)
    write_manifest(path + ".manifest.json", n_queries=len(out["loc"]), cfg=cfg)
    logger.info("extract %s: %d queries x %d candidates, %d loc + %d query feats", split, *out["loc"].shape, out["qf"].shape[1])
    return out


def inputs(d):
    return torch.cat([d["loc"].float(), static_feats().expand(len(d["loc"]), -1, -1),
                      d["qf"][:, None].expand(-1, d["loc"].shape[1], -1)], -1)


def report(te, choice):
    """Reranked vs B2 (same candidates) on all test queries and the >25 km-from-train subset, paired CIs."""
    dist, nn_km = te["dist"].numpy(), te["nn_km"].numpy()
    d = {"B2": dist[:, I_B2], "RERANK": dist[np.arange(len(dist)), choice.numpy()], "ORACLE": dist.min(1)}
    beat = lambda a, b, n: (b < n).mean() - (a < n).mean()
    res = {}
    for name, m in (("all", np.ones(len(nn_km), bool)), ("far25", nn_km > 25)):
        res[name] = {"n": int(m.sum()),
                     "methods": {k: {**distance_metrics(v[m]), "beat_oracle": float((v[m] < nn_km[m]).mean()),
                                     "geoscore_ci": bootstrap(CO_PRIMARY["geoscore"], v[m])} for k, v in d.items()},
                     "rerank_vs_B2": {**{c: bootstrap(lambda x, y, f=f: f(y) - f(x), d["B2"][m], d["RERANK"][m])
                                         for c, f in CO_PRIMARY.items()},
                                      "beat_oracle": bootstrap(beat, d["B2"][m], d["RERANK"][m], nn_km[m])},
                     "chosen_source": np.bincount(np.searchsorted(np.cumsum(SRC_N), choice.numpy()[m], side="right"),
                                                  minlength=len(SRC_N)).tolist()}
    return res


def lgb_fit(tr, seed=0, hold=0.1):
    """LightGBM lambdarank on flattened candidates, labels = 31 GeoScore grades; early-stops on a held-out val slice."""
    n, c = tr["loc"].shape[:2]
    perm = np.random.default_rng(seed).permutation(n)
    ho, fi = perm[:int(hold * n)], perm[int(hold * n):]
    gs = torch.exp(-tr["dist"] / GS_KM).numpy()
    ds = lambda q: lgb.Dataset(inputs({k: tr[k][q] for k in ("loc", "qf")}).flatten(0, 1).numpy(), np.rint(30 * gs[q]).ravel(),
                               group=[c] * len(q))
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [1], "label_gain": list(range(31)),
              "lambdarank_truncation_level": 10, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed, "num_threads": os.cpu_count(),
              "verbose": -1}
    return lgb.train(params, ds(fi), 2000, valid_sets=[ds(ho)], callbacks=[lgb.early_stopping(100, verbose=False)])


def lgb_scores(model, d, bs=4096):
    c = d["loc"].shape[1]
    return torch.cat([torch.as_tensor(model.predict(inputs({k: d[k][s:s + bs] for k in ("loc", "qf")}).flatten(0, 1).numpy(),
                                                    num_threads=os.cpu_count())).view(-1, c).float()
                      for s in range(0, len(d["loc"]), bs)])


def mbr(scores, cand, t):
    """Candidate maximizing expected GeoScore against softmax(scores / t) over the candidate set."""
    p = torch.softmax(scores / t, 1)
    return torch.cat([(torch.exp(-km(cand[s:s + 4096], cand[s:s + 4096]) / GS_KM) @ p[s:s + 4096, :, None])[..., 0].argmax(1)
                      for s in range(0, len(cand), 4096)])


def setup(ckpt, prior=None):
    """(prior model, tower model, Env, cfg, tower ckpt args): B0 / B2 tuned on the prior run, T2 kept sharp as in gate_cfg."""
    tm, ck = load_bridge(ckpt)
    pm = load_bridge(prior)[0] if prior else tm
    env = Env(ck["args"]["ae_radius"], ck["args"].get("feats", "streetclip"), ck["args"].get("rows", "all"))
    tuned_dir = os.path.dirname(os.path.abspath(prior or ckpt)) + (f"@rhead_{config.RHEAD}" if config.RHEAD else "")
    with open(os.path.join(tuned_dir, "tuned.json")) as f:
        tuned = json.load(f)
    g = gate_cfg(os.path.dirname(os.path.abspath(ckpt)))
    return pm, tm, env, {"B0": tuned["B0"], "B2": tuned["B2"], "T2": {**g["T2"], "alpha": g["B0"]["alpha"]}}, ck["args"]


def cache_tag(env, tower_dir, prior_dir):
    """Names a candidate cache by everything that changes it: runs, gallery size, retrieval head, grid fill."""
    return (f"v2_{os.path.basename(prior_dir)}_{os.path.basename(tower_dir)}_n{len(env.rows)}"
            + (f"_rh{config.RHEAD}" if config.RHEAD else "") + ("_fill" if config.GRID_FILL else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="run with the location tower (T2)")
    ap.add_argument("--prior", help="run supplying the prior, B0 and B2 (default: ckpt)")
    ap.add_argument("--test", action="store_true", help="also extract test, fit on val and score test once")
    a = ap.parse_args()
    pm, tm, env, cfg, _ = setup(a.ckpt, a.prior)
    run_dir = os.path.dirname(os.path.abspath(a.ckpt))
    tag = cache_tag(env, run_dir, os.path.dirname(os.path.abspath(a.prior or a.ckpt)))
    tr = extract(pm, tm, env, cfg, "val", tag)
    if not a.test:
        return logger.info("val cache %s: %d queries; compare with valcv.py", tag, len(tr["loc"]))
    out = f"{run_dir}@rerank_{tag}"
    os.makedirs(out, exist_ok=True)
    write_manifest(os.path.join(out, "manifest.json"), [a.ckpt, a.prior or ""], cfg=cfg)
    te = extract(pm, tm, env, cfg, "test", tag)
    model = lgb_fit(tr)
    choice = mbr(lgb_scores(model, te), te["cand"], MBR_T)
    res = {"cfg": cfg, "mbr_t": MBR_T, "test": report(te, choice)}
    np.savez_compressed(os.path.join(out, "test_dists.npz"), B2=te["dist"][:, I_B2].numpy(),
                        RERANK=te["dist"][torch.arange(len(choice)), choice].numpy())
    with open(os.path.join(out, "rerank.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    model.save_model(os.path.join(out, "scorer.lgb"))
    logger.info("rerank test: %s", json.dumps({s: {"gs": {k: round(v["geoscore"], 1) for k, v in r["methods"].items()},
                                                   "d_gs_ci": r["rerank_vs_B2"]["geoscore"]} for s, r in res["test"].items()},
                                              default=float))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
