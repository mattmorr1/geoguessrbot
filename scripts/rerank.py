#!/usr/bin/env python3
"""
Learned candidate reranker: one location per query, picked from photo-kNN hits, AlphaEarth-map hits, prior cells
and the B0/B2/T2 outputs by an MLP trained to maximize expected GeoScore. One pass on test.
Official: scorer cross-fit over test hash folds (fold k scored by a model fit on val + the other test folds).
--val-gate: 5-fold sequence cross-fit inside val only; exit 0 iff the paired GeoScore CI vs B2 excludes 0 upward.
Runs trained with --holdout seq: the scorer fits on the train holdout (inductive); the gate also requires it to beat
the val cross-fit scorer on val, and test is scored once by a model fit on holdout + val.

Usage:
  python rerank.py ../runs/ft/best.pt --prior ../runs/f1/best.pt   # T2 from ft, B0/B2/prior from f1
  SPLIT=blocks python rerank.py ../runs_blocks/ft/best.pt         # blocks: fit on val only
  python rerank.py ../runs/ft/best.pt --prior ../runs/f1/best.pt --scorer lgb   # inductive: LightGBM + MBR on val
"""

import argparse
import json
import logging
import math
import os

import bins
import config
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from eval import (
    CO_PRIMARY,
    COARSE_LEVELS,
    GALLERY_OF,
    Env,
    bootstrap,
    cell_centroid,
    distance_metrics,
    encode_queries,
    gate_cfg,
    method_hits,
    query_vecs,
    refine,
    to_dev,
)
from model import device, load_bridge
from torch import nn

logger = logging.getLogger(__name__)
K, TOPC, PRIOR_TOP = 32, 8, 64          # hits kept per method, prior cells as candidates, cells for prior mass
HIT_KM, PRIOR_KM = (1, 5, 25, 100, 500), (25, 100, 500)
SRC_N = (K, K, TOPC, 1, 1, 1)           # b2 hits | t2 hits | cells | B0 | B2 | T2
I_B2 = sum(SRC_N[:4])                   # column of the B2 output among candidates
GS_KM = 1492.7
MBR_T = 0.25                            # MBR softmax temperature, chosen leave-one-fold-out on val (rerank_exp.py)


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


def extract(pm, tm, env, cfg, split, tag):
    """Candidates, features and per-candidate km to the truth for one query split (cached under data/rerank).
    pm supplies the prior, B0 and B2; tm the T2 tower (and its own prior inside T2 retrieval)."""
    path = os.path.join(config.DATA_DIR, "rerank", f"{config.SPLIT}_{split}_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path)
    Q = env.queries(split)
    qp = encode_queries(pm, Q["X"], env.anc, alpha=cfg["B0"]["alpha"])
    qt = qp if tm is pm else encode_queries(tm, Q["X"], env.anc, alpha=cfg["T2"]["alpha"])
    fb = cell_centroid(env, qp["b0"].numpy())
    hits, preds, raw, leaves = {}, {"B0": fb}, {}, {}
    for m, q in (("B2", qp), ("T2", qt)):
        gal, qv = env.gallery(GALLERY_OF[m], tm.tower), query_vecs(q, Q["X"])
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
    xyz = lambda ll: torch.as_tensor(bins.to_xyz(*ll), dtype=torch.float32)
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
    out = {"cand": cand, "loc": loc, "qf": qf, "dist": km(cand, xyz((Q["lat"], Q["lon"]))[:, None])[..., 0],
           "nn_km": torch.as_tensor(env.nn_km(Q["lat"], Q["lon"]), dtype=torch.float32)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(out, path)
    logger.info("extract %s: %d queries x %d candidates, %d loc + %d query feats", split, *loc.shape, qf.shape[1])
    return out


class Scorer(nn.Module):
    def __init__(self, n_in, mean, std, hidden=128):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("static", static_feats())
        self.mlp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, loc, qf):
        n, c, _ = loc.shape
        x = torch.cat([loc.float(), self.static.expand(n, -1, -1), qf[:, None].expand(-1, c, -1)], -1)
        return self.mlp((x - self.mean) / self.std)[..., 0]


def inputs(d):
    return torch.cat([d["loc"].float(), static_feats().expand(len(d["loc"]), -1, -1),
                      d["qf"][:, None].expand(-1, d["loc"].shape[1], -1)], -1)


def fit(tr, epochs=60, bs=512, lr=1e-3, hold=0.1, seed=0):
    """Expected-GeoScore listwise loss; early-stops on argmax GeoScore of a held-out slice of val."""
    torch.manual_seed(seed)
    n = len(tr["loc"])
    perm = torch.randperm(n)
    ho, fi = perm[:int(hold * n)], perm[int(hold * n):]
    x = inputs({k: tr[k][fi[:20000]] for k in ("loc", "qf")}).flatten(0, 1)
    model = Scorer(x.shape[1], x.mean(0), x.std(0).clamp(min=1e-6)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dev = {k: tr[k].to(device) for k in ("loc", "qf", "dist")}
    gs = torch.exp(-dev["dist"] / GS_KM)
    best, state = -1.0, None
    for ep in range(epochs):
        model.train()
        for b in fi[torch.randperm(len(fi))].split(bs):
            b = b.to(device)
            loss = -(torch.softmax(model(dev["loc"][b], dev["qf"][b]), 1) * gs[b]).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            h = ho.to(device)
            score = float(5000 * gs[h].gather(1, model(dev["loc"][h], dev["qf"][h]).argmax(1, keepdim=True)).mean())
        if score > best:
            best, state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
        logger.info("epoch %d loss %.4f held-out GeoScore %.1f (B2 %.1f)", ep, float(loss), score,
                    float(5000 * gs[h, I_B2].mean()))
    model.load_state_dict(state)
    return model, best


@torch.no_grad()
def pick(model, d, bs=8192):
    return torch.cat([model(d["loc"][s:s + bs].to(device), d["qf"][s:s + bs].to(device)).argmax(1).cpu()
                      for s in range(0, len(d["loc"]), bs)])


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

def lgb_fit(tr, rank, seed=0, hold=0.1):
    """LightGBM on flattened candidates; lambdarank on 31 GeoScore grades or pointwise GeoScore regression."""
    n, c = tr["loc"].shape[:2]
    perm = np.random.default_rng(seed).permutation(n)
    ho, fi = perm[:int(hold * n)], perm[int(hold * n):]
    gs = torch.exp(-tr["dist"] / GS_KM).numpy()
    ds = lambda q: lgb.Dataset(inputs({k: tr[k][q] for k in ("loc", "qf")}).flatten(0, 1).numpy(),
                               np.rint(30 * gs[q]).ravel() if rank else gs[q].ravel(), group=[c] * len(q) if rank else None)
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [1], "label_gain": list(range(31)),
              "lambdarank_truncation_level": 10} if rank else {"objective": "regression", "metric": "l2"}
    params |= {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 200, "feature_fraction": 0.8,
               "bagging_fraction": 0.8, "bagging_freq": 1, "seed": seed, "num_threads": os.cpu_count(), "verbose": -1}
    dtr = ds(fi)
    return lgb.train(params, dtr, 2000, valid_sets=[ds(ho)], callbacks=[lgb.early_stopping(100, verbose=False)])


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


def crossfit(te, folds, extra=None):
    """Fold k of te is scored by a model fit on extra plus te folds != k; nothing scores rows it trained on."""
    extra = extra or {k: te[k][:0] for k in ("loc", "qf", "dist")}
    pool = {k: torch.cat([extra[k], te[k]]) for k in ("loc", "qf", "dist")}
    pf = torch.cat([torch.full((len(extra["loc"]),), -1), torch.as_tensor(folds, dtype=torch.long)])
    choice = torch.zeros(len(te["loc"]), dtype=torch.long)
    for k in np.unique(folds):
        scorer, held = fit({n: v[pf != k] for n, v in pool.items()})
        choice[torch.as_tensor(folds == k)] = pick(scorer, {n: te[n][folds == k] for n in ("loc", "qf")})
        logger.info("crossfit fold %d: held-out GeoScore %.1f", k, held)
    return choice


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", help="run with the location tower (T2)")
    ap.add_argument("--prior", help="run supplying the prior, B0 and B2 (default: ckpt)")
    ap.add_argument("--val-gate", action="store_true", help="val-only cross-fit go/no-go; never touches test")
    ap.add_argument("--scorer", choices=("mlp", "lgb"), default="mlp",
                    help="lgb: LightGBM lambdarank + MBR fit on val only, scored once on test (inductive)")
    a = ap.parse_args()
    tm, ck = load_bridge(a.ckpt)
    pm = load_bridge(a.prior)[0] if a.prior else tm
    run_dir = os.path.dirname(os.path.abspath(a.ckpt))
    prior_dir = os.path.dirname(os.path.abspath(a.prior or a.ckpt))
    env = Env(ck["args"]["ae_radius"], ck["args"].get("feats", "streetclip"), ck["args"].get("rows", "all"),
              ck["args"].get("holdout"))
    with open(os.path.join(prior_dir, "tuned.json")) as f:
        tuned = json.load(f)
    g = gate_cfg(run_dir)
    cfg = {"B0": tuned["B0"], "B2": tuned["B2"], "T2": {**g["T2"], "alpha": g["B0"]["alpha"]}}
    tag = f"v2_{os.path.basename(prior_dir)}"
    out = run_dir + "@rerank2"
    os.makedirs(out, exist_ok=True)
    rows, meta = env._rows("test"), bins.load("test")
    tr = extract(pm, tm, env, cfg, "val", tag)
    tho = extract(pm, tm, env, cfg, "trainho", tag) if ck["args"].get("holdout") else None
    take = lambda d, c: d["dist"][torch.arange(len(c)), c].numpy()
    if a.val_gate:
        seq = bins.read_csv("test")["sequence"].fillna("").to_numpy(object)
        folds = (pd.util.hash_array(seq)[rows[meta["tune"][rows]]] // bins.TUNE_MOD % 5).astype(np.int64)
        cv = crossfit(tr, folds)
        gate = {"val_cv": report(tr, cv)}
        ok = gate["val_cv"]["all"]["rerank_vs_B2"]["geoscore"][0] > 0
        if tho is not None:
            th = pick(fit(tho)[0], tr)
            gate["val_trainho"] = report(tr, th)
            gate["trainho_vs_cv"] = bootstrap(lambda x, y: CO_PRIMARY["geoscore"](y) - CO_PRIMARY["geoscore"](x),
                                              take(tr, cv), take(tr, th))
            ok = gate["val_trainho"]["all"]["rerank_vs_B2"]["geoscore"][0] > 0 and gate["trainho_vs_cv"][0] > 0
        with open(os.path.join(out, "val_gate.json"), "w") as f:
            json.dump({"cfg": cfg, **gate, "pass": ok}, f, indent=2, default=float)
        logger.info("val gate: %s -> %s", {k: {m: round(x["geoscore"], 1) for m, x in v["all"]["methods"].items()}
                                           if isinstance(v, dict) else v for k, v in gate.items()}, "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 3)
    te = extract(pm, tm, env, cfg, "test", tag)
    if a.scorer == "lgb":
        choice, out = mbr(lgb_scores(lgb_fit(tr, True), te), te["cand"], MBR_T), out + "_lgb"
        os.makedirs(out, exist_ok=True)
        res = {"cfg": cfg, "mbr_t": MBR_T, "test": report(te, choice)}
        prev = os.path.join(run_dir + "@rerank2", "test_dists.npz")
        if os.path.exists(prev):
            mlp = np.load(prev)["val_only"]
            res["vs_mlp_val_only"] = bootstrap(lambda x, y: CO_PRIMARY["geoscore"](y) - CO_PRIMARY["geoscore"](x), mlp, take(te, choice))
        np.savez_compressed(os.path.join(out, "test_dists.npz"), B2=te["dist"][:, I_B2].numpy(), lgb_mbr=take(te, choice))
        with open(os.path.join(out, "rerank.json"), "w") as f:
            json.dump(res, f, indent=2, default=float)
        logger.info("rerank lgb test: %s vs MLP val-only %s", {s: round(v["methods"]["RERANK"]["geoscore"], 1)
                                                              for s, v in res["test"].items()}, res.get("vs_mlp_val_only"))
        return
    val_scorer, held = fit(tr)
    picks = {"val_only": pick(val_scorer, te)}
    res = {"held_out_val_geoscore": held, "cfg": cfg, "test_val_only": report(te, picks["val_only"])}
    if tho is not None:
        cat = {k: torch.cat([tho[k], tr[k]]) for k in ("loc", "qf", "dist")}
        picks["inductive"] = pick(fit(cat)[0], te)
        res["test"] = report(te, picks["inductive"])
    elif config.SPLIT == "official":
        picks["crossfit"] = crossfit(te, bins.test_folds()[rows][~meta["tune"][rows]], extra=tr)
        res["test"] = report(te, picks["crossfit"])
    np.savez_compressed(os.path.join(out, "test_dists.npz"), B2=te["dist"][:, I_B2].numpy(),
                        **{k: te["dist"][torch.arange(len(v)), v].numpy() for k, v in picks.items()})
    with open(os.path.join(out, "rerank.json"), "w") as f:
        json.dump(res, f, indent=2, default=float)
    torch.save(val_scorer.state_dict(), os.path.join(out, "scorer_val.pt"))
    logger.info("rerank test: %s", json.dumps({p: {s: {"gs": {k: round(v["geoscore"], 1) for k, v in r["methods"].items()},
                                                       "d_gs_ci": r["rerank_vs_B2"]["geoscore"]} for s, r in res[p].items()}
                                               for p in ("test_val_only", "test") if p in res}, default=float))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
