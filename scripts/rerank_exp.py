#!/usr/bin/env python3
"""
Val-only reranker bake-off; never opens a test cache. Every arm is scored out-of-fold on the 5 val-gate sequence
folds: MLP vs LightGBM (lambdarank, pointwise), argmax vs minimum-Bayes-risk (MBR) pick, and with --trainho the train
holdout resampled so its nn_km distribution matches the training folds' (val), added to each fold's training set.
MBR temperature is chosen leave-one-fold-out. Decision rule: an arm replaces the MLP only if its paired GeoScore CI vs
the MLP excludes 0 upward.

Usage:
  python rerank_exp.py v2_f1                # scorers on the ft/f1 caches
  python rerank_exp.py v2_f1h --trainho     # + matched-holdout arms on the fth/f1h caches
"""

import argparse
import json
import logging
import os

import bins
import config
import numpy as np
import pandas as pd
import torch
from eval import CO_PRIMARY, bootstrap
from model import device
from rerank import I_B2, fit, lgb_fit, lgb_scores, mbr, report

logger = logging.getLogger(__name__)
TEMPS = (0.25, 0.5, 1, 2, 4)
NN_EDGES = np.r_[0, np.geomspace(0.1, 2000, 24), np.inf]
gs_of = lambda d: CO_PRIMARY["geoscore"](d)


def val_folds(n, feats):
    rows, meta = bins.encoded_rows("test", feats), bins.load("test")
    seq = bins.read_csv("test")["sequence"].fillna("").to_numpy(object)
    folds = (pd.util.hash_array(seq)[rows[meta["tune"][rows]]] // bins.TUNE_MOD % 5).astype(np.int64)
    if len(folds) != n:
        raise RuntimeError(f"val folds ({len(folds)}) do not match the cache ({n}); wrong --feats or stale cache")
    return folds


def matched(ho, ref_nn, seed):
    """Resample holdout queries (with replacement, same count) so their nn_km histogram matches ref_nn's."""
    b_ho, b_ref = (np.digitize(x, NN_EDGES) for x in (ho["nn_km"].numpy(), ref_nn))
    p_ref = np.bincount(b_ref, minlength=len(NN_EDGES) + 1) / len(b_ref)
    w = p_ref[b_ho] / np.bincount(b_ho, minlength=len(NN_EDGES) + 1)[b_ho]
    if (uncovered := p_ref[np.bincount(b_ho, minlength=len(NN_EDGES) + 1) == 0].sum()) > 0:
        logger.warning("holdout covers no queries in nn_km bins holding %.1f%% of val mass", 100 * uncovered)
    idx = np.random.default_rng(seed).choice(len(w), len(w), p=w / w.sum())
    logger.info("matched holdout: %d queries, %d unique, ESS %.0f", len(idx), len(np.unique(idx)), w.sum() ** 2 / (w ** 2).sum())
    return {k: ho[k][idx] for k in ("loc", "qf", "dist")}



@torch.no_grad()
def mlp_scores(model, d, bs=8192):
    return torch.cat([model(d["loc"][s:s + bs].to(device), d["qf"][s:s + bs].to(device)).cpu() for s in range(0, len(d["loc"]), bs)])




def oof(va, folds, scorer, extra=None):
    """Out-of-fold candidate scores on val; extra(train_fold_dict, fold) -> optional extra training rows."""
    out = torch.zeros(va["dist"].shape)
    for k in np.unique(folds):
        tr = {n: va[n][torch.as_tensor(folds != k)] for n in ("loc", "qf", "dist", "nn_km")}
        if extra is not None:
            ex = extra(tr, k)
            tr = {n: torch.cat([tr[n], ex[n]]) for n in ("loc", "qf", "dist")}
        out[torch.as_tensor(folds == k)] = scorer(tr, {n: va[n][torch.as_tensor(folds == k)] for n in ("loc", "qf")})
        logger.info("fold %d done", k)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", help="cache tag, e.g. v2_f1 (reads data/rerank/<SPLIT>_val_<tag>.pt)")
    ap.add_argument("--trainho", action="store_true", help="add matched train-holdout arms (needs <SPLIT>_trainho_<tag>.pt)")
    ap.add_argument("--feats", default="fusion768", help="backbone whose test rows define val")
    a = ap.parse_args()
    cache = lambda s: torch.load(os.path.join(config.DATA_DIR, "rerank", f"{config.SPLIT}_{s}_{a.tag}.pt"))
    va = cache("val")
    folds = val_folds(len(va["loc"]), a.feats)
    scorers = {"mlp": lambda tr, te: mlp_scores(fit(tr)[0], te),
               "lgb_rank": lambda tr, te: lgb_scores(lgb_fit(tr, True), te),
               "lgb_reg": lambda tr, te: lgb_scores(lgb_fit(tr, False), te)}
    arms = {n: oof(va, folds, f) for n, f in scorers.items()}
    if a.trainho:
        ho = cache("trainho")
        mix = lambda tr, k: matched(ho, tr["nn_km"].numpy(), seed=int(k))
        arms |= {f"{n}+ho": oof(va, folds, scorers[n], mix) for n in ("mlp", "lgb_rank")}
        arms["mlp_ho_only"] = oof(va, folds, lambda tr, te: mlp_scores(fit(matched(ho, tr["nn_km"].numpy(), 0))[0], te))
    rows = torch.arange(len(folds))
    take = lambda c: va["dist"][rows, c].numpy()
    choices = {}
    for n, s in arms.items():
        choices[n] = s.argmax(1)
        per_t = {t: mbr(s, va["cand"], t) for t in TEMPS}
        gs_t = {t: np.array([gs_of(take(c)[folds == k]) for k in range(5)]) for t, c in per_t.items()}
        lofo = np.array([max(TEMPS, key=lambda t: np.delete(gs_t[t], k).mean()) for k in range(5)])
        choices[f"{n}@mbr"] = torch.where(torch.as_tensor(lofo[folds] == np.array(TEMPS)[:, None]).T,
                                          torch.stack([per_t[t] for t in TEMPS], 1), 0).sum(1)
        logger.info("%s: argmax %.1f, MBR temps per fold %s", n, gs_of(take(choices[n])), lofo.tolist())
    base = take(choices["mlp"])
    res = {"tag": a.tag, "n": len(folds), "B2": gs_of(va["dist"][:, I_B2].numpy()), "oracle": gs_of(va["dist"].min(1).values.numpy()),
           "arms": {n: {"geoscore": gs_of(take(c)), "vs_mlp_ci": bootstrap(lambda x, y: gs_of(y) - gs_of(x), base, take(c)),
                        "report": report(va, c)} for n, c in choices.items()}}
    out = os.path.join(config.RUNS_DIR, f"rerank_exp_{a.tag}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2, default=float)
    logger.info("val OOF GeoScore (B2 %.1f, oracle %.1f): %s", res["B2"], res["oracle"],
                json.dumps({n: [round(r["geoscore"], 1), [round(x, 1) for x in r["vs_mlp_ci"]]] for n, r in res["arms"].items()}, default=float))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
