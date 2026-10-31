#!/usr/bin/env python3
"""
Validation-only comparison of candidate caches (rerank.py without --test). The reranker is fit with 5-fold
sequence cross-fitting inside val, so every val photo is scored by a model that never saw its sequence; arms are
compared photo by photo against the first (baseline) with paired bootstrap CIs, overall and by distance to train.
Never opens a test cache.

Usage:
  python valcv.py v2_f1_ft_n4894684 v2_f1_ft_n4894684_rhr2 [more tags] --name rhead   # -> results/valcv/rhead.json
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd
import torch

import bins
import config
from eval import CO_PRIMARY, bootstrap, geoscore, write_manifest
from rerank import I_B2, MBR_T, lgb_fit, lgb_scores, mbr

logger = logging.getLogger(__name__)
N_FOLDS = 5
STRATA = {"all": (0, np.inf), "nn<5km": (0, 5), "nn5-25km": (5, 25), "nn>25km": (25, np.inf)}


def val_folds(n):
    """Sequence-hash fold (0..4) of each val photo, in env.queries('val') order."""
    rows, meta = bins.encoded_rows("test", "fusion768"), bins.load("test")
    seq = bins.read_csv("test")["sequence"].fillna("").to_numpy(object)
    f = (pd.util.hash_array(seq)[rows[meta["tune"][rows]]] // bins.TUNE_MOD % N_FOLDS).astype(np.int64)
    if len(f) != n:
        raise ValueError(f"{len(f)} val rows vs {n} cached queries")
    return f


def cross_fit(va, folds):
    """Out-of-fold reranker pick per val photo -> km to the truth."""
    pick = torch.zeros(len(va["loc"]), dtype=torch.long)
    for k in range(N_FOLDS):
        tr, te = (torch.as_tensor(np.flatnonzero(m)) for m in (folds != k, folds == k))
        sub = lambda idx: {key: va[key][idx] for key in ("loc", "qf", "dist")}
        model = lgb_fit(sub(tr), seed=k)
        pick[te] = mbr(lgb_scores(model, sub(te)), va["cand"][te], MBR_T)
        logger.info("fold %d: %d trees", k, model.best_iteration)
    return va["dist"][torch.arange(len(pick)), pick].numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tags", nargs="+", help="cache tags; the first is the baseline")
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    paths = [os.path.join(config.DATA_DIR, "rerank", f"{config.SPLIT}_val_{t}.pt") for t in a.tags]
    C = {t: torch.load(p) for t, p in zip(a.tags, paths)}
    base = C[a.tags[0]]
    for t, va in C.items():
        if not torch.equal(va["nn_km"], base["nn_km"]):
            raise ValueError(f"{t} was built on different val queries than {a.tags[0]}")
    folds, nn_km = val_folds(len(base["loc"])), base["nn_km"].numpy()
    d = {t: {"rerank": cross_fit(va, folds), "B2": va["dist"][:, I_B2].numpy(), "oracle": va["dist"].min(1).values.numpy()}
         for t, va in C.items()}
    res = {"tags": a.tags, "n": len(nn_km), "strata": {}}
    for s, (lo, hi) in STRATA.items():
        m = (nn_km >= lo) & (nn_km < hi)
        res["strata"][s] = {"n": int(m.sum()), **{t: {
            **{f"{k}_geoscore": float(geoscore(v[m]).mean()) for k, v in dt.items()},
            "acc_25km": float((dt["rerank"][m] <= 25).mean()), "median_km": float(np.median(dt["rerank"][m])),
            **({} if t == a.tags[0] else {f"vs_base_{c}": [float(f(dt["rerank"][m]) - f(d[a.tags[0]]["rerank"][m])),
                                                           *bootstrap(lambda x, y, f=f: f(y) - f(x), d[a.tags[0]]["rerank"][m], dt["rerank"][m])]
                                          for c, f in CO_PRIMARY.items()})} for t, dt in d.items()}}
        logger.info("%s: %s", s, json.dumps({t: {k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()}
                                              for t, r in res["strata"][s].items() if t != "n"}))
    out = os.path.join(config.RESULTS_DIR, "valcv", f"{a.name}.json")
    write_manifest(out.replace(".json", ".manifest.json"), paths)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
    logger.info("wrote %s", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
