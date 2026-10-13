#!/usr/bin/env python3
"""
Train AlphaBridge on frozen StreetCLIP features over the nested OSV5M quadtree.

Usage:
  python train.py --name r4 --ae-radius 80                      # main config (linear heads, soft AE)
  python train.py --name pilot_r80 --subset pilot --ae-radius 80
Ablations: --head {linear,euclid,lorentz}  --w-ae 0  --w-aux 0  --target-tau 0 (hard InfoNCE)  --batch cell
Location tower: --loc {ae,gps,ae+gps} adds a two-tower image<->location loss on within-cell batches (eval T0/T2).
"""

import argparse
import copy
import json
import logging
import os
import time

import numpy as np
import torch
from torch.amp import GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import bins
import config
from eval import Env, quick_val
from model import AlphaBridge, ContrastiveQueue, bridge_loss, bridge_spec, cross_view_loss, device, pyramid_feats

logger = logging.getLogger(__name__)

USE_AMP = device == "cuda"
BATCH_KEYS = ("x", "xyz", "ae", "ae_ok", "aux_cat", "aux_reg")


# ── EMA ───────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return dict(self.shadow)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(subset, ae_radius, need_ctx=False, feats="streetclip", rows="all", pyr=None, loc_level=None):
    """Train dict over encoded train rows; val = tune slice of the test split (eval.Env). ctx stays on the host."""
    env = Env(ae_radius, feats, rows)
    meta, rows = env.train, env.rows
    pos = np.arange(len(rows)) if subset == "full" else np.flatnonzero(bins.zip_of("train")[rows] < 4)
    if "holdout" in meta:
        pos = pos[~meta["holdout"][rows[pos]]]
    r = rows[pos]
    a = np.full((len(r), 64), np.nan, np.float16) if env.ae_train is None else np.asarray(env.ae_train[r])
    tr = {"x": env.clip[pos], "xyz": bins.to_xyz(meta["lat"][r], meta["lon"][r]), "ae": np.nan_to_num(a),
          "ae_ok": ~np.isnan(a[:, 0]), "aux_cat": meta["aux_cat"][r].astype(np.int64),
          "aux_reg": meta["aux_reg"][r], "leaf": meta["cells"][r, bins.LEAF].astype(np.int64),
          "group": meta["cells"][r, bins.LEAF if loc_level is None else loc_level].astype(np.int64)}
    if need_ctx:
        ctx = env.ctx("train")
        if ctx is None:
            raise SystemExit(f"no AE contexts at {bins.ctx_path('train')}; run geo_lookup.py --contexts --sets train")
        tr["ctx"] = torch.from_numpy(np.asarray(ctx[r])).pin_memory() if device == "cuda" else torch.from_numpy(np.asarray(ctx[r]))
    if pyr is not None:
        tr["pyr"] = pyr.index(meta["lat"][r], meta["lon"][r])
    return env.cells, tr, env.queries("val")


def to_gpu(d):
    t = {k: torch.as_tensor(d[k]).to(device) for k in (*BATCH_KEYS, "leaf", "group")}
    t["ae_ok"] = t["ae_ok"].float()
    return t


def iid_batches(n, bs, steps):
    return torch.randperm(n, device=device)[:steps * bs].view(steps, bs)


def cell_batches(leaf, bs, steps, per=32):
    """Leaves drawn by size, `per` rows each: in-batch negatives become within-cell hard negatives."""
    order, cnt = torch.argsort(leaf), torch.bincount(leaf)
    start = torch.cumsum(cnt, 0) - cnt
    cells = torch.multinomial(cnt.float(), steps * (bs // per), replacement=True)
    off = (torch.rand(len(cells), per, device=device) * cnt[cells, None]).long()
    return order[start[cells, None] + off].view(steps, -1)


# ── Training ──────────────────────────────────────────────────────────────────

def train(a) -> None:
    torch.manual_seed(a.seed)
    run_dir = os.path.join(config.RUNS_DIR, a.name)
    os.makedirs(run_dir, exist_ok=True)

    pyr = bins.Pyramid() if "pyr" in a.loc else None
    cells, tr, va = load_data(a.subset, a.ae_radius, "ae" in a.loc.split("+"), a.feats, a.rows, pyr, a.loc_level)
    if a.w_ae > 0 and not tr["ae_ok"].any():
        raise SystemExit(f"--w-ae {a.w_ae} but no train AlphaEarth targets at r={a.ae_radius}")
    logger.info("train %d | val %d | AE valid %.1f%% | leaves %d",
                len(tr["x"]), len(va["X"]), 100 * tr["ae_ok"].mean(), len(cells["anc"]))

    T, H, n = to_gpu(tr), tr.get("ctx"), len(tr["x"])
    if H is not None:
        logger.info("AE context: %.1f%% rows with a valid centre pixel", 100 * (H[:, 4, 0] != -128).float().mean())
    PT = None if pyr is None else (torch.as_tensor(pyr.M, device=device), torch.as_tensor(pyr.H, device=device),
                                   torch.as_tensor(tr["pyr"], device=device))
    if PT is not None:
        logger.info("AE pyramid: %.1f%% of row x depth cells described", 100 * (tr["pyr"] < pyr.off[-1]).mean())
    del tr
    spec = bridge_spec(cells, T["x"].shape[1])
    model = AlphaBridge(**spec, head=a.head, loc=a.loc).to(device)
    if pyr is not None and model.tower.n_pyr != len(pyr.depths):
        raise SystemExit(f"tower expects {model.tower.n_pyr} pyramid depths, {bins.pyr_path()} has {len(pyr.depths)}")
    ema, ema_model = EMA(model, config.EMA_DECAY), copy.deepcopy(model)
    queue = ContrastiveQueue(64, config.QUEUE_SIZE, device=device)
    ctx = {"cent": [torch.as_tensor(bins.to_xyz(cells[f"lat_{l}"], cells[f"lon_{l}"]), device=device)
                    for l in range(bins.N_LEVELS)],
           "tau": [float(cells[f"tau_{l}"]) * a.tau_scale for l in range(bins.N_LEVELS)],
           "target_tau": a.target_tau}
    weights = {"cell": 1.0, "ae": a.w_ae, "aux": a.w_aux}

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=config.WEIGHT_DECAY)
    steps = n // a.batch_size
    total = a.epochs * steps
    warm = min(config.WARMUP_EPOCHS * steps, total)
    sched = SequentialLR(opt, [LinearLR(opt, start_factor=0.01, total_iters=warm),
                               CosineAnnealingLR(opt, T_max=max(1, total - warm))], milestones=[warm])
    scaler = GradScaler(enabled=USE_AMP)
    val_ae = va["ae_true"]
    val_ok = None if val_ae is None else ~np.isnan(val_ae[:, 0])

    best, patience = -float("inf"), 0
    with open(os.path.join(run_dir, "log.jsonl"), "a") as log:
        for epoch in range(a.epochs):
            model.train()
            t0, sums = time.time(), {}
            batches = cell_batches(T["leaf"], a.batch_size, steps) if a.batch == "cell" else iid_batches(n, a.batch_size, steps)
            # tower batches are cell-structured at --loc-level: negatives share the cell the prior hands the tower
            loc_b = cell_batches(T["group"], a.batch_size, steps) if a.loc != "none" else None
            loc_c = loc_b.cpu() if H is not None else None
            for s, idx in enumerate(batches):
                b = {k: T[k][idx] for k in BATCH_KEYS}
                ctx["queue"], ctx["temperature"] = queue.get(device), model.temperature
                with torch.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                    out = model(b["x"])
                    loss, parts = bridge_loss(out, b, ctx, weights)
                    if loc_b is not None:
                        j = loc_b[s]
                        v = model.tower(None if H is None else H[loc_c[s]].to(device, non_blocking=True), T["xyz"][j],
                                        None if PT is None else pyramid_feats(PT[0], PT[1], PT[2][j]))
                        l_loc = cross_view_loss(model.loc_head(model.trunk(T["x"][j])), v, T["xyz"][j],
                                                model.loc_temperature, a.loc_tau_km)
                        loss, parts["loc"] = loss + a.w_loc * l_loc, l_loc.detach()
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                ema.update(model)
                if a.w_ae > 0:
                    queue.push(b["ae"][b["ae_ok"] > 0])
                for k, v in parts.items():
                    sums[k] = sums.get(k, 0) + v

            ema_model.load_state_dict(ema.state_dict())
            ema_model.eval()
            metrics = quick_val(ema_model, va["X"], va["lat"], va["lon"], cells["anc"], cells, val_ae, val_ok)
            rec = {"epoch": epoch + 1, "sec": round(time.time() - t0, 1), "lr": opt.param_groups[0]["lr"],
                   "temp": float(model.temperature), **({"loc_temp": float(model.loc_temperature)} if a.loc != "none" else {}),
                   **{f"train_{k}": float(v) / steps for k, v in sums.items()}, **metrics}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            logger.info("%s", rec)

            ckpt = {"state": ema.state_dict(), "spec": spec, "args": vars(a), "epoch": epoch + 1, "val": metrics}
            torch.save(ckpt, os.path.join(run_dir, "last.pt"))
            if metrics["val_geoscore"] > best + config.EARLY_STOP_MIN_DELTA:
                best, patience = metrics["val_geoscore"], 0
                torch.save(ckpt, os.path.join(run_dir, "best.pt"))
            else:
                patience += 1
                if patience >= config.EARLY_STOP_PATIENCE:
                    logger.info("early stop at epoch %d", epoch + 1)
                    break
    logger.info("done: best val GeoScore %.1f -> %s", best, os.path.join(run_dir, "best.pt"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--subset", choices=["full", "pilot"], default="full")
    ap.add_argument("--head", choices=["linear", "euclid", "lorentz"], default="linear")
    ap.add_argument("--ae-radius", type=int, default=80, help="AlphaEarth target scale in metres: 80 | 160 | 320")
    ap.add_argument("--w-ae", type=float, default=1.0)
    ap.add_argument("--w-aux", type=float, default=0.3)
    ap.add_argument("--target-tau", type=float, default=0.05, help="0 = hard InfoNCE")
    ap.add_argument("--tau-scale", type=float, default=1.0, help="multiplier on per-level haversine smoothing")
    ap.add_argument("--batch", choices=["iid", "cell"], default="iid")
    ap.add_argument("--feats", choices=["streetclip", "siglip2", "fusion", "fusion768"], default="streetclip")
    ap.add_argument("--rows", choices=["all", "common"], default="all", help="common = rows every backbone encoded")
    ap.add_argument("--loc", choices=["none", "ae", "gps", "ae+gps", "ae+pyr", "pyr"], default="none",
                    help="location tower inputs: ae point context, gps Fourier features, pyr AE cell pyramid")
    ap.add_argument("--w-loc", type=float, default=1.0)
    ap.add_argument("--loc-level", type=int, help="quadtree level grouping tower batches (default: leaf)")
    ap.add_argument("--loc-tau-km", type=float, default=1.0, help="geodesic soft-target scale of the tower loss")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(config.RUNS_DIR, a.name), exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(config.RUNS_DIR, a.name, "train.log"))])
    train(a)


if __name__ == "__main__":
    main()
