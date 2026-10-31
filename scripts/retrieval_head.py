#!/usr/bin/env python3
"""
Geo-retrieval head over frozen photo features: photo<->photo soft InfoNCE so nearby photos embed nearby (B2 retrieval).
Trains on train photos only; no val or test row is read.

Usage:
  python retrieval_head.py --name r2 --batch cell      # same-leaf groups of 32: hard negatives inside the prior cell
  python retrieval_head.py --name r1 --batch iid       # ablation: in-batch negatives only
Writes runs/rhead_<name>.pt (+ .manifest.json) and data/feats/rhead_<name>_train.npy (fp16, env.rows order).
Then: RHEAD=<name> python eval.py tune ../runs/f1/best.pt --methods B2 ; RHEAD=<name> python rerank.py ...
"""

import argparse
import logging
import os
import time

import numpy as np
import torch

import bins
import config
from eval import Env, write_manifest
from model import RetrievalHead, device, embed_rhead, photo_loss
from train import cell_batches, iid_batches

logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--batch", choices=["iid", "cell"], default="cell")
    ap.add_argument("--tau-km", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--d-out", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if config.RHEAD:
        raise SystemExit("unset RHEAD: the head trains on raw features")
    torch.manual_seed(a.seed)
    env = Env(80, "fusion768")
    X, r = torch.as_tensor(np.asarray(env.clip)), env.rows
    xyz = torch.as_tensor(bins.to_xyz(env.train["lat"][r], env.train["lon"][r]), device=device)
    leaf = torch.as_tensor(env.train["cells"][r, bins.LEAF].astype(np.int64), device=device)
    head = RetrievalHead(X.shape[1], a.hidden, a.d_out).to(device)
    n, steps = len(X), len(X) // a.batch_size
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.epochs * steps, pct_start=1 / a.epochs)
    logger.info("rhead %s: %d photos, %s batches, tau %.1f km, %d epochs", a.name, n, a.batch, a.tau_km, a.epochs)
    losses = []
    for ep in range(a.epochs):
        t0, tot = time.time(), 0.0
        B = cell_batches(leaf, a.batch_size, steps) if a.batch == "cell" else iid_batches(n, a.batch_size, steps)
        Bc = B.cpu()
        for s in range(steps):
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                e = head(X[Bc[s]].to(device, non_blocking=True))
            loss = photo_loss(e.float(), xyz[B[s]], head.log_inv_temp.exp().clamp(max=200), a.tau_km)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss)
        losses.append(tot / steps)
        logger.info("epoch %d: loss %.4f temp %.4f (%.0f s)", ep + 1, losses[-1], float(1 / head.log_inv_temp.exp()), time.time() - t0)
    path = os.path.join(config.RUNS_DIR, f"rhead_{a.name}.pt")
    torch.save({"state": head.state_dict(), "args": vars(a), "d_in": X.shape[1], "losses": losses}, path)
    feats = os.path.join(config.FEAT_DIR, f"rhead_{a.name}_train.npy")
    np.save(feats, embed_rhead(head.eval(), X))
    write_manifest(path + ".manifest.json", [feats], args=vars(a), n_train=n, losses=losses)
    logger.info("saved rhead_%s", a.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
