#!/usr/bin/env python3
"""
Image(s) -> JSON-lines payloads for the Rust inference engine.

Usage:
  python predict.py photo.jpg [more.jpg ...] | inference/target/release/inference

Reads results/best_config.json (tuned on val) and data/serve/leaf_counts.npy (written by `ingest.py milvus`).
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import bins
import config
from eval import Decoupled, encode_queries
from model import encode_pixels, load_bridge, load_streetclip


def make_payloads(model, clip_feats, cfg):
    """Top-k leaves with log p - log n (the eval prior), predicted AE vector, and the B0 centroid fallback."""
    cells = bins.load("cells")
    counts = np.load(os.path.join(config.SERVE_DIR, "leaf_counts.npy"))
    q = encode_queries(model, clip_feats, cells["anc"], m=cfg["k_cells"], alpha=cfg["alpha"])
    for i in range(len(q["b0"])):
        idx, lp, b0 = q["top_idx"][i].numpy(), q["top_lp"][i].numpy(), int(q["b0"][i])
        keep = counts[idx] > 0
        yield {"field": cfg["field"], "vec": q["ae"][i].float().tolist(), "kappa": cfg["kappa"],
               "radius_km": cfg["radius_km"], "limit": cfg["limit"],
               "cells": [[int(l), float(p - np.log(counts[l]))] for l, p in zip(idx[keep], lp[keep])],
               "fallback": [float(cells[f"lat_{bins.LEAF}"][b0]), float(cells[f"lon_{bins.LEAF}"][b0])]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(config.RESULTS_DIR, "best_config.json")))
    model, ck = load_bridge(cfg["ckpt"])
    if ck["args"].get("feats", "streetclip") != "streetclip":
        raise SystemExit(f"serving encodes with StreetCLIP; {cfg['ckpt']} was trained on {ck['args']['feats']} features")
    if cfg["prior_ckpt"]:
        model = Decoupled(load_bridge(cfg["prior_ckpt"])[0], model)
    processor, vision = load_streetclip()
    px = processor(images=[Image.open(p).convert("RGB") for p in a.images], return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        feats = encode_pixels(vision, px).float().cpu()
    for pl in make_payloads(model, feats, cfg):
        print(json.dumps(pl))


if __name__ == "__main__":
    main()
