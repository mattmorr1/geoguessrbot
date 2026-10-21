#!/usr/bin/env python3
"""
Local demo: drop a photo, see where B0 (cell prior), B2 (photo kNN), T2 (AlphaEarth map) and the gated FINAL place it.

Usage (on the GPU box; block-split checkpoints never saw the held-out regions):
  SPLIT=blocks python eval.py gate ../runs_blocks/ft/best.pt --tag gate   # once: writes ft@gate/gate.json
  SPLIT=blocks python serve.py --run ft
  ssh -N -L 8000:127.0.0.1:8000 matt-pc   # IPv4 loopback: WSL forwarding ignores ::1; then open http://localhost:8000 on the Mac
"""

import argparse
import http.server
import io
import json
import logging
import os

import numpy as np
import torch
from PIL import Image, ImageOps

import bins
import config
from eval import FEATS, GALLERY_OF, Env, cell_centroid, encode_queries, gate, gate_cfg, method_hits, query_vecs, refine
from model import encode_pixels, load_backbone, load_bridge

logger = logging.getLogger(__name__)
METHODS = ("B2", "T2")
TOP_CELLS = 64
PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")


def load_encoders(feats):
    """Frozen backbones for a checkpoint's feature set; fusion768 also loads the train-fit PCA."""
    pca = np.load(os.path.join(config.FEAT_DIR, f"{feats}_pca.npz")) if feats == "fusion768" else None
    return [load_backbone(b) for b in (FEATS["fusion"] if pca is not None else FEATS[feats])], pca


@torch.no_grad()
def encode_images(encoders, images):
    """PIL images -> [N, dim] fp16 features, matching bins.load_feats / bins.fuse on stored shards."""
    backbones, pca = encoders
    parts = [encode_pixels(v, p(images=images, return_tensors="pt")["pixel_values"]).float().cpu().numpy() for p, v in backbones]
    if len(parts) == 1:
        return parts[0].astype(np.float16)
    x = np.concatenate([f / np.linalg.norm(f, axis=1, keepdims=True) for f in parts], 1)
    return (((x.astype(np.float16).astype(np.float32) - pca["mean"]) @ pca["W"]) if pca is not None else x).astype(np.float16)


def exif_gps(img):
    """(lat, lon) from EXIF GPSInfo, or None."""
    g = img.getexif().get_ifd(0x8825)
    if not all(k in g for k in (1, 2, 3, 4)):
        return None
    dms = lambda v: float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    return dms(g[2]) * (-1 if g[1] == "S" else 1), dms(g[4]) * (-1 if g[3] == "W" else 1)


class Demo:
    def __init__(self, run):
        run_dir = os.path.join(config.RUNS_DIR, run)
        self.model, ck = load_bridge(os.path.join(config.RUNS_DIR, run.split("@")[0], "best.pt"))
        feats = ck["args"].get("feats", "streetclip")
        self.env = Env(ck["args"]["ae_radius"], feats, ck["args"].get("rows", "all"))
        self.tuned = gate_cfg(run_dir)
        with open(os.path.join(run_dir + "@gate", "gate.json")) as f:
            self.gate = json.load(f)["gate"]
        self.gals = {m: self.env.gallery(GALLERY_OF[m], self.model.tower) for m in METHODS if m in self.tuned}
        self.encoders = load_encoders(feats)
        self.leaf_ll = self.env.cells[f"lat_{bins.LEAF}"], self.env.cells[f"lon_{bins.LEAF}"]
        logger.info("ready: run=%s methods=%s gate=%s", run, list(self.gals), self.gate)

    def locate(self, img, gt=None):
        X = encode_images(self.encoders, [img])
        q = encode_queries(self.model, X, self.env.anc, alpha=self.tuned["B0"]["alpha"])
        qv, fb = query_vecs(q, X), cell_centroid(self.env, q["b0"].numpy())
        preds = {"B0": fb} | {m: refine(*method_hits(m, q, g, qv, self.tuned[m], self.env), g["xyz"],
                                        self.tuned[m]["radius_km"], fb) for m, g in self.gals.items()}
        preds["FINAL"] = gate(preds["B2"], preds["T2"], q["p1000"].numpy(), self.gate["tau"], self.gate["delta"])
        idx, lp = q["top_idx"][0, :TOP_CELLS].numpy(), q["top_lp"][0, :TOP_CELLS].float().numpy()
        p = np.exp(lp - lp.max())
        out = {"preds": {m: [float(la[0]), float(lo[0])] for m, (la, lo) in preds.items()}, "p1000": float(q["p1000"][0]),
               "cells": [[float(self.leaf_ll[0][i]), float(self.leaf_ll[1][i]), float(w)] for i, w in zip(idx, p / p.sum())]}
        if gt is not None:
            out["truth"] = list(gt)
            out["err_km"] = {m: float(bins.haversine(gt[0], gt[1], *ll)) for m, ll in out["preds"].items()}
            out["nn_train_km"] = float(self.env.nn_km(np.array([gt[0]]), np.array([gt[1]]))[0])
        return out


def handler(demo):
    class H(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/":
                return self._send(404, b'{"error": "not found"}')
            with open(PAGE, "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, b'{"error": "not found"}')
            try:
                raw = Image.open(io.BytesIO(self.rfile.read(int(self.headers["Content-Length"]))))
                gt, img = exif_gps(raw), ImageOps.exif_transpose(raw).convert("RGB")
            except Exception as e:
                logger.warning("bad upload: %s", e)
                return self._send(400, json.dumps({"error": f"unreadable image: {e}"}).encode())
            self._send(200, json.dumps(demo.locate(img, gt)).encode())

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="ft", help="run dir under RUNS_DIR holding tuned.json; '@tag' reuses the base checkpoint")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    # single-threaded on purpose: requests serialize on the one GPU
    http.server.HTTPServer(("127.0.0.1", a.port), handler(Demo(a.run))).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
