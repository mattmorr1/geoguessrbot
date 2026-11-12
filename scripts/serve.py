#!/usr/bin/env python3
"""
Demo server for the headline model: drop a photo, see the reranked FINAL pick plus the B0 (cell prior), B2 (photo kNN)
and T2 (AlphaEarth map) candidates it chose among.

Usage (scorer.lgb comes from rerank.py --test; galleries are cached beside it on first start):
  FEAT_DIR=~/serve_feats python serve.py --run ft --prior f1   # serve_feats: the 2M-photo headline gallery files
  ssh -N -L 8000:127.0.0.1:8000 matt-pc   # IPv4 loopback: WSL forwarding ignores ::1; then open http://localhost:8000
Cloud Run (Dockerfile, deploy.sh): BUNDLE=gs://... is downloaded into the repo layout before the model loads. The port
opens first, so the page renders at once; /health?wait=1 holds a request open (and CPU allocated) until the model is up.
"""

import argparse
import http.server
import io
import json
import logging
import os
import threading
import urllib.parse

import lightgbm as lgb
import numpy as np
import torch
from PIL import Image, ImageOps

import bins
import config
import rerank
from eval import FEATS, GALLERY_OF, gallery_dev
from model import device, encode_pixels, load_backbone

logger = logging.getLogger(__name__)
TOP_CELLS = 64
MAX_UPLOAD = 25 << 20
WAIT_S = 240          # under the Cloud Run request timeout (300 s)
SOURCES = ("photo kNN", "AlphaEarth map", "prior cell", "B0", "B2", "T2")
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
PAGES = {"/": "index.html", "/index.html": "index.html", "/how.html": "how.html"}


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


def fetch_bundle(url, root):
    """gs://bucket/prefix/** -> root/**, each blob in parallel 32 MB slices; on Cloud Run this download is the cold start."""
    from google.cloud.storage import Client, transfer_manager
    bucket, _, prefix = url.removeprefix("gs://").partition("/")
    blobs = list(Client().list_blobs(bucket, prefix=prefix.rstrip("/") + "/"))
    for b in blobs:
        dst = os.path.join(root, b.name[len(prefix.rstrip("/")) + 1:])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        transfer_manager.download_chunks_concurrently(b, dst, chunk_size=32 << 20, max_workers=16,
                                                      worker_type=transfer_manager.THREAD)
    logger.info("bundle: %d files, %.2f GB from %s", len(blobs), sum(b.size for b in blobs) / 1e9, url)


def cached_gallery(env, m, tower, path):
    """env.gallery once, then torch.load: the T2 tower pass over ~2.5M points is minutes on CPU."""
    if os.path.exists(path):
        gal = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        return gal | {"leaf": gal["leaf"].to(device), "vec": {k: v.to(gallery_dev(*v.shape)) for k, v in gal["vec"].items()}}
    gal = env.gallery(GALLERY_OF[m], tower)
    torch.save({k: v.cpu() if torch.is_tensor(v) else {n: t.cpu() for n, t in v.items()} if k == "vec" else v
                for k, v in gal.items()}, path)
    return gal


class Demo:
    def __init__(self, run, prior):
        ckpt = lambda r: os.path.join(config.RUNS_DIR, r, "best.pt")
        self.pm, self.tm, self.env, self.cfg, args = rerank.setup(ckpt(run), ckpt(prior))
        run_dir = os.path.join(config.RUNS_DIR, run)
        out = f"{run_dir}@rerank_{rerank.cache_tag(self.env, run_dir, os.path.join(config.RUNS_DIR, prior))}"
        self.scorer = lgb.Booster(model_file=os.path.join(out, "scorer.lgb"))
        self.gals = {m: cached_gallery(self.env, m, self.tm.tower, os.path.join(out, f"serve_{m}.pt")) for m in ("B2", "T2")}
        self.encoders = load_encoders(args.get("feats", "streetclip"))
        self.leaf_ll = self.env.cells[f"lat_{bins.LEAF}"], self.env.cells[f"lon_{bins.LEAF}"]
        logger.info("ready: run=%s prior=%s gallery B2=%d T2=%d", run, prior, *(len(g["lat"]) for g in self.gals.values()))

    def locate(self, img, gt=None):
        d, preds, q = rerank.candidates(self.pm, self.tm, self.env, self.cfg, encode_images(self.encoders, [img]),
                                        self.gals.__getitem__)
        pick = rerank.mbr(rerank.lgb_scores(self.scorer, d), d["cand"], rerank.MBR_T)
        preds["FINAL"] = bins.to_latlon(d["cand"][0, pick].double().numpy())
        idx, lp = q["top_idx"][0, :TOP_CELLS].numpy(), q["top_lp"][0, :TOP_CELLS].float().numpy()
        p = np.exp(lp - lp.max())
        out = {"preds": {m: [float(la[0]), float(lo[0])] for m, (la, lo) in preds.items()}, "p1000": float(q["p1000"][0]),
               "source": SOURCES[int(np.searchsorted(np.cumsum(rerank.SRC_N), int(pick[0]), side="right"))],
               "cells": [[float(self.leaf_ll[0][i]), float(self.leaf_ll[1][i]), float(w)] for i, w in zip(idx, p / p.sum())]}
        if gt is not None:
            out["truth"] = list(gt)
            out["err_km"] = {m: float(bins.haversine(gt[0], gt[1], *ll)) for m, ll in out["preds"].items()}
            out["nn_train_km"] = float(self.env.nn_km(np.array([gt[0]]), np.array([gt[1]]))[0])
        return out


class Loader(threading.Thread):
    """Builds the Demo off the request path; requests that need it wait on `ready`, then serialize on `lock`."""

    def __init__(self, a):
        super().__init__(daemon=True)
        self.a, self.ready, self.lock, self.demo, self.err = a, threading.Event(), threading.Lock(), None, None

    def run(self):
        try:
            if self.a.bundle:
                fetch_bundle(self.a.bundle, config.ROOT)
            self.demo = Demo(self.a.run, self.a.prior)
        except Exception as e:
            logger.exception("model load failed")
            self.err = f"model failed to load: {e}"
        finally:
            self.ready.set()

    def wait(self, timeout):
        """None once the model is up, else why not (still loading, or the load error)."""
        if not self.ready.wait(timeout):
            return "model still loading, try again shortly"
        return self.err


def handler(loader, origin):
    class H(http.server.BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _cors(self):
            if origin:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Max-Age", "86400")
            self.end_headers()

        def do_GET(self):
            url = urllib.parse.urlsplit(self.path)
            if url.path == "/health":
                err = loader.wait(WAIT_S if "wait" in urllib.parse.parse_qs(url.query) else 0)
                return self._send(503 if loader.err else 200, json.dumps({"ready": err is None, "error": err}).encode())
            page = PAGES.get(url.path)
            if page is None:
                return self._send(404, b'{"error": "not found"}')
            with open(os.path.join(STATIC, page), "rb") as f:
                self._send(200, f.read(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, b'{"error": "not found"}')
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_UPLOAD:
                return self._send(413, json.dumps({"error": f"upload must be 1 B to {MAX_UPLOAD >> 20} MB"}).encode())
            try:
                raw = Image.open(io.BytesIO(self.rfile.read(n)))
                gt, img = exif_gps(raw), ImageOps.exif_transpose(raw).convert("RGB")
            except Exception as e:
                logger.warning("bad upload: %s", e)
                return self._send(400, json.dumps({"error": f"unreadable image: {e}"}).encode())
            if err := loader.wait(WAIT_S):
                return self._send(503, json.dumps({"error": err}).encode())
            with loader.lock:
                out = loader.demo.locate(img, gt)
            self._send(200, json.dumps(out).encode())

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="ft", help="tower run; its @rerank2_lgb dir holds scorer.lgb")
    ap.add_argument("--prior", default="f1", help="run supplying the prior, B0 and B2")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    ap.add_argument("--origin", default=os.getenv("ALLOW_ORIGIN"), help="CORS origin of a separately hosted page")
    ap.add_argument("--bundle", default=os.getenv("BUNDLE"), help="gs:// prefix holding data/ and runs/ to download first")
    a = ap.parse_args()
    loader = Loader(a)
    loader.start()
    http.server.ThreadingHTTPServer((a.host, a.port), handler(loader, a.origin)).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
