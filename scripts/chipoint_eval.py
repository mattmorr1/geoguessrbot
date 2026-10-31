#!/usr/bin/env python3
"""
Chipoint v1 (huggingface.co/chiikabu-labs/chipoint, MIT) as a baseline on our exact official-test folds: their three
frozen encoders + projection heads, top-M retrieval against their 4.89M OSV-5M gallery, and their locked
density-weighted consensus, re-implemented batched and resumable. fp16 throughout (sm_75 has no bf16).

Stages (enc/ret/pred/check run in the Chipoint venv; score runs in the project venv, which has bins):
  python chipoint_eval.py check             # re-encode train photos that sit in their gallery; cos must be ~1
  python chipoint_eval.py enc  [--tag T]    # test images -> projected 512-d per encoder (memmap, resumable)
  python chipoint_eval.py ret               # top-200 gallery hits per encoder
  python chipoint_eval.py pred              # fused top-64 candidates + consensus pick -> pred.npz
  python chipoint_eval.py score             # folds 1-9 metrics, paired CIs vs our reranker -> chipoint_v1.json
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)
CP = os.path.expanduser(os.getenv("CHIPOINT_DIR", "~/chipoint"))
W, IMG, OUT = (os.path.join(CP, d) for d in ("weights", "test_images", "out"))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M, K = 200, 64                                          # retrieve top-M per encoder, keep top-K fused candidates
GV, LAM, MU, BW1, BW2 = 0.8, 1.7, 0.8, 1492.7, 750.0     # their locked consensus config
ENC_D = {"so400m256": 1152, "dinov3": 1024, "gopt": 1536}  # dict order = their fusion order
CELLS = (0.25, 0.5, 1.0, 2.0)
dev = "cuda" if torch.cuda.is_available() else "cpu"


class Head(nn.Module):
    def __init__(s, d):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024, 512))

    def forward(s, x):
        return F.normalize(s.net(x), dim=-1)


def load_head(tag):
    h, sd = Head(ENC_D[tag]), torch.load(os.path.join(W, f"proj_head_{tag}.pt"), map_location="cpu", weights_only=True)
    h.load_state_dict({k: v for k, v in sd.items() if k.startswith("net.")})
    return h.to(dev).eval()


def load_encoder(tag):
    """-> (forward(x) -> unit embedding, PIL -> tensor transform), matching run_chipoint.py's preprocessing."""
    if tag != "dinov3":
        import open_clip
        m, _, pre = open_clip.create_model_and_transforms(
            {"so400m256": "ViT-SO400M-16-SigLIP2-256", "gopt": "ViT-gopt-16-SigLIP2-256"}[tag], pretrained="webli")
        # image tower only, fp16 weights: fp32 gopt + autocast copies overflow 8 GB and WDDM spills to host RAM (~1 img/s)
        v = m.visual.half().to(dev).eval()
        del m

        def fwd(x):
            return F.normalize(v(x.half()).float(), dim=-1)
        return fwd, pre
    import torchvision.transforms as T
    from modelscope import snapshot_download
    from transformers import AutoImageProcessor, AutoModel
    src = os.getenv("DINOV3_PATH") or snapshot_download("facebook/dinov3-vitl16-pretrain-lvd1689m")
    dm, proc = AutoModel.from_pretrained(src).to(dev).eval(), AutoImageProcessor.from_pretrained(src)
    mean, std = torch.tensor(proc.image_mean).view(3, 1, 1), torch.tensor(proc.image_std).view(3, 1, 1)
    pre = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Lambda(lambda t: (t - mean) / std)])
    skip = 1 + dm.config.num_register_tokens

    # fp32: fp16 overflows to NaN. Patch-token mean matches their gallery (cos 0.94); their config's pooler_output gives 0.31.
    def fwd(x):
        return F.normalize(dm(x).last_hidden_state[:, skip:].mean(1), dim=-1)
    return fwd, pre


class Images(torch.utils.data.Dataset):
    def __init__(s, paths, pre):
        s.paths, s.pre = paths, pre

    def __len__(s):
        return len(s.paths)

    def __getitem__(s, i):
        from PIL import Image
        return s.pre(Image.open(s.paths[i]).convert("RGB"))


@torch.no_grad()
def encode(tag, paths, out_path, bs=64, chunk=8192):
    """paths -> projected [N,512] fp16 memmap at out_path; resumes from the last finished chunk."""
    n, done_path = len(paths), out_path + ".done"
    mm = np.lib.format.open_memmap(out_path, "r+" if os.path.exists(out_path) else "w+", np.float16, (n, 512))
    start = int(open(done_path).read()) if os.path.exists(done_path) else 0
    if start >= n:
        return
    fwd, pre = load_encoder(tag)
    head = load_head(tag)
    for c in range(start, n, chunk):
        dl = torch.utils.data.DataLoader(Images(paths[c:c + chunk], pre), batch_size=bs, num_workers=10, pin_memory=True)
        mm[c:c + chunk] = torch.cat([head(fwd(x.to(dev, non_blocking=True))).half().cpu() for x in dl]).numpy()
        mm.flush()
        open(done_path, "w").write(str(min(c + chunk, n)))
        logger.info("%s: %d / %d", tag, min(c + chunk, n), n)


@torch.no_grad()
def retrieve(tag, q, qb=2048, gb=1 << 17):
    """q [N,512] fp16 -> top-M gallery (idx int32, sim fp32) per query; gallery chunks stream through the GPU once."""
    g = np.load(os.path.join(W, f"proj_head_gallery_512_{tag}.npy"), mmap_mode="r")
    n = len(q)
    bs, bi = torch.full((n, M), -1e9, device=dev), torch.zeros((n, M), dtype=torch.int32, device=dev)
    for i in range(0, len(g), gb):
        gg = torch.from_numpy(np.ascontiguousarray(g[i:i + gb], dtype=np.float16)).to(dev)
        for s in range(0, n, qb):
            cs, ci = (torch.from_numpy(q[s:s + qb]).to(dev) @ gg.T).topk(min(M, len(gg)), 1)
            al, ai = torch.cat([bs[s:s + qb], cs.float()], 1), torch.cat([bi[s:s + qb], (ci + i).int()], 1)
            o = al.topk(M, 1).indices
            bs[s:s + qb], bi[s:s + qb] = al.gather(1, o), ai.gather(1, o)
        logger.info("%s: gallery %d / %d", tag, min(i + gb, len(g)), len(g))
    return bi.cpu().numpy(), bs.cpu().numpy()


def zc(a, axis=-1):
    return (a - a.mean(axis, keepdims=True)) / (a.std(axis, keepdims=True) + 1e-6)


def cell_key(ll, c):
    return (np.round(ll[..., 0] / c).astype(np.int64) << 20) ^ (np.round(ll[..., 1] / c).astype(np.int64) & 0xFFFFF)


def hav(a, b):
    la, lo, lb, ob = (np.radians(x) for x in (a[..., 0], a[..., 1], b[..., 0], b[..., 1]))
    h = np.sin((lb - la) / 2) ** 2 + np.cos(la) * np.cos(lb) * np.sin((ob - lo) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def fuse(R):
    """Per query: sum of per-encoder z-scored sims over the union of hits -> top-K ids/scores (their dict order on ties)."""
    n = len(R["so400m256"][0])
    ci, cs = np.zeros((n, K), np.int64), np.zeros((n, K), np.float32)
    z = {t: zc(R[t][1].astype(np.float64)) for t in ENC_D}
    for i in range(n):
        ids = np.concatenate([R[t][0][i] for t in ENC_D]).astype(np.int64)
        u, first, inv = np.unique(ids, return_index=True, return_inverse=True)
        sc = np.bincount(inv, np.concatenate([z[t][i] for t in ENC_D]))
        o = np.lexsort((first, -sc))[:K]                     # score desc, then first insertion
        ci[i, :len(o)], cs[i, :len(o)] = u[o], sc[o]
    return ci, cs


def consensus(ci, cs, gc, bs=4096):
    """Their density-weighted consensus over the K fused candidates -> chosen column per query."""
    dens = {}
    for c in CELLS:
        k, cnt = np.unique(cell_key(gc, c), return_counts=True)
        dens[c] = (k, cnt)
    look = lambda c, q: dens[c][1][np.searchsorted(dens[c][0], q)]
    pick = np.zeros(len(ci), np.int64)
    for s in range(0, len(ci), bs):
        cc, csim = gc[ci[s:s + bs]], cs[s:s + bs]                # float32 like theirs, so cell keys round identically
        dw = np.mean([-np.log(look(c, cell_key(cc, c)) + 1.0) for c in CELLS], 0)
        d = hav(cc[:, :, None], cc[:, None, :])
        vote = np.maximum(csim, 0) * np.exp(GV * zc(dw))
        kv1, kv2 = ((vote[:, None, :] * np.exp(-d / bw)).sum(-1) for bw in (BW1, BW2))
        pick[s:s + bs] = (zc(csim) + 0.3 * zc(dw) + LAM * zc(kv1) + MU * zc(kv2)).argmax(1)
    return pick


def test_paths():
    te = pd.read_csv(os.path.join(ROOT, "data", "osv5m", "test.csv"), usecols=["id"])
    have = {int(f[:-4]): os.path.join(r, f) for r, _, fs in os.walk(IMG) for f in fs if f.endswith(".jpg")}
    missing = [i for i in te["id"] if i not in have]
    if missing:
        raise RuntimeError(f"{len(missing)} test images missing under {IMG}, e.g. {missing[:5]}")
    return [have[i] for i in te["id"]]


def check(n=512):
    """Encode n train photos from one train zip and compare with their own gallery rows. Gallery order is not
    train.csv order, so rows are matched by coordinates (all rows within 10 m; best cosine counts)."""
    from huggingface_hub import hf_hub_download
    from scipy.spatial import cKDTree
    import zipfile
    tr = pd.read_csv(os.path.join(ROOT, "data", "osv5m", "train.csv"), usecols=["id", "latitude", "longitude"]).set_index("id")
    gc = np.load(os.path.join(W, "gallery_all_C_so400m256.npy")).astype(np.float64)
    xyz = lambda ll: np.stack([np.cos(np.radians(ll[:, 0])) * np.cos(np.radians(ll[:, 1])),
                               np.cos(np.radians(ll[:, 0])) * np.sin(np.radians(ll[:, 1])), np.sin(np.radians(ll[:, 0]))], 1)
    d = os.path.join(CP, "check_imgs")
    if not os.path.isdir(d):
        zp = hf_hub_download("osv5m/osv5m", "images/train/00.zip", repo_type="dataset", local_dir=os.path.join(CP, "osv5m"))
        with zipfile.ZipFile(zp) as z:
            z.extractall(d, [x for x in z.namelist() if x.endswith(".jpg")][:n])
        os.remove(zp)
    paths = sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith(".jpg"))[:n]
    ll = tr.loc[[int(os.path.basename(x)[:-4]) for x in paths], ["latitude", "longitude"]].to_numpy()
    near = cKDTree(xyz(gc)).query_ball_point(xyz(ll), 0.01 / 6371.0)
    logger.info("check: %d photos, gallery rows within 10 m per photo: min %d median %d", len(paths),
                min(map(len, near)), int(np.median(list(map(len, near)))))
    res = {}
    for tag in ENC_D:
        p = os.path.join(OUT, f"check_{tag}.npy")
        encode(tag, paths, p)
        g, q = np.load(os.path.join(W, f"proj_head_gallery_512_{tag}.npy"), mmap_mode="r"), np.load(p).astype(np.float32)
        cos = np.array([(g[sorted(r)].astype(np.float32) @ v).max() if r else np.nan for v, r in zip(q, near)])
        res[tag] = {k: float(f(cos)) for k, f in (("min", np.nanmin), ("p5", lambda c: np.nanpercentile(c, 5)),
                                                   ("median", np.nanmedian), ("unmatched", lambda c: np.isnan(c).sum()))}
        logger.info("check %s: cos to own gallery row %s", tag, res[tag])
    json.dump(res, open(os.path.join(OUT, "check.json"), "w"), indent=1)


def score():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bins
    from scipy.spatial import cKDTree
    from eval import CO_PRIMARY, bootstrap, distance_metrics
    p = np.load(os.path.join(OUT, "pred.npz"))
    meta, folds = bins.load("test"), bins.test_folds()
    assert len(p["lat"]) == len(folds) == len(meta["lat"])
    d = hav(np.stack([p["lat"], p["lon"]], -1), np.stack([meta["lat"], meta["lon"]], -1))
    ours_rows = bins.encoded_rows("test", "fusion768")
    ours_rows = ours_rows[~meta["tune"][ours_rows]]
    z = np.load(os.path.join(ROOT, "runs", "ft@rerank2_lgb", "test_dists.npz"))
    ours = z["RERANK" if "RERANK" in z else "lgb_mbr"]                # lgb_mbr: pre-cleanup rerank.py key
    assert len(ours) == len(ours_rows)
    tr = bins.load("train")
    seen = ~tr["holdout"] if "holdout" in tr else slice(None)      # same definition as eval.Env.nn_km
    chord, _ = cKDTree(bins.to_xyz(tr["lat"][seen], tr["lon"][seen])).query(
        bins.to_xyz(meta["lat"][ours_rows], meta["lon"][ours_rows]), workers=-1)
    far = 2 * bins.R_EARTH_KM * np.arcsin(np.clip(chord / 2, 0, 1)) > 25
    m19 = folds != 0
    res = {"chipoint_v1": {"folds1_9": distance_metrics(d[m19]), "fold0": distance_metrics(d[folds == 0]),
                           "all": distance_metrics(d), "n_folds1_9": int(m19.sum())}}
    for name, m in (("all", np.ones(len(ours), bool)), ("far25", far)):
        a, b = ours[m], d[ours_rows][m]
        res[f"paired_{name}"] = {"n": int(m.sum()), "ours": distance_metrics(a), "chipoint_v1": distance_metrics(b),
                                 "chipoint_minus_ours": {c: [float(f(b) - f(a)), bootstrap(lambda x, y, f=f: f(y) - f(x), a, b)]
                                                         for c, f in CO_PRIMARY.items()}}
    out = os.path.join(ROOT, "runs", "chipoint_v1.json")
    json.dump(res, open(out, "w"), indent=1, default=float)
    logger.info("%s", json.dumps({k: (v.get("chipoint_minus_ours") or v.get("folds1_9")) for k, v in res.items()}, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["check", "enc", "ret", "pred", "score"])
    ap.add_argument("--tag", choices=list(ENC_D), help="enc: one encoder only (default: all, in order)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.stage == "check":
        check()
    elif a.stage == "enc":
        paths = test_paths()
        for tag in [a.tag] if a.tag else ENC_D:
            encode(tag, paths, os.path.join(OUT, f"q_{tag}.npy"))
    elif a.stage == "ret":
        for tag in ENC_D:
            if not os.path.exists(f := os.path.join(OUT, f"hits_{tag}.npz")):
                np.savez(f, *retrieve(tag, np.load(os.path.join(OUT, f"q_{tag}.npy"))))
    elif a.stage == "pred":
        R = {t: tuple(np.load(os.path.join(OUT, f"hits_{t}.npz"))[k] for k in ("arr_0", "arr_1")) for t in ENC_D}
        gc = np.load(os.path.join(W, "gallery_all_C_so400m256.npy")).astype(np.float32)
        ci, cs = fuse(R)
        pick = consensus(ci, cs, gc)
        ll = gc[ci[np.arange(len(ci)), pick]]
        np.savez(os.path.join(OUT, "pred.npz"), lat=ll[:, 0], lon=ll[:, 1], cand=ci, cand_score=cs, pick=pick)
    else:
        score()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
