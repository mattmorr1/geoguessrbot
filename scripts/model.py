import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

R_EARTH_KM = 6371.0088
STREETCLIP = "geolocal/StreetCLIP"
BACKBONES = {"streetclip": STREETCLIP, "siglip2": "google/siglip2-so400m-patch14-384"}


def lorentz_expmap0(v, eps=1e-10):
    norm = v.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)
    return torch.cat([torch.cosh(norm), torch.sinh(norm) * v / norm], dim=-1)

def lorentz_logmap0(y, eps=1e-10):
    time = y[..., :1]
    space = y[..., 1:]
    norm_space = space.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)
    theta = torch.acosh(time.clamp(min=1 + eps))
    return theta * space / norm_space

def lorentz_dist(x, y, eps=1e-12):
    """d = 2 asinh(||x - y||_L / 2); differencing avoids the fp32 cancellation in -<x,y>_L near d = 0."""
    diff = x - y
    sq = -diff[..., 0] ** 2 + (diff[..., 1:] ** 2).sum(dim=-1)
    return 2 * torch.asinh(sq.clamp(min=eps).sqrt() / 2)

def lorentz_cdist(x, y, eps=1e-5):
    """Pairwise geodesic distance [N, M] via one GEMM instead of an [N, M, D] broadcast."""
    inner = x[:, 1:] @ y[:, 1:].T - x[:, :1] @ y[:, :1].T
    return torch.acosh((-inner).clamp(min=1 + eps))

def lorentz_project(x, eps=1e-5):
    space = x[..., 1:]
    time = torch.sqrt(1 + space.pow(2).sum(dim=-1, keepdim=True) + eps)
    return torch.cat([time, space], dim=-1)

def clip_norm(v, max_norm):
    return v * (max_norm / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)).clamp(max=1.0)


# ============================================================
# Embedding Augmentation
# ============================================================
# Training on cached StreetCLIP vectors means no image-level augmentation.
# Inputs are L2-normalized, so noise_std is per-dim on a unit vector (~0.036 per dim).

class EmbeddingAugmentor(nn.Module):
    def __init__(self, noise_std=0.005, feat_drop=0.1):
        super().__init__()
        self.noise_std = noise_std
        self.drop = nn.Dropout(feat_drop)

    def forward(self, x):
        if not self.training:
            return x
        x = self.drop(x)
        return x + torch.randn_like(x) * self.noise_std


# ============================================================
# Contrastive Memory Queue (MoCo-style)
# ============================================================
# Stores recent AlphaEarth targets as extra InfoNCE negatives. Targets are fixed
# (not from a momentum encoder), so queued negatives never go stale.

class ContrastiveQueue:
    def __init__(self, dim=64, size=8192, device="cpu"):
        self.queue = torch.zeros(size, dim, device=device)
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def push(self, batch):
        B = batch.size(0)
        batch = batch.detach().to(self.queue.device, self.queue.dtype)
        end = self.ptr + B
        if end <= len(self.queue):
            self.queue[self.ptr:end] = batch
        else:
            overflow = end - len(self.queue)
            self.queue[self.ptr:] = batch[:B - overflow]
            self.queue[:overflow] = batch[B - overflow:]
        self.full = self.full or end >= len(self.queue)
        self.ptr = end % len(self.queue)

    def get(self, dev):
        if self.full:
            return self.queue.to(dev)
        if self.ptr == 0:
            return torch.empty(0, self.queue.shape[1], device=dev)
        return self.queue[:self.ptr].to(dev)


# ============================================================
# Multi-View Attention Pooling
# ============================================================

class MultiViewEncoder(nn.Module):
    def __init__(self, dim=768, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.dim() == 2:
            return x
        B = x.size(0)
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, x], dim=1)
        out, _ = self.attn(cls, tokens, tokens)
        return self.norm(out.squeeze(1))


# ============================================================
# Bridge Architecture
# ============================================================

class SharedEncoder(nn.Module):
    def __init__(self, in_dim=768, hidden=1024, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        # residual skip: preserves gradient flow through the deep path
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.net(x) + self.skip(x)


class PhysicalHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class CellHead(nn.Module):
    """
    Per-level logits over the nested quadtree. kind: linear | euclid | lorentz.
    Prototype arms share a tree parametrization (child = parent + delta); euclid vs lorentz isolates curvature.
    """

    def __init__(self, in_dim, n_cells, parents, kind="linear", dim=128, max_norm=6.0):
        super().__init__()
        self.kind, self.max_norm = kind, max_norm
        if kind == "linear":
            self.fc = nn.ModuleList(nn.Linear(in_dim, int(n)) for n in n_cells)
            return
        if kind not in ("euclid", "lorentz"):
            raise ValueError(f"unknown cell head {kind!r}")
        self.proj = nn.Linear(in_dim, dim)
        self.delta = nn.ParameterList(
            nn.Parameter(torch.randn(int(n), dim) * (1.0 if l == 0 else 0.3) / math.sqrt(dim))
            for l, n in enumerate(n_cells))
        for l, p in enumerate(parents, 1):
            self.register_buffer(f"parent_{l}", torch.as_tensor(p, dtype=torch.long))
        self.log_scale = nn.Parameter(torch.zeros(len(n_cells)))

    def prototypes(self):
        protos = [self.delta[0]]
        for l in range(1, len(self.delta)):
            protos.append(protos[-1][getattr(self, f"parent_{l}")] + self.delta[l])
        return protos

    def forward(self, z):
        if self.kind == "linear":
            return [fc(z) for fc in self.fc]
        with torch.autocast(z.device.type, enabled=False):
            h, s = self.proj(z.float()), self.log_scale.exp()
            if self.kind == "euclid":
                hh = h.pow(2).sum(-1, keepdim=True)
                return [-(hh - 2 * h @ e.T + e.pow(2).sum(-1)).clamp(min=0) * s[l]
                        for l, e in enumerate(self.prototypes())]
            hx = lorentz_expmap0(clip_norm(h, self.max_norm))
            return [-lorentz_cdist(hx, lorentz_expmap0(clip_norm(e, self.max_norm))) * s[l]
                    for l, e in enumerate(self.prototypes())]


class AuxHeads(nn.Module):
    """OSV5M per-image attributes: categorical (drive_side, climate, land_cover, soil) + regression."""

    def __init__(self, in_dim, n_cat, n_reg=2):
        super().__init__()
        self.cat = nn.ModuleList(nn.Linear(in_dim, int(n)) for n in n_cat)
        self.reg = nn.Linear(in_dim, n_reg)

    def forward(self, z):
        return [h(z) for h in self.cat], self.reg(z)


def dequantize_t(q):
    """Torch twin of geo_lookup.dequantize for int8 AE: (q/127.5)^2 sign(q), masked (-128) -> 0."""
    f = q.float() / 127.5
    return torch.where(q == -128, torch.zeros_like(f), f * f.abs())


def pyramid_feats(M, H, idx):
    """Cell descriptors [B, D, 128] from device pyramid tables (int8 mean AE, uint8 sqrt histogram) at rows idx [B, D]."""
    return torch.cat([dequantize_t(M[idx]), H[idx].float() / 255], -1)


class LocationTower(nn.Module):
    """
    Location -> unit embedding from any '+'-joined inputs: ae (point context [B,18,64] int8), gps (random Fourier
    features of xyz), pyr (AE cell pyramid descriptors [B,n_pyr,128]). GPS phases run in fp32 (fp16 xyz would alias).
    """

    def __init__(self, loc, out_dim=64, n_ctx=18, n_pyr=5, sigmas=(4, 32, 256, 2048), n_freq=128):
        super().__init__()
        parts = set(loc.split("+"))
        if not parts <= {"ae", "gps", "pyr"}:
            raise ValueError(f"unknown location tower {loc!r}")
        self.ctx, self.gps, self.pyr, self.n_pyr, self.out_dim = "ae" in parts, "gps" in parts, "pyr" in parts, n_pyr, out_dim
        if self.gps:
            g = torch.Generator().manual_seed(0)
            self.register_buffer("freq", torch.cat([torch.randn(3, n_freq, generator=g) * s for s in sigmas], 1))
        in_dim = self.ctx * n_ctx * 64 + self.gps * 2 * n_freq * len(sigmas) + self.pyr * n_pyr * 128
        self.net = nn.Sequential(nn.Linear(in_dim, 1024), nn.LayerNorm(1024), nn.GELU(),
                                 nn.Linear(1024, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, out_dim))

    def forward(self, ctx=None, xyz=None, pyr=None):
        parts = [dequantize_t(ctx).flatten(1)] if self.ctx else []
        if self.gps:
            with torch.autocast(xyz.device.type, enabled=False):
                p = xyz.float() @ self.freq
            parts.append(torch.cat([p.cos(), p.sin()], 1))
        if self.pyr:
            parts.append(pyr.flatten(1))
        return F.normalize(self.net(torch.cat(parts, 1)).float(), dim=-1)


class AlphaBridge(nn.Module):
    def __init__(self, n_cells, parents, n_aux, head="linear", ae_dim=64, loc="none", in_dim=768):
        super().__init__()
        self.augment = EmbeddingAugmentor()
        self.views = MultiViewEncoder(in_dim)
        self.encoder = SharedEncoder(in_dim)
        self.cells = CellHead(512, n_cells, parents, head)
        self.physical = PhysicalHead(512, ae_dim)
        self.aux = AuxHeads(512, n_aux)

        # learnable temperature for InfoNCE (CLIP-style)
        # stored as log(1/t) so it stays positive and well-scaled
        self.log_inv_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        # two-tower cross-view arm: image head on the shared trunk vs a learned location tower
        self.tower = None if loc == "none" else LocationTower(loc)
        if self.tower is not None:
            self.loc_head = PhysicalHead(512, self.tower.out_dim)
            self.log_inv_temp_loc = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @property
    def temperature(self):
        return (1 / self.log_inv_temp.exp()).clamp(min=0.005, max=1.0)

    @property
    def loc_temperature(self):
        return (1 / self.log_inv_temp_loc.exp()).clamp(min=0.005, max=1.0)

    def trunk(self, x):
        return self.encoder(self.views(self.augment(F.normalize(x.float(), dim=-1))))

    def forward(self, x):
        z = self.trunk(x)
        aux_cat, aux_reg = self.aux(z)
        out = {'cells': self.cells(z), 'ae': self.physical(z), 'aux_cat': aux_cat, 'aux_reg': aux_reg}
        if self.tower is not None:
            out['loc'] = self.loc_head(z)
        return out


def bridge_spec(cells, in_dim=768):
    """Constructor kwargs from bins.load('cells'); stored in checkpoints so inference needs no meta files."""
    n = [int(v) for v in cells["n_cells"]]
    return {"n_cells": n, "parents": [cells[f"parent_{l}"] for l in range(1, len(n))],
            "n_aux": [int(v) for v in cells["n_aux"]], "in_dim": int(in_dim)}


def load_bridge(path, map_location=device):
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = AlphaBridge(**ck["spec"], head=ck["args"]["head"], loc=ck["args"].get("loc", "none")).to(map_location)
    model.load_state_dict(ck["state"])
    return model.eval(), ck


# ============================================================
# Frozen StreetCLIP (shared by ingest and predict so features match exactly)
# ============================================================

def load_streetclip(dev=device):
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    vision = CLIPVisionModelWithProjection.from_pretrained(STREETCLIP)
    return CLIPImageProcessor.from_pretrained(STREETCLIP), vision.to(dev, torch.float16 if dev == "cuda" else torch.float32).eval()


def load_backbone(name, dev=device):
    """(processor, vision) for a frozen image encoder; SigLIP2 keeps only its vision tower."""
    if name == "streetclip":
        return load_streetclip(dev)
    from transformers import AutoModel, SiglipImageProcessorPil
    vision = AutoModel.from_pretrained(BACKBONES[name]).vision_model
    # PIL backend pinned: same preprocessing in encode and serve, and no torchvision dependency
    return SiglipImageProcessorPil.from_pretrained(BACKBONES[name]), vision.to(dev, torch.float16 if dev == "cuda" else torch.float32).eval()


@torch.no_grad()
def encode_pixels(vision, pixel_values):
    out = vision(pixel_values=pixel_values.to(vision.device, vision.dtype))
    return out.image_embeds if hasattr(out, "image_embeds") else out.pooler_output


# ============================================================
# Loss Functions
# ============================================================

def haversine_soft_ce(logits, xyz, cent_xyz, tau_km):
    """CE against PIGEON-style haversine-smoothed labels q_c = softmax(-d(x, centroid_c) / tau)."""
    with torch.no_grad():
        d = R_EARTH_KM * torch.acos((xyz.double() @ cent_xyz.T).clamp(-1, 1))
        q = F.softmax(-d / tau_km, dim=-1).float()
    return -(q * F.log_softmax(logits.float(), dim=-1)).sum(-1).mean()


def info_nce_loss(pred, target, temperature, queue_negs=None, target_tau=0.0, weight=None):
    """InfoNCE; target_tau > 0 swaps one-hot labels for soft ones from target-target similarity."""
    if queue_negs is not None and queue_negs.shape[0] > 0:
        all_targets = torch.cat([target, queue_negs.to(target.dtype)], dim=0)
    else:
        all_targets = target
    logits = pred @ all_targets.T / temperature
    if target_tau <= 0:
        per = F.cross_entropy(logits.float(), torch.arange(len(pred), device=pred.device), reduction="none")
    else:
        with torch.no_grad():
            w = F.softmax(target.float() @ all_targets.float().T / target_tau, dim=-1)
        per = -(w * F.log_softmax(logits.float(), dim=-1)).sum(-1)
    return per.mean() if weight is None else (per * weight).sum() / weight.sum().clamp(min=1)


def cross_view_loss(u, v, xyz, temperature, tau_km):
    """Symmetric image<->location InfoNCE; soft targets softmax(-d_ij / tau_km) share credit among co-located rows."""
    with torch.no_grad():
        w = F.softmax(-R_EARTH_KM * torch.acos((xyz.double() @ xyz.double().T).clamp(-1, 1)) / tau_km, dim=-1).float()
    logits = u.float() @ v.float().T / temperature
    ce = lambda lg: -(w * F.log_softmax(lg, dim=-1)).sum(-1).mean()
    return 0.5 * (ce(logits) + ce(logits.T))


def aux_loss(outputs, cat_t, reg_t):
    ce = sum(F.cross_entropy(lg.float(), t.long(), ignore_index=-1, reduction="sum") / (t >= 0).sum().clamp(min=1)
             for lg, t in zip(outputs['aux_cat'], cat_t.T))
    m = ~torch.isnan(reg_t)
    reg = (F.smooth_l1_loss(outputs['aux_reg'].float(), torch.nan_to_num(reg_t), reduction="none") * m).sum()
    return ce + reg / m.sum().clamp(min=1)


def bridge_loss(outputs, batch, ctx, weights):
    """
    batch: xyz [B,3] float64, ae [B,64], ae_ok [B] float, aux_cat [B,4], aux_reg [B,2].
    ctx: cent (per-level [C,3] float64), tau (per-level km), queue, temperature, target_tau.
    """
    l_cell = torch.stack([haversine_soft_ce(lg, batch['xyz'], c, t)
                          for lg, c, t in zip(outputs['cells'], ctx['cent'], ctx['tau'])]).mean()
    total, parts = weights['cell'] * l_cell, {'cell': l_cell.detach()}
    if weights['ae'] > 0:
        l_ae = info_nce_loss(outputs['ae'], batch['ae'], ctx['temperature'], ctx['queue'],
                             ctx['target_tau'], weight=batch['ae_ok'])
        total, parts['ae'] = total + weights['ae'] * l_ae, l_ae.detach()
    if weights['aux'] > 0:
        l_aux = aux_loss(outputs, batch['aux_cat'], batch['aux_reg'])
        total, parts['aux'] = total + weights['aux'] * l_aux, l_aux.detach()
    return total, parts
