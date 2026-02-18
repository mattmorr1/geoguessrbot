import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import math

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ============================================================
# Lorentz Manifold Geometry
# ============================================================

def lorentz_expmap0(v, eps=1e-10):
    norm = v.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)
    return torch.cat([torch.cosh(norm), torch.sinh(norm) * v / norm], dim=-1)

def lorentz_logmap0(y, eps=1e-10):
    time = y[..., :1]
    space = y[..., 1:]
    norm_space = space.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)
    theta = torch.acosh(time.clamp(min=1 + eps))
    return theta * space / norm_space

def lorentz_dist(x, y, eps=1e-5):
    inner = -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    return torch.acosh((-inner).clamp(min=1 + eps))

def lorentz_project(x, eps=1e-5):
    space = x[..., 1:]
    time = torch.sqrt(1 + space.pow(2).sum(dim=-1, keepdim=True) + eps)
    return torch.cat([time, space], dim=-1)


# ============================================================
# Embedding Augmentation
# ============================================================
# Training on cached StreetCLIP vectors means no image-level augmentation.
# This compensates: Gaussian noise simulates CLIP encoding variance,
# feature dropout forces the bridge to not over-rely on any single
# dimension of the 768d input.

class EmbeddingAugmentor(nn.Module):
    def __init__(self, noise_std=0.02, feat_drop=0.1):
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
# InfoNCE quality scales with number of negatives. With batch=32,
# you only get 31 negatives per sample. This queue stores recent
# AlphaEarth targets, giving thousands of additional negatives
# without increasing batch size or VRAM.

class ContrastiveQueue:
    def __init__(self, dim=64, size=8192):
        self.queue = torch.zeros(size, dim)
        self.ptr = 0
        self.full = False

    @torch.no_grad()
    def push(self, batch):
        B = batch.size(0)
        batch = batch.detach().cpu()
        end = self.ptr + B
        if end <= len(self.queue):
            self.queue[self.ptr:end] = batch
        else:
            overflow = end - len(self.queue)
            self.queue[self.ptr:] = batch[:B - overflow]
            self.queue[:overflow] = batch[B - overflow:]
            self.full = True
        self.ptr = end % len(self.queue)

    def get(self, dev):
        if self.full:
            return self.queue.to(dev)
        if self.ptr == 0:
            return None
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
# Multi-Head Bridge Architecture
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


class AdministrativeHead(nn.Module):
    def __init__(self, in_dim=512, num_classes=250):
        super().__init__()
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(self.drop(x))


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


class HierarchicalHead(nn.Module):
    def __init__(self, in_dim=512, hyp_dim=128, num_continents=7, num_countries=250):
        super().__init__()
        self.hyp_dim = hyp_dim
        self.proj = nn.Linear(in_dim, hyp_dim)
        self.continent_embeds = nn.Embedding(num_continents, hyp_dim)
        self.country_embeds = nn.Embedding(num_countries, hyp_dim)
        nn.init.uniform_(self.continent_embeds.weight, -0.01, 0.01)
        nn.init.uniform_(self.country_embeds.weight, -0.05, 0.05)

    def tangent(self, x):
        return self.proj(x)

    def forward(self, x):
        return lorentz_expmap0(self.tangent(x))

    def embed_continents(self, idx):
        return lorentz_expmap0(self.continent_embeds(idx))

    def embed_countries(self, idx):
        return lorentz_expmap0(self.country_embeds(idx))


class AuxiliaryDecoder(nn.Module):
    NUM_CLIMATE = 6
    NUM_ELEV = 8

    def __init__(self, dna_dim=64):
        super().__init__()
        self.climate = nn.Sequential(
            nn.Linear(dna_dim, 32), nn.GELU(), nn.Linear(32, self.NUM_CLIMATE),
        )
        self.elevation = nn.Sequential(
            nn.Linear(dna_dim, 32), nn.GELU(), nn.Linear(32, self.NUM_ELEV),
        )

    def forward(self, dna):
        return {
            'climate_logits': self.climate(dna),
            'elevation_logits': self.elevation(dna),
        }


class AlphaBridge(nn.Module):
    def __init__(self, num_countries=250, num_continents=7, hyp_dim=128):
        super().__init__()
        self.augment = EmbeddingAugmentor()
        self.views = MultiViewEncoder()
        self.encoder = SharedEncoder()
        self.admin = AdministrativeHead(num_classes=num_countries)
        self.physical = PhysicalHead()
        self.hierarchical = HierarchicalHead(
            num_continents=num_continents,
            num_countries=num_countries,
            hyp_dim=hyp_dim,
        )
        self.aux = AuxiliaryDecoder()

        # learnable temperature for InfoNCE (CLIP-style)
        # stored as log(1/t) so it stays positive and well-scaled
        self.log_inv_temp = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    @property
    def temperature(self):
        return (1 / self.log_inv_temp.exp()).clamp(min=0.005, max=1.0)

    def forward(self, x):
        x = self.augment(x)
        x = self.views(x)
        z = self.encoder(x)
        tangent = self.hierarchical.tangent(z)
        dna = self.physical(z)
        aux_out = self.aux(dna)
        return {
            'admin_logits': self.admin(z),
            'dna': dna,
            'hyp': lorentz_expmap0(tangent),
            'hyp_tangent': tangent,
            'climate_logits': aux_out['climate_logits'],
            'elevation_logits': aux_out['elevation_logits'],
        }


# ============================================================
# Full Model
# ============================================================

class AlphaEarthScout(nn.Module):
    def __init__(self, num_countries=250, num_continents=7):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("geolocal/StreetCLIP")
        self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP", use_fast=True)
        self.bridge = AlphaBridge(num_countries=num_countries, num_continents=num_continents)
        for p in self.clip.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode_images(self, images):
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(device)
        return self.clip.get_image_features(**inputs)

    @torch.no_grad()
    def encode_multi_view(self, image_groups):
        all_vecs = []
        for views in image_groups:
            inputs = self.processor(images=views, return_tensors="pt", padding=True).to(device)
            all_vecs.append(self.clip.get_image_features(**inputs))
        return torch.stack(all_vecs)

    def forward(self, clip_vecs):
        return self.bridge(clip_vecs)

    def predict(self, images):
        return self.bridge(self.encode_images(images))

    def predict_multi_view(self, image_groups):
        return self.bridge(self.encode_multi_view(image_groups))

    def infer(self, images, idx_to_cc):
        out = self.predict(images)
        probs = F.softmax(out['admin_logits'], dim=-1)
        conf, cidx = probs.max(dim=-1)
        countries = [idx_to_cc[i.item()] for i in cidx]
        return {
            'countries': countries,
            'confidences': conf,
            'dna': out['dna'],
            'hyp': out['hyp'],
            'hyp_tangent': out['hyp_tangent'],
            'admin_logits': out['admin_logits'],
        }


# ============================================================
# Loss Functions
# ============================================================

def info_nce_loss(pred, target, temperature, queue_negs=None):
    if queue_negs is not None:
        all_targets = torch.cat([target, queue_negs], dim=0)
    else:
        all_targets = target
    logits = pred @ all_targets.T / temperature
    labels = torch.arange(len(pred), device=pred.device)
    return F.cross_entropy(logits, labels)


def hierarchy_loss(hyp_embed, continent_idx, country_idx, hier_head):
    hyp_embed = hyp_embed.float()
    c_embed = hier_head.embed_continents(continent_idx).float()
    k_embed = hier_head.embed_countries(country_idx).float()
    d_country = lorentz_dist(hyp_embed, k_embed).mean()
    d_continent = lorentz_dist(hyp_embed, c_embed).mean()
    d_parent = lorentz_dist(k_embed, c_embed).mean()
    return d_country + 0.3 * d_continent + 0.2 * d_parent


def auxiliary_loss(outputs, targets):
    l_climate = F.cross_entropy(outputs['climate_logits'], targets['climate_zone'])
    l_elev = F.cross_entropy(outputs['elevation_logits'], targets['elevation_bin'])
    return l_climate + l_elev


# ============================================================
# Curriculum Scheduler
# ============================================================

class CurriculumScheduler:
    def __init__(self, total_epochs, warmup=2):
        self.total = total_epochs
        self.warmup = warmup

    def get_weights(self, epoch):
        t = min(epoch / max(self.total - 1, 1), 1.0)
        if epoch < self.warmup:
            return {'alpha': 2.0, 'beta': 0.3, 'gamma': 0.1, 'aux': 0.1}
        return {
            'alpha': 2.0 - 1.5 * t,
            'beta':  0.3 + 1.7 * t,
            'gamma': 0.1 + 0.9 * t,
            'aux':   0.1 + 0.4 * t,
        }


def scout_loss(outputs, targets, hier_head, temperature, weights=None,
               queue_negs=None, label_smoothing=0.1):
    if weights is None:
        weights = {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.5, 'aux': 0.3}

    # label smoothing on country CE -- reverse geocoding is noisy near borders
    l_admin = F.cross_entropy(
        outputs['admin_logits'], targets['country_idx'],
        label_smoothing=label_smoothing,
    )
    l_dna = info_nce_loss(
        outputs['dna'], targets['alpha_vecs'], temperature, queue_negs,
    )
    l_hier = hierarchy_loss(
        outputs['hyp'], targets['continent_idx'], targets['country_idx'], hier_head,
    )
    l_aux = auxiliary_loss(outputs, targets)

    total = (
        weights['alpha'] * l_admin
        + weights['beta'] * l_dna
        + weights['gamma'] * l_hier
        + weights['aux'] * l_aux
    )
    return total, {
        'admin': l_admin.item(),
        'dna': l_dna.item(),
        'hier': l_hier.item(),
        'aux': l_aux.item(),
        'temp': temperature.item() if hasattr(temperature, 'item') else temperature,
    }
