import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# ============================================================
# Lorentz Manifold Geometry
# ============================================================
# Switched from Poincare ball to Lorentz hyperboloid.
# Poincare has gradient explosion near the boundary (norm->1).
# Lorentz avoids this entirely -- gradients stay well-behaved
# because the manifold is unbounded in ambient space.

def lorentz_expmap0(v, eps=1e-10):
    """Tangent space at origin -> Lorentz hyperboloid.
    Input:  (B, d)   tangent vector
    Output: (B, d+1) point on H^d, first dim is time component
    """
    norm = v.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)
    return torch.cat([torch.cosh(norm), torch.sinh(norm) * v / norm], dim=-1)

def lorentz_logmap0(y, eps=1e-10):
    """Lorentz hyperboloid -> tangent space at origin."""
    time = y[..., :1]
    space = y[..., 1:]
    norm_space = space.norm(p=2, dim=-1, keepdim=True).clamp(min=eps)
    theta = torch.acosh(time.clamp(min=1 + eps))
    return theta * space / norm_space

def lorentz_dist(x, y, eps=1e-5):
    """Geodesic distance on Lorentz hyperboloid.
    d(x,y) = arccosh(-<x,y>_L)
    """
    inner = -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)
    return torch.acosh((-inner).clamp(min=1 + eps))

def lorentz_project(x, eps=1e-5):
    """Re-project onto hyperboloid: fix x0 = sqrt(1 + ||x_space||^2)."""
    space = x[..., 1:]
    time = torch.sqrt(1 + space.pow(2).sum(dim=-1, keepdim=True) + eps)
    return torch.cat([time, space], dim=-1)


# ============================================================
# Multi-View Attention Pooling (PIGEON-style)
# ============================================================
# When 4 cardinal patches are available, aggregate via learned
# attention. Falls through to identity for single-view input,
# so the same model works with or without multi-view data.

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

    def forward(self, x):
        return self.net(x)


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
    """Projects to 128D tangent space, then lifts to 129D Lorentz hyperboloid."""
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
    """Decode climate zone and elevation from the predicted 64D DNA.
    Forces the DNA vector to encode meaningful environmental semantics
    instead of being an uninterpretable black box.
    """
    NUM_CLIMATE = 6   # tropical, arid, temperate, continental, polar, unknown
    NUM_ELEV = 8      # <0m, 0-100, 100-500, 500-1k, 1k-2k, 2k-3k, 3k-5k, >5k

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

    def forward(self, x):
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
# Full Model: Frozen StreetCLIP + Trainable Bridge
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
        """Encode N views per location.
        image_groups: list of (list of PIL images), inner list = views of same place
        Returns: (B, N, 768)
        """
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

def info_nce_loss(pred, target, temperature=0.07):
    logits = pred @ target.T / temperature
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
# Country classification is the easiest signal -- high alpha early.
# DNA contrastive alignment is harder -- ramp beta up over time.
# Hierarchy and auxiliary are regularizers -- ramp gamma/aux slowly.

class CurriculumScheduler:
    def __init__(self, total_epochs, warmup=2):
        self.total = total_epochs
        self.warmup = warmup

    def get_weights(self, epoch):
        t = min(epoch / max(self.total - 1, 1), 1.0)
        if epoch < self.warmup:
            return {'alpha': 2.0, 'beta': 0.3, 'gamma': 0.1, 'aux': 0.1}
        return {
            'alpha': 2.0 - 1.5 * t,  # 2.0 -> 0.5
            'beta':  0.3 + 1.7 * t,  # 0.3 -> 2.0
            'gamma': 0.1 + 0.9 * t,  # 0.1 -> 1.0
            'aux':   0.1 + 0.4 * t,  # 0.1 -> 0.5
        }


def scout_loss(outputs, targets, hier_head, weights=None):
    if weights is None:
        weights = {'alpha': 1.0, 'beta': 1.0, 'gamma': 0.5, 'aux': 0.3}

    l_admin = F.cross_entropy(outputs['admin_logits'], targets['country_idx'])
    l_dna = info_nce_loss(outputs['dna'], targets['alpha_vecs'])
    l_hier = hierarchy_loss(
        outputs['hyp'], targets['continent_idx'], targets['country_idx'], hier_head
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
    }
