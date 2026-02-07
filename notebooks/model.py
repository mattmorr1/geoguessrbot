import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class AlphaBridge(nn.Module):
    def __init__(self, input_dim=768, output_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.GELU(),

            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        z = self.network(x)
        return F.normalize(z, p=2, dim=1)


class GeoModel(nn.Module):
    """Frozen StreetCLIP (768d) -> trainable AlphaBridge (64d)"""

    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("geolocal/StreetCLIP")
        self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP", use_fast=True)
        self.bridge = AlphaBridge()

        # Freeze StreetCLIP — static weights, only bridge trains
        for param in self.clip.parameters():
            param.requires_grad = False

    def encode_images(self, images):
        """PIL images -> 768d StreetCLIP vectors (no grad)."""
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            clip_vecs = self.clip.get_image_features(**inputs)
        return clip_vecs

    def forward(self, clip_vecs):
        """768d StreetCLIP vectors -> 64d AlphaEarth-aligned vectors."""
        return self.bridge(clip_vecs)

    def predict(self, images):
        """End-to-end: PIL images -> 64d vectors."""
        clip_vecs = self.encode_images(images)
        return self.bridge(clip_vecs)


# --- Training setup ---
model = GeoModel().to(device)
optimizer = torch.optim.AdamW(model.bridge.parameters(), lr=1e-4, weight_decay=1e-2)


def contrastive_loss(pred_alpha, true_alpha, temperature=0.07):
    logits = torch.matmul(pred_alpha, true_alpha.T) / temperature
    labels = torch.arange(pred_alpha.size(0)).to(device)
    return F.cross_entropy(logits, labels)


def train_step(images, alpha_vecs):
    """
    Args:
        images: list of PIL images
        alpha_vecs: Tensor of shape (B, 64) — AlphaEarth target embeddings
    """
    model.train()
    optimizer.zero_grad()

    clip_vecs = model.encode_images(images)
    pred_y = model(clip_vecs)
    loss = contrastive_loss(pred_y, alpha_vecs)

    loss.backward()
    optimizer.step()
    return loss.item()
