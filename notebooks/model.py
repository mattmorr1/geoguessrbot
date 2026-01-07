import torch
import torch.nn as nn
import torch.nn.functional as F

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

device = "mps" if torch.cuda.is_available() else "cpu"
scout = AlphaBridge().to(device)
optimizer = torch.optim.AdamW(scout.parameters(), lr=1e-4, weight_decay=1e-2)


def contrastive_loss(pred_alpha, true_alpha, temperature=0.07):
    logits = torch.matmul(pred_alpha, true_alpha.T) / temperature
    labels = torch.arange(pred_alpha.size(0)).to(device)
    return F.cross_entropy(logits, labels)

def train_step(batch_images, batch_coords):
    scout.train()
    optimizer.zero_grad()
    pred_y = scout(batch_visual_x)
    loss = contrastive_loss(pred_y, batch_alpha_y)
    
    loss.backward()
    optimizer.step()
    return loss.item()