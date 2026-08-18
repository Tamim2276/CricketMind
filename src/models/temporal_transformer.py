"""
Lightweight Temporal Transformer head — replaces the GRU in the baseline.

Targets the long-range frame forgetting issue: GRUs struggle to relate
frame 1 (pre-delivery stance) to frame 15 (follow-through), which is
critical for distinguishing shots like Sweep vs. Reverse Sweep.

Design choices (tuned for CricShot10k's 10k-clip dataset size):
    - 2–4 layers, 4 heads: deep enough to capture temporal patterns,
      shallow enough not to overfit on ~7k training clips.
    - Positional embedding: learnable (not sinusoidal) — more flexible.
    - Temporal mean pooling: more stable than CLS-token on small datasets.
    - Dropout 0.1: light regularization.

Do NOT scale this up to ViT-B/16 size — that needs 24 GB+ VRAM and will
overfit immediately on 10k clips.
"""
import torch
import torch.nn as nn


class TemporalTransformerHead(nn.Module):
    """
    Input:  (B, T, in_dim)  — sequence of per-frame features
    Output: (B, num_classes)
    """

    def __init__(
        self,
        in_dim:      int = 1280,
        num_layers:  int = 2,
        num_heads:   int = 4,
        num_classes: int = 15,
        num_frames:  int = 15,
        dropout:     float = 0.1,
    ):
        super().__init__()

        # Learnable positional embedding — one vector per frame position
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, in_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim * 2,   # 2× keeps it lightweight
            dropout=dropout,
            batch_first=True,
            norm_first=True,              # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            enable_nested_tensor=False,   # Pre-LN doesn't support nested tensors; suppress warning
        )

        self.bn   = nn.BatchNorm1d(in_dim)
        self.fc1  = nn.Linear(in_dim, 1024)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.fc2  = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim)
        x = x + self.pos_embed                  # inject positional info
        x = self.transformer(x)                 # (B, T, in_dim)
        x = x.mean(dim=1)                       # temporal mean pool → (B, in_dim)
        x = self.bn(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)                      # (B, num_classes)
