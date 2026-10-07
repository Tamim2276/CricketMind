"""
Lightweight Temporal Transformer head — replaces the GRU in the baseline.

Targets the long-range frame forgetting issue: GRUs struggle to relate
frame 1 (pre-delivery stance) to frame 15 (follow-through), which is
critical for distinguishing shots like Sweep vs. Reverse Sweep.

Design choices (tuned for CricShot10k's ~7k training clips):
    - 2 layers, 4 heads: deep enough to capture temporal patterns,
      shallow enough not to overfit on small dataset.
    - Positional embedding: learnable (not sinusoidal) — more flexible
      for short fixed-length sequences (T=15).
    - Temporal mean pooling: more stable than CLS-token on small datasets.
      CLS-token needs more data to learn what to aggregate.
    - norm_first=True (Pre-LN): more stable gradient flow than Post-LN,
      especially important on small datasets where loss can spike.
    - dim_feedforward = in_dim * 4: standard Transformer ratio (not 2x).
      Previous version used 2x which underfits — 4x adds needed capacity
      without exceeding 12GB VRAM at batch=4.

CHANGES vs previous version:
    - dim_feedforward: in_dim * 2 -> in_dim * 4  (standard ratio, more capacity)
    - Added attn_dropout param separate from classifier dropout for clarity
    - Works with both FrameEncoder (1280-d) and MultiScaleEncoder (1280-d projected)
    - enable_nested_tensor=False kept: required for norm_first=True (Pre-LN)
"""
import torch
import torch.nn as nn


class TemporalTransformerHead(nn.Module):
    """
    Input:  (B, T, in_dim)  — sequence of per-frame features
    Output: (B, num_classes)

    Args:
        in_dim:       feature dim from encoder (1280 for both baseline and multiscale)
        num_layers:   transformer encoder layers (2 = lightweight, safe for 7k clips)
        num_heads:    attention heads (4 = divides 1280 evenly → head_dim=320)
        num_classes:  output classes (15 for CricShot10k)
        num_frames:   sequence length (15 uniformly sampled frames)
        attn_dropout: dropout inside attention layers (light: 0.1)
        cls_dropout:  dropout before final classifier (strong: 0.5)
    """

    def __init__(
        self,
        in_dim:       int   = 1280,
        num_layers:   int   = 2,
        num_heads:    int   = 4,
        num_classes:  int   = 15,
        num_frames:   int   = 15,
        attn_dropout: float = 0.1,   # light dropout inside transformer layers
        cls_dropout:  float = 0.5,   # strong dropout before classifier
    ):
        super().__init__()

        # Positional embedding
        # Learnable: one vector per frame position (T=15)
        # Initialized near zero (trunc_normal std=0.02) so training starts
        # from near-zero positional bias and learns what matters.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_frames, in_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=in_dim * 4,   
            dropout=attn_dropout,          
            batch_first=True,
            norm_first=True,             
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,   
        )

        # Classifier head
        self.bn   = nn.BatchNorm1d(in_dim)
        self.fc1  = nn.Linear(in_dim, 1024)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=cls_dropout)
        self.fc2  = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 1. Inject positional information
        x = x + self.pos_embed  

        # 2. Temporal self-attention across all 15 frames
        x = self.transformer(x)

        # 3. Temporal mean pooling
        x = x.mean(dim=1)

        # 4. Classify
        x = self.bn(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)