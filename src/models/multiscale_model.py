"""
Multi-Scale CNN + GRU model.

Uses the updated MultiScaleEncoder (with projection layer) + GRUHead.

Key changes vs original:
  - MultiScaleEncoder now outputs 1280-d (projected) instead of raw 1504-d
  - GRU hidden increased from 128 to 256 (more capacity for richer features)
  - dropout=0.5 (regularization)

Use this for ablation: isolates multi-scale contribution before adding
the Transformer head (see combined_model.py for the full upgrade).
"""
import torch
import torch.nn as nn

from src.models.multiscale_cnn import MultiScaleEncoder
from src.models.temporal_gru import GRUHead


class MultiScaleModel(nn.Module):
    """
    Input:  clip (B, T, C, H, W)
    Output: logits (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int   = 15,
        num_frames:  int   = 15,
        pretrained:  bool  = True,
        gru_hidden:  int   = 256,  
        proj_dim:    int   = 1280, 
        dropout:     float = 0.5,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.encoder = MultiScaleEncoder(pretrained=pretrained, proj_dim=proj_dim)
        self.head    = GRUHead(
            in_dim=self.encoder.out_dim,   # 1280
            hidden=gru_hidden,             # 256
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = clip.shape
        feats = self.encoder(clip.view(b * t, c, h, w))   # (B*T, 1280)
        feats = feats.view(b, t, -1)                       # (B, T, 1280)
        return self.head(feats)                            # (B, num_classes)