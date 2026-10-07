"""
Combined model: MultiScaleEncoder (with projection) + TemporalTransformerHead.

This is the full proposed architecture — both upgrades together:
  1. Multi-Scale CNN with learned projection (1504 -> 1280)
     Captures fine bat angle + mid body shape + global pose,
     combined by a learned projection instead of raw concatenation.
  2. Temporal Transformer (2-layer, 4-head, mean pooling)
     Long-range frame attention: frame 1 stance directly informs
     frame 15 follow-through without GRU forgetting.
"""
import torch
import torch.nn as nn

from src.models.multiscale_cnn import MultiScaleEncoder
from src.models.temporal_transformer import TemporalTransformerHead


class CombinedModel(nn.Module):
    """
    Input:  clip (B, T, C, H, W)
    Output: logits (B, num_classes)
    """

    def __init__(
        self,
        num_classes: int = 15,
        num_frames:  int = 15,
        pretrained:  bool = True,
        proj_dim:    int = 1280,   # MultiScaleEncoder output dim
        num_layers:  int = 2,
        num_heads:   int = 4,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.encoder = MultiScaleEncoder(pretrained=pretrained, proj_dim=proj_dim)
        self.head    = TemporalTransformerHead(
            in_dim=self.encoder.out_dim,   # 1280
            num_layers=num_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            num_frames=num_frames,
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = clip.shape
        feats = self.encoder(clip.view(b * t, c, h, w))   # (B*T, 1280)
        feats = feats.view(b, t, -1)                       # (B, T, 1280)
        return self.head(feats)                            # (B, num_classes)