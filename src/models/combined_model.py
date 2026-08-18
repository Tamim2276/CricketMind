"""
Combined model: MultiScaleEncoder + TemporalTransformerHead.

This is the paper's proposed architecture — both upgrades together:
    1. Multi-Scale CNN (3 shared-weight EfficientNetV2-S crops) → richer spatial features
    2. Temporal Transformer (2-layer, 4-head) → better long-range temporal reasoning

Expected accuracy gain over baseline: +3–5pp (targeting 92–94% on CricShot10k).
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
        num_layers:  int = 2,
        num_heads:   int = 4,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.encoder = MultiScaleEncoder(pretrained=pretrained)   # out_dim = 3840
        self.head    = TemporalTransformerHead(
            in_dim=self.encoder.out_dim,   # 3840
            num_layers=num_layers,
            num_heads=num_heads,
            num_classes=num_classes,
            num_frames=num_frames,
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = clip.shape
        # Encode all frames in one batched pass through the 3-scale backbone
        feats = self.encoder(clip.view(b * t, c, h, w))   # (B*T, 3840)
        feats = feats.view(b, t, -1)                       # (B, T, 3840)
        return self.head(feats)                            # (B, num_classes)
