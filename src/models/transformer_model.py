"""
Transformer-only model: single-scale FrameEncoder + TemporalTransformerHead.

Used for the Day 5-6 ablation experiment — isolates the transformer's
contribution before combining with the multi-scale backbone.
"""
import torch
import torch.nn as nn

from src.models.backbone import FrameEncoder
from src.models.temporal_transformer import TemporalTransformerHead


class TransformerModel(nn.Module):
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
        self.encoder = FrameEncoder(pretrained=pretrained)
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
