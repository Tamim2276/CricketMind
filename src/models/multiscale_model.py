"""
Multi-Scale CNN only model: MultiScaleEncoder + GRU head.

Used for the Day 3-4 ablation experiment — isolates the contribution of
multi-scale feature extraction without the transformer change.
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
        num_classes: int = 15,
        num_frames:  int = 15,
        pretrained:  bool = True,
        gru_hidden:  int = 128,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.encoder = MultiScaleEncoder(pretrained=pretrained)
        self.head    = GRUHead(
            in_dim=self.encoder.out_dim,   # 3840
            hidden=gru_hidden,
            num_classes=num_classes,
        )

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = clip.shape
        feats = self.encoder(clip.view(b * t, c, h, w))   # (B*T, 3840)
        feats = feats.view(b, t, -1)                       # (B, T, 3840)
        return self.head(feats)                            # (B, num_classes)
