"""
Baseline model: EfficientNetV2-S (FrameEncoder) + GRU head.
Combines backbone.py and temporal_gru.py into a single nn.Module
that takes a full clip and returns class logits.
"""
import torch
import torch.nn as nn

from src.models.backbone import FrameEncoder
from src.models.temporal_gru import GRUHead


class BaselineModel(nn.Module):
    """
    Input:  clip  (B, T, C, H, W)  — T=15 frames, each 224×224, ImageNet-normalized
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
        self.encoder = FrameEncoder(pretrained=pretrained)
        self.head    = GRUHead(
            in_dim=self.encoder.out_dim,
            hidden=gru_hidden,
            num_classes=num_classes,
        )

    def forward(self, clip):
        # clip: (B, T, C, H, W)
        b, t, c, h, w = clip.shape
        # Encode all frames in one batched forward pass
        feats = self.encoder(clip.view(b * t, c, h, w))  # (B*T, 1280)
        feats = feats.view(b, t, -1)                       # (B, T, 1280)
        return self.head(feats)                            # (B, num_classes)
