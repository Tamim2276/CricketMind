"""
EfficientNetV2-S backbone — classifier head removed, returns 1280-d features.
Matches the backbone used in the original CricShotNet paper.
"""
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class FrameEncoder(nn.Module):
    """EfficientNetV2-S feature extractor, classifier head removed.

    Input:  (B, C, H, W) — single frame, ImageNet-normalized, H=W=224
    Output: (B, 1280)    — pooled feature vector
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_v2_s(weights=weights)
        self.features = backbone.features
        self.pool = backbone.avgpool
        self.out_dim = 1280  # EfficientNetV2-S output channels

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return x  # (B, 1280)
