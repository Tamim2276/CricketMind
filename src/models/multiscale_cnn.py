"""
Multi-Scale CNN encoder using Feature Pyramid Network (FPN) approach.

Taps three internal stages of ONE EfficientNetV2-S forward pass:
  features.3 ->  64-d  (fine:   local texture, wrist angle, bat face)
  features.5 -> 160-d  (mid:    arms, bat shape, upper body)
  features.7 -> 1280-d (global: full pose, stance, foot position)

FIX vs original version:
  OLD: raw concat (1504-d) fed directly to GRU-128
       Problem: 11.7x compression bottleneck, fine/mid features drowned
       by 1280-d global (85% of signal)

  NEW: concat -> Projection(1504 -> 1280) -> LayerNorm -> GELU
       The projection layer LEARNS how to combine all three scales.
       Output is 1280-d (same as baseline) so GRU/Transformer head
       is directly comparable across models.
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class MultiScaleEncoder(nn.Module):
    """
    Input:  (B, C, 224, 224)
    Output: (B, 1280)  — projected multi-scale features

    out_dim = 1280  (same as FrameEncoder baseline for fair comparison)
    """

    _FINE_DIM   =   64   
    _MID_DIM    =  160   
    _GLOBAL_DIM = 1280   
    _CONCAT_DIM = _FINE_DIM + _MID_DIM + _GLOBAL_DIM  # 1504

    def __init__(self, pretrained: bool = True, proj_dim: int = 1280):
        super().__init__()

        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_v2_s(weights=weights)

        self.extractor = create_feature_extractor(
            base_model,
            return_nodes={
                "features.3": "fine",
                "features.5": "mid",
                "features.7": "global",
            },
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Projection: learned combination of all three scales
        # This is the key fix — raw concatenation loses scale balance.
        # A learned projection lets the model decide how much each
        # scale contributes to the final representation.
        self.proj = nn.Sequential(
            nn.Linear(self._CONCAT_DIM, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        self.out_dim = proj_dim  # 1280 — matches FrameEncoder for fair ablation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 224, 224)
        feats = self.extractor(x)

        fine   = torch.flatten(self.pool(feats["fine"]),   1)  # (B, 64)
        mid    = torch.flatten(self.pool(feats["mid"]),    1)  # (B, 160)
        glob   = torch.flatten(self.pool(feats["global"]), 1)  # (B, 1280)

        concat = torch.cat([fine, mid, glob], dim=1)           # (B, 1504)
        return self.proj(concat)                               # (B, 1280)