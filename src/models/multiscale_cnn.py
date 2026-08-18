"""
Multi-Scale CNN encoder — three crops of the same frame (full, mid, fine)
through a single shared-weight EfficientNetV2-S backbone.

Targets the Pull-vs-Hook and Lofted-Legside-vs-Offside confusion errors
identified in the original CricShotNet paper by capturing fine-grained
bat-angle and wrist detail that a single global crop misses.

Architecture (per frame):
    full  frame (224×224) → shared EfficientNetV2-S → 1280-d
    mid   crop (112→224)  → shared EfficientNetV2-S → 1280-d
    fine  crop  (56→224)  → shared EfficientNetV2-S → 1280-d
    concat → 3840-d per frame

Shared weights keep parameter count and VRAM identical to a single backbone.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import FrameEncoder


class MultiScaleEncoder(nn.Module):
    """
    Input:  (B, C, H, W)  — single frame, 224×224, ImageNet-normalized
    Output: (B, out_dim)  — concatenated multi-scale features (out_dim = 1280 * 3)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = FrameEncoder(pretrained=pretrained)  # shared across all 3 scales
        self.out_dim = self.encoder.out_dim * 3             # 3840

    def _center_crop_and_resize(self, x: torch.Tensor, crop_size: int) -> torch.Tensor:
        """Center-crop to crop_size × crop_size, then resize back to 224 for the backbone."""
        _, _, h, w = x.shape
        top  = (h - crop_size) // 2
        left = (w - crop_size) // 2
        # clamp so we never go out of bounds if crop_size > h or w
        top  = max(0, top)
        left = max(0, left)
        cs   = min(crop_size, h - top, w - left)
        patch = x[:, :, top:top + cs, left:left + cs]
        return F.interpolate(patch, size=224, mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 224, 224) — full frame already preprocessed by CricShotNet pipeline
        full = self.encoder(x)                                   # global context
        mid  = self.encoder(self._center_crop_and_resize(x, 112))  # upper-body + hands
        fine = self.encoder(self._center_crop_and_resize(x,  56))  # bat-face detail
        return torch.cat([full, mid, fine], dim=1)               # (B, 3840)
