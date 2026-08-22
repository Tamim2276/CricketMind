"""
Multi-Scale CNN encoder — three crops of the same frame (full, mid, fine)
through a single shared-weight EfficientNetV2-S backbone.

Targets the Pull-vs-Hook and Lofted-Legside-vs-Offside confusion errors
identified in the original CricShotNet paper by capturing fine-grained
bat-angle and wrist detail that a single global crop misses.

Architecture (per frame):
    full  frame (224*224) → shared EfficientNetV2-S → 1280-d
    mid   crop (112->224)  → shared EfficientNetV2-S → 1280-d
    fine  crop  (56->224)  → shared EfficientNetV2-S → 1280-d
    concat → 3840-d per frame

Shared weights keep parameter count and VRAM identical to a single backbone.
"""
import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models.feature_extraction import create_feature_extractor

class MultiScaleEncoder(nn.Module):
    """
    True Multi-Scale Feature Pyramid Encoder.
    Extracts features at three different depths inside EfficientNetV2-S:
      - fine   (features.3): 64 channels, 28x28 resolution (local texture/wrist angle)
      - mid    (features.5): 160 channels, 14x14 resolution (body/arms/bat shape)
      - global (features.7): 1280 channels, 7x7 resolution (full pose context)
    
    This avoids redundant backbone passes and doesn't rely on arbitrary center-crops.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        base_model = efficientnet_v2_s(weights=weights)
        
        # We hook into 3 intermediate stages of the network
        return_nodes = {
            'features.3': 'fine',
            'features.5': 'mid',
            'features.7': 'global'
        }
        self.extractor = create_feature_extractor(base_model, return_nodes=return_nodes)
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 64 + 160 + 1280  # 1504

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 224, 224)
        features = self.extractor(x)
        
        fine   = torch.flatten(self.pool(features['fine']), 1)
        mid    = torch.flatten(self.pool(features['mid']), 1)
        glob   = torch.flatten(self.pool(features['global']), 1)
        
        return torch.cat([fine, mid, glob], dim=1)  # (B, 1504)
