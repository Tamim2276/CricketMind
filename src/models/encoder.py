"""One image model, applied to all 15 frames.

The shape dance, which is the whole idea:

    (B, T, 3, 224, 224)     a batch of clips
    (B*T, 3, 224, 224)      fold time into the batch -- "just a pile of photos"
    (B*T, 1280)             the CNN describes each one
    (B, T, 1280)            unfold back into clips

Nothing here knows about video. The same weights see every frame, and the GPU
gets one large job instead of T small ones -- a python loop over T would be
correct and far slower.

The backbone is a parameter because Phase 3 swaps it ten times. Everything
downstream reads `out_dim` rather than assuming 1280.
"""
import torch
import torch.nn as nn
from torchvision.models import get_model

__all__ = ["FrameEncoder", "DEFAULT_BACKBONE"]

DEFAULT_BACKBONE = "efficientnet_v2_s"


def _find_linear(mod: nn.Module):
    """The Linear layer inside a torchvision head, or None."""
    if isinstance(mod, nn.Linear):
        return mod
    for layer in mod.modules():
        if isinstance(layer, nn.Linear):
            return layer
    return None


def _strip_classifier(model: nn.Module) -> int:
    """Drop the 1000-class ImageNet head. Returns the feature width."""
    for attr in ("classifier", "fc", "head"):
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        linear = _find_linear(mod)
        if linear is None:
            continue
        width = linear.in_features
        setattr(model, attr, nn.Identity())
        return width
    raise ValueError(f"no classifier found on {type(model).__name__}")


class FrameEncoder(nn.Module):
    """Every frame through the same CNN. (B, T, 3, H, W) -> (B, T, out_dim)."""

    def __init__(self, name: str = DEFAULT_BACKBONE, pretrained: bool = True,
                 freeze: bool = False):
        super().__init__()
        self.name = name
        if pretrained:
            weights = "DEFAULT"
        else:
            weights = None
        self.backbone = get_model(name, weights=weights)
        self.out_dim = _strip_classifier(self.backbone)
        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        """A frozen backbone stays in eval, or its BatchNorm keeps learning.

        Freezing the weights is not enough: BatchNorm updates running mean and
        variance in train mode whether or not it has gradients, so "frozen"
        features would drift anyway.
        """
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected (B, T, C, H, W), got {tuple(x.shape)}")
        b, t = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1))     # (B*T, C, H, W) -> (B*T, D)
        return feats.unflatten(0, (b, t))          # -> (B, T, D)
