"""A Kinetics 3D ResNet that watches motion instead of colour (W01, W05).

    flow   (B, 14, 2, 112, 112)  RAFT's u and v, in pixels at 112 (W01)
    diff   (B, 15, 3, 224, 224)  the box clip as dataset.py hands it over;
                                 |frame t+1 - frame t| is taken here, then
                                 shrunk to 112 (W05)

Only the first convolution changes: it takes 2 or 3 motion channels
instead of RGB, and each new input channel starts from the mean of the
pretrained RGB filters, the usual way to carry ImageNet or Kinetics
weights over to flow. The rest is r3d_18 as published, with a new last
layer, trained as Day 23 trains everything.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model

from src.models.video_classifier import _swap_linear
from src.models.video_encoder import input_spec

__all__ = ["MotionClassifier", "KINDS", "FLOW_SCALE", "adapt_stem"]

KINDS = ("flow", "diff")
# flow in pixels -> roughly unit scale. Measured on 20 clips at 112: median
# move 2.4 px, 90% under 6.4, 99% under 14.5, the largest 42.5
FLOW_SCALE = 10.0


def adapt_stem(model, channels):
    """The first Conv3d, rebuilt for `channels` inputs, each starting from
    the mean of the old filters over their input channels."""
    old = model.stem[0]
    new = nn.Conv3d(channels, old.out_channels, old.kernel_size,
                    stride=old.stride, padding=old.padding,
                    bias=old.bias is not None)
    with torch.no_grad():
        mean = old.weight.mean(dim=1, keepdim=True)
        new.weight.copy_(mean.repeat(1, channels, 1, 1, 1))
        if old.bias is not None:
            new.bias.copy_(old.bias)
    model.stem[0] = new


class MotionClassifier(nn.Module):
    """r3d_18 (or another torchvision video ResNet) on flow or on frame
    differences. (B, T, C, H, W) -> (B, classes)."""

    def __init__(self, name, num_classes, kind, pretrained=True):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        self.name = name
        self.kind = kind
        if pretrained:
            weights = "DEFAULT"
        else:
            weights = None
        self.backbone = get_model(name, weights=weights)
        if kind == "flow":
            adapt_stem(self.backbone, 2)
        else:
            adapt_stem(self.backbone, 3)
        _swap_linear(self.backbone, "fc", num_classes)
        self.size = input_spec(name)[0]

    def prepare(self, x):
        """Our input as the network takes it: (B, T, C, size, size)."""
        if x.dim() != 5:
            raise ValueError(f"expected (B, T, C, H, W), got {tuple(x.shape)}")
        if self.kind == "flow":
            return x / FLOW_SCALE
        moved = (x[:, 1:] - x[:, :-1]).abs()
        b, t = moved.shape[:2]
        if moved.shape[-1] != self.size:
            moved = F.interpolate(moved.flatten(0, 1),
                                  size=(self.size, self.size),
                                  mode="bilinear", antialias=True,
                                  align_corners=False).unflatten(0, (b, t))
        return moved

    def forward(self, x):
        return self.backbone(self.prepare(x).transpose(1, 2))
