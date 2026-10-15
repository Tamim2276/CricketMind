"""A whole clip through a 3D CNN. The motion question of step 8.10.

FrameEncoder looks at 15 photos, one at a time. This one watches the clip:

    (B, T, 3, 112, 112)     a batch of clips, laid out as dataset.py does
    (B, 3, T, 112, 112)     the layout 3D CNNs read: channels before time
    (B, 512)                one vector for the whole clip

Time is pooled away inside the network, so a cache of these has no frames
left for a GRU to read. It feeds a plain classifier instead.

**Input size and colours come from the weights, not from dataset.py.**
r3d_18 was trained on Kinetics-400 at 112x112 with its own mean and std;
feeding it 224x224 ImageNet-normalised frames runs without error and is
wrong. torchvision ships those numbers with the weights, so they are read
from there rather than typed in.

15 frames is fine. Kinetics clips are 16, but r3d_18 pools time adaptively,
so any length goes through (measured: 15 and 16 both give (1, 512)).
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models import get_model, get_model_weights, list_models

from src.models.encoder import _strip_classifier

__all__ = ["VideoEncoder", "is_video", "input_spec", "DEFAULT_VIDEO"]

DEFAULT_VIDEO = "r3d_18"


def is_video(name):
    return name in list_models(module=torchvision.models.video)


def input_spec(name):
    """(size, mean, std) that the pretrained weights were trained with."""
    t = get_model_weights(name).DEFAULT.transforms()
    return t.crop_size[0], tuple(t.mean), tuple(t.std)


class VideoEncoder(nn.Module):
    """(B, T, 3, size, size) -> (B, out_dim). One vector per clip."""

    def __init__(self, name=DEFAULT_VIDEO, pretrained=True, freeze=True,
                 frames=15):
        super().__init__()
        if not is_video(name):
            raise ValueError(f"{name} is not a torchvision video model")
        self.name = name
        if pretrained:
            weights = "DEFAULT"
        else:
            weights = None
        self.backbone = get_model(name, weights=weights)
        _strip_classifier(self.backbone)
        self.size, self.mean, self.std = input_spec(name)
        self.out_dim = self._measure(frames)
        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def _measure(self, frames):
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            out = self.backbone(torch.zeros(1, 3, frames, self.size, self.size))
        self.backbone.train(was_training)
        if out.dim() != 2:
            raise ValueError(f"{self.name} returned {tuple(out.shape)}, "
                             f"expected (N, D)")
        return int(out.shape[1])

    def train(self, mode=True):
        # frozen means eval, or BatchNorm keeps updating its running stats
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"expected (B, T, C, H, W), got {tuple(x.shape)}")
        return self.backbone(x.transpose(1, 2))     # -> (B, C, T, H, W)
