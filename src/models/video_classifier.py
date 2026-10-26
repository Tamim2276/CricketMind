"""A whole video model, trained end to end on the shots (Family C, Day 23).

Family V froze these models and kept one vector per clip. Here the whole
network trains, and only its last layer is new: 400 Kinetics classes
become our 15.

    (B, T, 3, 224, 224)     ImageNet colours, as dataset.py hands them over
    (B, frames, 3, S, S)    the frame count, size and colours of the weights
    (B, 15)                 logits

The clip is converted on the GPU exactly as Family V's extraction did it on
the CPU, so C02 against V01 differs only in what trains: evenly spaced
frames when the weights insist on a count, an antialiased resize to the
size they were trained at, and their colours. Resizing and recolouring are
both linear, so doing them in the other order gives the same numbers.

    r3d_18, r2plus1d_18     torchvision 3D CNNs, 112px, any length
    mvit_v2_s               torchvision MViT, 224px, exactly 16
    swin3d_t                torchvision video Swin, 224px, any length
    i3d_r50, x3d_m          pytorchvideo, 224px, 8 and 16 frames
    slowfast_r50            pytorchvideo, 224px, two copies of the clip:
                            32 frames (fast) and 8 (slow)
    videomae_base           transformers ViT, 224px, exactly 16

VideoMAE runs eager attention, as DINOv2 does (14.2): SDPA in bf16 on the
Arc is not repeatable, and gave NaNs in full runs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.models.hf import WEIGHTS, hf_file, quiet_loading
from src.models.video_encoder import (HF_KIND, HF_VIDEO, input_spec,
                                      is_video, needed_frames,
                                      resample_frames)

__all__ = ["VideoClassifier", "PYTORCHVIDEO", "known_video"]

ATTENTION = "eager"

# pytorchvideo's hub models: (frames, size) they were trained on. Their
# heads pool with a fixed kernel sized for exactly this input
PYTORCHVIDEO = {
    "i3d_r50": (8, 224),
    "slowfast_r50": (32, 224),    # the fast path; the slow one takes 8
    "x3d_m": (16, 224),
}
PYTORCHVIDEO_MEAN = (0.45, 0.45, 0.45)
PYTORCHVIDEO_STD = (0.225, 0.225, 0.225)
SLOW = 8


def known_video(name):
    return name in HF_VIDEO or name in PYTORCHVIDEO or is_video(name)


def _swap_linear(parent, attr, num_classes):
    """parent.attr, a Linear or a Sequential ending in one, gets a fresh
    Linear to num_classes. Whatever came before it (MViT's dropout) stays."""
    old = getattr(parent, attr)
    if isinstance(old, nn.Linear):
        setattr(parent, attr, nn.Linear(old.in_features, num_classes))
        return
    if isinstance(old, nn.Sequential) and isinstance(old[-1], nn.Linear):
        old[-1] = nn.Linear(old[-1].in_features, num_classes)
        return
    raise ValueError(f"{type(parent).__name__}.{attr} does not end in a "
                     f"Linear")


def _torchvision(name, num_classes, pretrained):
    if pretrained:
        weights = "DEFAULT"
    else:
        weights = None
    model = get_model(name, weights=weights)
    if hasattr(model, "fc"):
        _swap_linear(model, "fc", num_classes)        # the ResNet3Ds
    elif hasattr(model, "head"):
        _swap_linear(model, "head", num_classes)      # MViT, Swin3D
    else:
        raise ValueError(f"no classifier found on {name}")
    size, mean, std = input_spec(name)
    return model, needed_frames(name), size, mean, std


def _pytorchvideo(name, num_classes, pretrained):
    from pytorchvideo.models import hub

    model = getattr(hub, name)(pretrained=pretrained)
    head = model.blocks[-1]
    _swap_linear(head, "proj", num_classes)
    if getattr(head, "activation", None) is not None:
        # a softmax here would sit between the logits and the loss
        head.activation = None
    frames, size = PYTORCHVIDEO[name]
    return model, frames, size, PYTORCHVIDEO_MEAN, PYTORCHVIDEO_STD


def _transformers(name, num_classes, pretrained):
    from transformers import AutoConfig, AutoModelForVideoClassification

    repo = HF_VIDEO[name]
    if pretrained:
        hf_file(repo, WEIGHTS[repo])
        with quiet_loading():
            model = AutoModelForVideoClassification.from_pretrained(
                repo, attn_implementation=ATTENTION)
        size, mean, std = input_spec(name)
    else:
        # same shape, random weights, no download: for tests
        config = AutoConfig.for_model(HF_KIND[name], num_labels=400)
        model = AutoModelForVideoClassification.from_config(
            config, attn_implementation=ATTENTION)
        size = model.config.image_size
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    _swap_linear(model, "classifier", num_classes)
    return model, model.config.num_frames, size, mean, std


class VideoClassifier(nn.Module):
    """A video model with a new last layer. (B, T, 3, H, W) -> (B, classes)."""

    def __init__(self, name, num_classes, pretrained=True):
        super().__init__()
        self.name = name
        self.library = "torchvision"
        if name in HF_VIDEO:
            self.library = "transformers"
            built = _transformers(name, num_classes, pretrained)
        elif name in PYTORCHVIDEO:
            self.library = "pytorchvideo"
            built = _pytorchvideo(name, num_classes, pretrained)
        elif is_video(name):
            built = _torchvision(name, num_classes, pretrained)
        else:
            raise ValueError(f"{name} is not a video model this knows")
        self.backbone, self.frames, self.size, mean, std = built
        self.mean = tuple(mean)
        self.std = tuple(std)

        # ImageNet colours in, the weights' colours out, as one multiply-add.
        # Not saved: it follows from the name
        scale = []
        shift = []
        for m, s, im, ist in zip(mean, std, IMAGENET_MEAN, IMAGENET_STD):
            scale.append(ist / s)
            shift.append((im - m) / s)
        self.register_buffer("scale", torch.tensor(scale).view(1, 1, 3, 1, 1),
                             persistent=False)
        self.register_buffer("shift", torch.tensor(shift).view(1, 1, 3, 1, 1),
                             persistent=False)

    def prepare(self, x):
        """Our clip as the weights expect it: (B, frames, 3, size, size)."""
        if x.dim() != 5:
            raise ValueError(f"expected (B, T, C, H, W), got {tuple(x.shape)}")
        if self.frames is not None:
            x = resample_frames(x, self.frames)
        x = x * self.scale + self.shift
        if x.shape[-1] != self.size:
            b, t = x.shape[:2]
            # antialias, or halving the size aliases the bat's thin edge
            x = F.interpolate(x.flatten(0, 1), size=(self.size, self.size),
                              mode="bilinear", antialias=True,
                              align_corners=False)
            x = x.unflatten(0, (b, t))
        return x

    def forward(self, x):
        x = self.prepare(x)
        if self.library == "transformers":
            return self.backbone(pixel_values=x).logits
        x = x.transpose(1, 2)                       # -> (B, C, T, H, W)
        if self.name == "slowfast_r50":
            slow = resample_frames(x.transpose(1, 2), SLOW).transpose(1, 2)
            return self.backbone([slow, x])
        return self.backbone(x)
