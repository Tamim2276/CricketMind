"""A whole clip through a video model. The motion question of step 8.10.

FrameEncoder looks at 15 photos, one at a time. These watch the clip:

    (B, T, 3, H, W)         a batch of clips, laid out as dataset.py does
    (B, D)                  one vector for the whole clip

Time is pooled away inside the network, so a cache of these has no frames
left for a GRU to read. It feeds a plain classifier instead.

Two families:

    r3d_18          torchvision 3D CNN, Kinetics-400, 112px, any length
    videomae_base   transformers ViT, Kinetics-400, 224px, exactly 16 frames

**Input size and colours come from the weights, not from dataset.py.**
r3d_18 was trained at 112x112 with Kinetics colours; feeding it 224
ImageNet frames runs and quietly costs a third of its confidence
(measured). Both libraries ship those numbers with the weights.

**VideoMAE needs exactly 16 frames; the clips have 15.** `resample_frames`
takes 16 evenly spaced picks, which repeats frame 7 and keeps the timing
even. Repeating the last frame instead would freeze the follow-through.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models import get_model, get_model_weights, list_models

from src.models.encoder import _strip_classifier
from src.models.hf import hf_file, hf_input_spec, quiet_loading

__all__ = ["VideoEncoder", "VideoMAEEncoder", "video_encoder", "reference",
           "is_video", "input_spec", "resample_frames", "HF_VIDEO",
           "DEFAULT_VIDEO"]

DEFAULT_VIDEO = "r3d_18"

# the Kinetics-finetuned checkpoint: a probe needs features that already
# learned labels, and its classifier lets verify check the cached vector
HF_VIDEO = {"videomae_base": "MCG-NJU/videomae-base-finetuned-kinetics"}


def is_video(name):
    if name in HF_VIDEO:
        return True
    return name in list_models(module=torchvision.models.video)


def input_spec(name):
    """(size, mean, std) that the pretrained weights were trained with."""
    if name in HF_VIDEO:
        return hf_input_spec(HF_VIDEO[name])
    t = get_model_weights(name).DEFAULT.transforms()
    return t.crop_size[0], tuple(t.mean), tuple(t.std)


def resample_frames(x, n):
    """(B, T, ...) -> (B, n, ...), n evenly spaced picks over the T frames."""
    t = x.shape[1]
    if t == n:
        return x
    idx = torch.linspace(0, t - 1, n).round().long()
    return x[:, idx.to(x.device)]


def _freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)


class VideoEncoder(nn.Module):
    """torchvision 3D CNN. (B, T, 3, size, size) -> (B, out_dim)."""

    def __init__(self, name=DEFAULT_VIDEO, pretrained=True, freeze=True,
                 frames=15):
        super().__init__()
        if name in HF_VIDEO or not is_video(name):
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
            _freeze(self.backbone)

    def _measure(self, frames):
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            out = self(torch.zeros(1, frames, 3, self.size, self.size))
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


class VideoMAEEncoder(VideoEncoder):
    """transformers VideoMAE. (B, T, 3, 224, 224) -> (B, 768).

    The vector is the mean over all tokens after the final LayerNorm --
    exactly what its classifier reads. Only the classifier is removed.
    VideoMAE already takes (B, T, C, H, W), so no transpose here.
    """

    def __init__(self, name="videomae_base", pretrained=True, freeze=True,
                 frames=15):
        nn.Module.__init__(self)
        from transformers import VideoMAEConfig, VideoMAEForVideoClassification

        if name not in HF_VIDEO:
            raise ValueError(f"{name} is not one of {tuple(HF_VIDEO)}")
        self.name = name
        repo = HF_VIDEO[name]
        if pretrained:
            hf_file(repo, "model.safetensors")
            with quiet_loading():
                model = VideoMAEForVideoClassification.from_pretrained(repo)
        else:
            # same shape, random weights, no download: for tests
            model = VideoMAEForVideoClassification(VideoMAEConfig(num_labels=400))
        model.classifier = nn.Identity()
        self.backbone = model
        self.frames = model.config.num_frames
        if pretrained:
            self.size, self.mean, self.std = input_spec(name)
        else:
            self.size = model.config.image_size
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        self.out_dim = self._measure(frames)
        self.frozen = freeze
        if freeze:
            _freeze(self.backbone)

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"expected (B, T, C, H, W), got {tuple(x.shape)}")
        x = resample_frames(x, self.frames)
        return self.backbone(pixel_values=x).logits


def video_encoder(name, pretrained=True, freeze=True):
    if name in HF_VIDEO:
        return VideoMAEEncoder(name, pretrained=pretrained, freeze=freeze)
    return VideoEncoder(name, pretrained=pretrained, freeze=freeze)


def reference(name, device):
    """The untouched pretrained model, its last Linear, and its class names.

    For verify: the cached vector through that Linear must give what the
    whole model gives on the same clip. `run` takes (1, T, 3, H, W).
    """
    if name in HF_VIDEO:
        from transformers import VideoMAEForVideoClassification

        with quiet_loading():
            full = VideoMAEForVideoClassification.from_pretrained(
                HF_VIDEO[name])
        full = full.to(device).eval()
        names = []
        for i in range(full.config.num_labels):
            names.append(full.config.id2label[i])
        frames = full.config.num_frames

        def run(x):
            return full(pixel_values=resample_frames(x, frames)).logits

        return run, full.classifier, names

    weights = get_model_weights(name).DEFAULT
    full = get_model(name, weights="DEFAULT").to(device).eval()
    linear = None
    for layer in full.modules():
        if isinstance(layer, nn.Linear):
            linear = layer

    def run(x):
        return full(x.transpose(1, 2))

    return run, linear, list(weights.meta["categories"])
