"""A whole clip through a video model. Step 8.10's motion question, and the
frozen half of Family V (Day 14).

FrameEncoder looks at 15 photos, one at a time. These watch the clip:

    (B, T, 3, H, W)         a batch of clips, laid out as dataset.py does
    (B, D)                  one vector for the whole clip

Time is pooled away inside the network, so a cache of these has no frames
left for a GRU to read. It feeds a plain classifier instead.

    r3d_18, mc3_18,     torchvision 3D CNNs, Kinetics-400, 112px, any length
      r2plus1d_18
    s3d                 torchvision, Kinetics-400, 224px, 14 frames or more
    swin3d_t            torchvision video Swin, Kinetics-400, 224px, any length
    mvit_v2_s           torchvision MViT, Kinetics-400, 224px, exactly 16
    videomae_base       transformers ViT, Kinetics-400, 224px, exactly 16
    timesformer_base    transformers, Kinetics-400, 224px, exactly 8

**Input size and colours come from the weights, not from dataset.py.**
r3d_18 was trained at 112x112 with Kinetics colours; feeding it 224
ImageNet frames runs and quietly costs a third of its confidence
(measured). Both libraries ship those numbers with the weights.

**A model that insists on a frame count gets evenly spaced picks.**
`resample_frames` turns 15 into 16 by repeating frame 7, and into 8 by
keeping every other frame. Repeating the last frame instead would freeze
the follow-through.

**S3D has no Linear to remove.** Its classifier is a 1x1x1 Conv3d, and its
forward averages after that conv. A linear map commutes with an average, so
with the classifier gone the forward returns exactly the vector the conv
reads, and `conv_as_linear` turns the conv into that Linear for verify.
"""
import torch
import torch.nn as nn
import torchvision
from torchvision.models import get_model, get_model_weights, list_models

from src.models.encoder import _strip_classifier
from src.models.hf import WEIGHTS, hf_file, hf_input_spec, quiet_loading

__all__ = ["VideoEncoder", "HFVideoEncoder", "video_encoder", "reference",
           "is_video", "input_spec", "needed_frames", "resample_frames",
           "conv_as_linear", "HF_VIDEO", "HF_KIND", "DEFAULT_VIDEO"]

DEFAULT_VIDEO = "r3d_18"

# the Kinetics-finetuned checkpoints: a probe needs features that already
# learned labels, and their classifiers let verify check the cached vector
HF_VIDEO = {
    "videomae_base": "MCG-NJU/videomae-base-finetuned-kinetics",
    "timesformer_base": "facebook/timesformer-base-finetuned-k400",
}
# each one's config type, so a test can build the same shape offline
HF_KIND = {"videomae_base": "videomae", "timesformer_base": "timesformer"}


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


def needed_frames(name, have=15):
    """The frame count a torchvision video model insists on, or None when
    `have` frames already work. Read from the weights: MViT says 16, S3D
    says 14, so S3D takes the 15 as they are."""
    least = get_model_weights(name).DEFAULT.meta.get("min_temporal_size", 1)
    if least > have:
        return least
    return None


def resample_frames(x, n):
    """(B, T, ...) -> (B, n, ...), n evenly spaced picks over the T frames."""
    t = x.shape[1]
    if t == n:
        return x
    idx = torch.linspace(0, t - 1, n).round().long()
    return x[:, idx.to(x.device)]


def conv_as_linear(conv):
    """S3D's 1x1x1 Conv3d as the Linear it really is."""
    linear = nn.Linear(conv.in_channels, conv.out_channels)
    with torch.no_grad():
        linear.weight.copy_(conv.weight.flatten(1))     # (out, in, 1, 1, 1)
        linear.bias.copy_(conv.bias)
    return linear.to(conv.weight.device)


def _strip_video_head(name, backbone):
    if name == "s3d":
        # forward() averages after the classifier, so without it forward()
        # returns the average that the 1x1x1 conv would have read
        backbone.classifier = nn.Identity()
        return
    _strip_classifier(backbone)


def _freeze(module):
    for p in module.parameters():
        p.requires_grad_(False)


class VideoEncoder(nn.Module):
    """torchvision video model. (B, T, 3, size, size) -> (B, out_dim)."""

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
        _strip_video_head(name, self.backbone)
        self.size, self.mean, self.std = input_spec(name)
        self.frames = needed_frames(name, frames)
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
        if self.frames is not None:
            x = resample_frames(x, self.frames)
        return self.backbone(x.transpose(1, 2))     # -> (B, C, T, H, W)


class HFVideoEncoder(VideoEncoder):
    """transformers VideoMAE or TimeSformer. (B, T, 3, 224, 224) -> (B, 768).

    Only the classifier is removed, so the vector is exactly what it reads:
    for VideoMAE the mean of the tokens after fc_norm, for TimeSformer the
    first token after the final LayerNorm. Both already take
    (B, T, C, H, W), so no transpose here.
    """

    def __init__(self, name="videomae_base", pretrained=True, freeze=True,
                 frames=15):
        nn.Module.__init__(self)
        from transformers import AutoConfig, AutoModelForVideoClassification

        if name not in HF_VIDEO:
            raise ValueError(f"{name} is not one of {tuple(HF_VIDEO)}")
        self.name = name
        repo = HF_VIDEO[name]
        if pretrained:
            hf_file(repo, WEIGHTS[repo])
            with quiet_loading():
                model = AutoModelForVideoClassification.from_pretrained(repo)
        else:
            # same shape, random weights, no download: for tests
            config = AutoConfig.for_model(HF_KIND[name], num_labels=400)
            model = AutoModelForVideoClassification.from_config(config)
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
        return HFVideoEncoder(name, pretrained=pretrained, freeze=freeze)
    return VideoEncoder(name, pretrained=pretrained, freeze=freeze)


def reference(name, device):
    """The untouched pretrained model, its last Linear, and its class names.

    For verify: the cached vector through that Linear must give what the
    whole model gives on the same clip. `run` takes (1, T, 3, H, W).
    """
    if name in HF_VIDEO:
        from transformers import AutoModelForVideoClassification

        with quiet_loading():
            full = AutoModelForVideoClassification.from_pretrained(
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
    if name == "s3d":
        head = conv_as_linear(full.classifier[1])
    else:
        head = None
        for layer in full.modules():
            if isinstance(layer, nn.Linear):
                head = layer
    frames = needed_frames(name)

    def run(x):
        if frames is not None:
            x = resample_frames(x, frames)
        return full(x.transpose(1, 2))

    return run, head, list(weights.meta["categories"])
