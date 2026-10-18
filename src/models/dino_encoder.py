"""DINOv2's image tower, applied to every frame. V10 of Family V (Day 14).

DINOv2 learned from 142 million images with no labels and no captions at
all: a fourth kind of pretraining, next to ImageNet's labels, Kinetics'
actions and CLIP's captions.

    (B, T, 3, 224, 224) -> (B, T, 1536)

**1,536, not 768.** DINOv2's own linear evaluation, and the ImageNet head
shipped in this checkpoint, read the first token joined to the mean of the
patch tokens, both after the final LayerNorm. That is the vector the 8.10
rule asks for, and the head lets verify check it. The backbone is the
frozen DINOv2-Base that head was trained on.

Trained at 518px, used at 224: 224 is what its own preprocessor crops to,
and the model interpolates its position embeddings for the 16 x 16 patches.
"""
import torch
import torch.nn as nn

from src.models.hf import WEIGHTS, hf_file, hf_input_spec, quiet_loading

__all__ = ["DinoFrameEncoder", "is_dino", "reference", "HF_DINO"]

HF_DINO = {"dinov2_base": "facebook/dinov2-base-imagenet1k-1-layer"}


def is_dino(name):
    return name in HF_DINO


class DinoFrameEncoder(nn.Module):
    """Every frame through DINOv2. (B, T, 3, H, W) -> (B, T, 1536)."""

    def __init__(self, name="dinov2_base", pretrained=True, freeze=True):
        super().__init__()
        from transformers import Dinov2Config, Dinov2ForImageClassification

        if name not in HF_DINO:
            raise ValueError(f"{name} is not one of {tuple(HF_DINO)}")
        self.name = name
        repo = HF_DINO[name]
        if pretrained:
            hf_file(repo, WEIGHTS[repo])
            with quiet_loading():
                model = Dinov2ForImageClassification.from_pretrained(repo)
            self.size, self.mean, self.std = hf_input_spec(repo)
        else:
            # same shape, random weights, no download: for tests
            model = Dinov2ForImageClassification(Dinov2Config(num_labels=1000))
            self.size = 224
            self.mean = (0.485, 0.456, 0.406)
            self.std = (0.229, 0.224, 0.225)
        # its classifier reads cat(first token, mean of patches); with the
        # classifier gone, .logits is that vector
        model.classifier = nn.Identity()
        self.backbone = model
        self.out_dim = self._measure()
        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    def _measure(self):
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            out = self(torch.zeros(1, 1, 3, self.size, self.size))
        self.backbone.train(was_training)
        return int(out.shape[-1])

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        if x.dim() != 5:
            raise ValueError(f"expected (B, T, C, H, W), got {tuple(x.shape)}")
        b, t = x.shape[:2]
        out = self.backbone(pixel_values=x.flatten(0, 1)).logits
        return out.unflatten(0, (b, t))                 # (B*T, D) -> (B, T, D)


def reference(name, device):
    """For verify: the whole model, its ImageNet head, the class names.

    `run` gives the full model's logits for (T, 3, H, W) frames; `head` is
    the Linear applied to the cached vectors. They must agree.
    """
    from transformers import Dinov2ForImageClassification

    with quiet_loading():
        full = Dinov2ForImageClassification.from_pretrained(HF_DINO[name])
    full = full.to(device).eval()
    names = []
    for i in range(full.config.num_labels):
        names.append(full.config.id2label[i])

    def run(x):
        return full(pixel_values=x).logits

    return run, full.classifier, names, None
