"""CLIP's image half, applied to every frame. Part 4 of step 8.10.

CLIP learned from 400 million pictures with their captions, so its features
are organised around what a sentence would say about an image -- a third kind
of pretraining, next to ImageNet's 1000 labels and Kinetics' 400 actions.

    (B, T, 3, 224, 224) -> (B, T, 768)

**768, not 512.** The 512 is CLIP's shared image-text space, one Linear (no
bias) after this vector. The CLIP paper's own linear probes read the 768
before it, and a probe on 768 can learn anything a probe on 512 can.

**Colours are CLIP's own**, read from its config: mean 0.481/0.458/0.408,
not ImageNet's 0.485/0.456/0.406. Close enough to look the same in a
table; not the numbers it was trained on.
"""
import torch
import torch.nn as nn

from src.models.hf import hf_file, hf_input_spec, quiet_loading

__all__ = ["ClipFrameEncoder", "is_clip", "reference", "HF_FRAME", "SPORTS"]

HF_FRAME = {"clip_vit_b16": "openai/clip-vit-base-patch16"}

# for verify only: what CLIP thinks a frame shows, zero-shot. Not a result.
SPORTS = ("cricket", "baseball", "tennis", "golf", "football", "hockey",
          "badminton", "volleyball")


def is_clip(name):
    return name in HF_FRAME


class ClipFrameEncoder(nn.Module):
    """Every frame through CLIP's image tower. (B, T, 3, H, W) -> (B, T, 768).

    The vector is `pooler_output`: the first token after the final LayerNorm,
    which is exactly what CLIP's projection reads.
    """

    def __init__(self, name="clip_vit_b16", pretrained=True, freeze=True):
        super().__init__()
        from transformers import CLIPVisionConfig, CLIPVisionModel

        if name not in HF_FRAME:
            raise ValueError(f"{name} is not one of {tuple(HF_FRAME)}")
        self.name = name
        repo = HF_FRAME[name]
        if pretrained:
            hf_file(repo, "pytorch_model.bin")
            with quiet_loading():
                self.backbone = CLIPVisionModel.from_pretrained(repo)
            self.size, self.mean, self.std = hf_input_spec(repo)
        else:
            # same shape, random weights, no download: for tests
            self.backbone = CLIPVisionModel(CLIPVisionConfig(patch_size=16))
            self.size = 224
            self.mean = (0.48145466, 0.4578275, 0.40821073)
            self.std = (0.26862954, 0.26130258, 0.27577711)
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
        out = self.backbone(pixel_values=x.flatten(0, 1)).pooler_output
        return out.unflatten(0, (b, t))                 # (B*T, D) -> (B, T, D)


def reference(name, device):
    """For verify: the whole CLIP, its projection, and a zero-shot scorer.

    `run` gives the full model's image embedding for (T, 3, H, W) frames;
    `head` is the projection applied to cached vectors. They must agree.
    `to_scores` turns an embedding into similarity with "a photo of a person
    playing <sport>", so verify can print what CLIP thinks it is looking at.
    """
    from transformers import AutoTokenizer, CLIPModel

    repo = HF_FRAME[name]
    with quiet_loading():
        full = CLIPModel.from_pretrained(repo).to(device).eval()
        tok = AutoTokenizer.from_pretrained(repo)
    prompts = []
    for sport in SPORTS:
        prompts.append(f"a photo of a person playing {sport}")
    with torch.no_grad():
        text = tok(prompts, padding=True, return_tensors="pt").to(device)
        words = full.get_text_features(**text).pooler_output.float()
        words = words / words.norm(dim=-1, keepdim=True)

    def run(x):
        return full.get_image_features(pixel_values=x).pooler_output

    def head(saved):
        return full.visual_projection(saved)

    def to_scores(emb):
        emb = emb.float()
        return (emb / emb.norm(dim=-1, keepdim=True)) @ words.T

    return run, head, list(SPORTS), to_scores
