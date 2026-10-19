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

`pool` decides what a frame becomes:

    "avg"      global average pool -> 1280 numbers
    "flatten"  the whole 7x7 grid  -> 62720 numbers

**The authors used "flatten".** Their shipped `.keras` file has a GRU kernel
of shape (62720, 384) -- 62720 is 7 x 7 x 1280, 384 is 3 gates x 128 -- so
their EfficientNet stops at `top_activation` and a Flatten feeds the GRU the
whole spatial map. Average pooling throws away *where* things are, and for a
cricket shot the bat's position relative to the body is much of the signal.
It also costs 44x the GRU parameters: 24.1 M against 0.54 M.

**torchvision first, then timm.** B02 (EfficientNetV2-B3) and B03 (Xception)
exist only in timm, so a name torchvision does not know is looked up there.
torchvision wins a name both have (resnet50), so every earlier run builds
exactly what it built before. A timm model may want other colours than the
ImageNet ones the dataset uses (Xception wants 0.5 / 0.5); the encoder
converts, the 8.10 rule: colours come from the weights.
"""
import torch
import torch.nn as nn
from torchvision.models import get_model, list_models

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD

__all__ = ["FrameEncoder", "DEFAULT_BACKBONE", "DEFAULT_SIZE", "POOLS",
           "is_timm", "backbone_problem"]

DEFAULT_BACKBONE = "efficientnet_v2_s"
DEFAULT_SIZE = 224
POOLS = ("avg", "flatten")


def is_timm(name: str) -> bool:
    """A backbone torchvision lacks but timm has, e.g. legacy_xception.tf_in1k."""
    if name in list_models():
        return False
    import timm
    return timm.is_model(name)


def _hf_weights(name: str):
    """(repo, file) when timm fetches this model's weights from Hugging Face,
    or None when it downloads them itself from a plain URL."""
    import timm
    cfg = timm.models.get_pretrained_cfg(name)
    if cfg is None or not cfg.hf_hub_id:
        return None
    return cfg.hf_hub_id, "model.safetensors"


def _hf_cached(repo: str, filename: str) -> bool:
    from huggingface_hub import try_to_load_from_cache
    return isinstance(try_to_load_from_cache(repo, filename), str)


def backbone_problem(name: str, pretrained: bool = True):
    """Why this backbone cannot be built yet, or None.

    HF_HUB_OFFLINE=1 is set on this machine, so a timm model whose weights
    live on Hugging Face must be downloaded once by hand; this names the
    command. torchvision and URL-hosted timm weights download themselves.
    """
    if name in list_models():
        return None
    if not is_timm(name):
        return f"neither torchvision nor timm has {name}"
    if not pretrained:
        return None
    hosted = _hf_weights(name)
    if hosted is None or _hf_cached(*hosted):
        return None
    return (f"weights not downloaded: HF_HUB_OFFLINE=0 hf download "
            f"{hosted[0]} --include \"*.json\" --include \"{hosted[1]}\"")


def _timm_backbone(name: str, pretrained: bool, pool: str):
    """A timm model without its classifier, and the colours it was trained on.

    num_classes=0 drops the classifier; global_pool="" also drops the pool,
    leaving the (N, C, 7, 7) grid, which a Flatten turns into one row.
    """
    import timm
    if pool == "flatten":
        model = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                  global_pool="")
        mean = tuple(model.pretrained_cfg["mean"])
        std = tuple(model.pretrained_cfg["std"])
        return nn.Sequential(model, nn.Flatten(1)), mean, std
    model = timm.create_model(name, pretrained=pretrained, num_classes=0)
    return model, tuple(model.pretrained_cfg["mean"]), \
        tuple(model.pretrained_cfg["std"])


def _same(a, b) -> bool:
    for x, y in zip(a, b):
        if abs(x - y) > 1e-6:
            return False
    return True


def _find_linear(mod: nn.Module):
    """The Linear layer inside a torchvision head, or None."""
    if isinstance(mod, nn.Linear):
        return mod
    for layer in mod.modules():
        if isinstance(layer, nn.Linear):
            return layer
    return None


def _strip_classifier(model: nn.Module) -> None:
    """Drop the 1000-class Linear, keep the real layers in front of it.

    ConvNeXt's head is LayerNorm -> Flatten -> Linear, and that LayerNorm is
    the network's last layer, not part of the classifier. Dropout goes: it
    would fire in a trainable run. EfficientNet's head is Dropout -> Linear,
    so it still becomes Identity, exactly as before.
    """
    for attr in ("classifier", "fc", "head"):
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        if _find_linear(mod) is None:
            continue
        keep = []
        if isinstance(mod, nn.Sequential):
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    break
                if not isinstance(layer, nn.Dropout):
                    keep.append(layer)
        if keep:
            setattr(model, attr, nn.Sequential(*keep))
        else:
            setattr(model, attr, nn.Identity())
        return
    raise ValueError(f"no classifier found on {type(model).__name__}")


def _strip_pooling(model: nn.Module) -> None:
    """Drop the global average pool, keeping the spatial grid.

    torchvision backbones all end features -> avgpool -> flatten -> classifier,
    so removing avgpool makes the flatten produce the whole map.
    """
    if not hasattr(model, "avgpool"):
        raise ValueError(
            f"{type(model).__name__} has no avgpool to remove; "
            f"pool='flatten' is not supported for it")
    model.avgpool = nn.Identity()


def _measure_out_dim(model: nn.Module, size: int) -> int:
    """Run one empty frame through and look. Cheaper than deriving it.

    Deriving the width means knowing the backbone's stride and whether it
    pools, which differs per model; one forward pass just answers.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, size, size))
    model.train(was_training)
    if out.dim() != 2:
        raise ValueError(f"backbone returned {tuple(out.shape)}, expected (N, D)")
    return int(out.shape[1])


class FrameEncoder(nn.Module):
    """Every frame through the same CNN. (B, T, 3, H, W) -> (B, T, out_dim)."""

    def __init__(self, name: str = DEFAULT_BACKBONE, pretrained: bool = True,
                 freeze: bool = False, pool: str = "avg",
                 size: int = DEFAULT_SIZE):
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
        self.name = name
        self.pool = pool
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        if is_timm(name):
            self.backbone, mean, std = _timm_backbone(name, pretrained, pool)
        else:
            if pretrained:
                weights = "DEFAULT"
            else:
                weights = None
            self.backbone = get_model(name, weights=weights)
            _strip_classifier(self.backbone)
            if pool == "flatten":
                _strip_pooling(self.backbone)
        # the dataset hands over ImageNet colours; convert if the weights
        # were trained on others. Not saved: it follows from the name
        self.recolour = not (_same(mean, IMAGENET_MEAN)
                             and _same(std, IMAGENET_STD))
        if self.recolour:
            scale = []
            shift = []
            for m, s, im, ist in zip(mean, std, IMAGENET_MEAN, IMAGENET_STD):
                scale.append(ist / s)
                shift.append((im - m) / s)
            self.register_buffer("scale", torch.tensor(scale).view(1, 3, 1, 1),
                                 persistent=False)
            self.register_buffer("shift", torch.tensor(shift).view(1, 3, 1, 1),
                                 persistent=False)
        self.out_dim = _measure_out_dim(self.backbone, size)
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
        frames = x.flatten(0, 1)                    # (B*T, C, H, W)
        if self.recolour:
            # (x * std_in + mean_in - mean) / std, as one multiply-add
            frames = frames * self.scale + self.shift
        feats = self.backbone(frames)              # -> (B*T, D)
        return feats.unflatten(0, (b, t))          # -> (B, T, D)
