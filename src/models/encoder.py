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
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import get_model, list_models

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD

__all__ = ["FrameEncoder", "DEFAULT_BACKBONE", "DEFAULT_SIZE", "POOLS",
           "is_timm", "backbone_problem"]

DEFAULT_BACKBONE = "efficientnet_v2_s"
DEFAULT_SIZE = 224
POOLS = ("avg", "flatten")
SCALES = ("global", "multi")
# scales="multi": the stages whose maps join the last one, fine then mid,
# as (stage, channels) (Family F, Day 33). Normalised, shrunk to the last
# stage's grid and stacked as channels
TAPS = {"efficientnet_v2_s": ((3, 64), (5, 160))}


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


# torchvision backbones whose stochastic depth can be set (Days 38-40)
DROP_PATH_TORCHVISION = ("efficientnet_v2_s",)


def _timm_backbone(name: str, pretrained: bool, pool: str, drop_path=None):
    """A timm model without its classifier, and the colours it was trained on.

    num_classes=0 drops the classifier; global_pool="" also drops the pool,
    leaving the (N, C, 7, 7) grid, which a Flatten turns into one row.
    drop_path None keeps timm's default, which is none.
    """
    import timm
    extra = {}
    if drop_path is not None:
        extra["drop_path_rate"] = drop_path
    if pool == "flatten":
        model = timm.create_model(name, pretrained=pretrained, num_classes=0,
                                  global_pool="", **extra)
        mean = tuple(model.pretrained_cfg["mean"])
        std = tuple(model.pretrained_cfg["std"])
        return nn.Sequential(model, nn.Flatten(1)), mean, std
    model = timm.create_model(name, pretrained=pretrained, num_classes=0,
                              **extra)
    return model, tuple(model.pretrained_cfg["mean"]), \
        tuple(model.pretrained_cfg["std"])


def _same(a, b) -> bool:
    for x, y in zip(a, b):
        if abs(x - y) > 1e-6:
            return False
    return True


class _Checkpointed(nn.Sequential):
    """A backbone's stages, keeping only the boundaries between them for the
    backward pass and recomputing the rest. Same maths, about a third more
    compute, a fraction of the memory: how B07 trains at batch 4 (17.2).

    The recompute runs every BatchNorm forward a second time, which would
    update its running stats twice. Each stage but the last (torch does not
    recompute the last) gets momentum 1 - sqrt(1 - m), so the two updates add
    up to exactly one at m. Subclassing Sequential keeps the weight names, so
    a checkpoint loads with or without it.
    """

    def __init__(self, stages):
        super().__init__(stages)                    # an OrderedDict, by name
        for stage in list(self)[:-1]:
            for layer in stage.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    if layer.momentum is not None:
                        layer.momentum = 1.0 - (1.0 - layer.momentum) ** 0.5

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint_sequential
            return checkpoint_sequential(self, len(self), x,
                                         use_reentrant=False)
        return super().forward(x)


def _checkpoint_stages(model: nn.Module) -> None:
    """Wrap the backbone's list of stages in _Checkpointed."""
    for attr in ("trunk_output", "features"):
        stages = getattr(model, attr, None)
        if isinstance(stages, nn.Sequential) and len(stages) > 1:
            # by name, or RegNet's block1..block4 would become 0..3 and no
            # saved weights would load
            named = OrderedDict()
            for name, stage in stages.named_children():
                named[name] = stage
            setattr(model, attr, _Checkpointed(named))
            return
    raise ValueError(f"{type(model).__name__} has no list of stages to "
                     f"checkpoint")


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


class FrameEncoder(nn.Module):
    """Every frame through the same CNN. (B, T, 3, H, W) -> (B, T, out_dim)."""

    def __init__(self, name: str = DEFAULT_BACKBONE, pretrained: bool = True,
                 freeze: bool = False, pool: str = "avg",
                 size: int = DEFAULT_SIZE, grad_checkpoint: bool = False,
                 drop_path=None, scales: str = "global", extra_size=None):
        super().__init__()
        if pool not in POOLS:
            raise ValueError(f"pool must be one of {POOLS}, got {pool!r}")
        if scales not in SCALES:
            raise ValueError(f"scales must be one of {SCALES}, got {scales!r}")
        if scales == "multi":
            if name not in TAPS:
                raise ValueError(f"multi-scale knows the stages of "
                                 f"{tuple(TAPS)}, not {name}")
            if pool != "flatten":
                raise ValueError("multi-scale stacks grids; it needs "
                                 "pool='flatten'")
            if grad_checkpoint:
                raise ValueError("multi-scale walks the stages itself; "
                                 "grad_checkpoint is not supported with it")
        self.name = name
        self.pool = pool
        self.scales = scales
        self.extra_size = extra_size
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        if is_timm(name):
            if grad_checkpoint:
                raise ValueError("grad_checkpoint is torchvision-only for now")
            self.backbone, mean, std = _timm_backbone(name, pretrained, pool,
                                                      drop_path)
        else:
            extra = {}
            if drop_path is not None:
                # torchvision bakes its own in (0.2 for EfficientNetV2-S)
                if name not in DROP_PATH_TORCHVISION:
                    raise ValueError(f"drop_path works on timm backbones and "
                                     f"{DROP_PATH_TORCHVISION}, not {name}")
                extra["stochastic_depth_prob"] = drop_path
            if pretrained:
                weights = "DEFAULT"
            else:
                weights = None
            self.backbone = get_model(name, weights=weights, **extra)
            _strip_classifier(self.backbone)
            if pool == "flatten":
                _strip_pooling(self.backbone)
            if grad_checkpoint:
                _checkpoint_stages(self.backbone)
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
        if scales == "multi":
            # the inner stages come out ~5x wider than the last one (std
            # ~4 against 0.87, 2026-11-01) and saturated the GRU, so each
            # channel is normalised first. No learned weights: only stats
            norms = []
            for _, channels in TAPS[name]:
                norms.append(nn.BatchNorm2d(channels, affine=False))
            self.tap_norms = nn.ModuleList(norms)
        self.out_dim = self._measure(size)
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

    def _one_size(self, frames):
        """(N, 3, H, W) -> (N, D) at one resolution."""
        if self.scales == "global":
            return self.backbone(frames)
        # the last stage's grid first, so its numbers sit where B01's do,
        # then the fine and mid maps shrunk to that grid (F01)
        taps = []
        for stage_at, _ in TAPS[self.name]:
            taps.append(stage_at)
        x = frames
        early = []
        for i, stage in enumerate(self.backbone.features):
            x = stage(x)
            if i in taps:
                early.append(x)
        maps = [x]
        for k, found in enumerate(early):
            maps.append(F.adaptive_avg_pool2d(self.tap_norms[k](found),
                                              x.shape[-2:]))
        return torch.cat(maps, dim=1).flatten(1)

    def _describe(self, frames):
        """(N, 3, H, W) -> (N, out_dim): what the head reads for a frame.
        With extra_size, the same frame shrunk joins it (F04)."""
        out = self._one_size(frames)
        if self.extra_size:
            small = F.interpolate(frames, size=(self.extra_size,
                                                self.extra_size),
                                  mode="bilinear", antialias=True,
                                  align_corners=False)
            out = torch.cat([out, self._one_size(small)], dim=1)
        return out

    def _measure(self, size):
        """Run one empty frame through and look. Cheaper than deriving the
        width, which depends on the backbone's stride and pooling."""
        # every layer in eval, set directly: train() reads self.frozen,
        # not set yet, and a zero frame must not move any running stats
        modes = []
        for layer in self.modules():
            modes.append((layer, layer.training))
            layer.training = False
        with torch.no_grad():
            out = self._describe(torch.zeros(1, 3, size, size))
        for layer, mode in modes:
            layer.training = mode
        if out.dim() != 2:
            raise ValueError(f"backbone returned {tuple(out.shape)}, "
                             f"expected (N, D)")
        return int(out.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected (B, T, C, H, W), got {tuple(x.shape)}")
        b, t = x.shape[:2]
        frames = x.flatten(0, 1)                    # (B*T, C, H, W)
        if self.recolour:
            # (x * std_in + mean_in - mean) / std, as one multiply-add
            frames = frames * self.scale + self.shift
        feats = self._describe(frames)             # -> (B*T, D)
        return feats.unflatten(0, (b, t))          # -> (B, T, D)
