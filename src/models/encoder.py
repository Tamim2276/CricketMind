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
"""
import torch
import torch.nn as nn
from torchvision.models import get_model

__all__ = ["FrameEncoder", "DEFAULT_BACKBONE", "DEFAULT_SIZE", "POOLS"]

DEFAULT_BACKBONE = "efficientnet_v2_s"
DEFAULT_SIZE = 224
POOLS = ("avg", "flatten")


def _find_linear(mod: nn.Module):
    """The Linear layer inside a torchvision head, or None."""
    if isinstance(mod, nn.Linear):
        return mod
    for layer in mod.modules():
        if isinstance(layer, nn.Linear):
            return layer
    return None


def _strip_classifier(model: nn.Module) -> None:
    """Drop the 1000-class ImageNet head."""
    for attr in ("classifier", "fc", "head"):
        mod = getattr(model, attr, None)
        if mod is None:
            continue
        if _find_linear(mod) is None:
            continue
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
        if pretrained:
            weights = "DEFAULT"
        else:
            weights = None
        self.backbone = get_model(name, weights=weights)
        _strip_classifier(self.backbone)
        if pool == "flatten":
            _strip_pooling(self.backbone)
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
        feats = self.backbone(x.flatten(0, 1))     # (B*T, C, H, W) -> (B*T, D)
        return feats.unflatten(0, (b, t))          # -> (B, T, D)
