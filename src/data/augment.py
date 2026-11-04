"""Clip-level augmentation.

The authors use **horizontal flip only**, so the baseline flips and nothing
else. Days 38-40 add three more to try, all off unless a config asks.

Every decision is made once per clip, not per frame. Flipping frame 7 and
not frame 8 would invent a camera jump in the middle of a shot, which is
exactly the discontinuity Day 3 spent its time removing.

Worth knowing what a flip means here: it turns a right-handed batter into a
left-handed one. A mirrored Cover Drive is still a Cover Drive, so the label
survives -- but if any class in this dataset is defined partly by which side
of the wicket the ball went, flipping teaches the model that both sides are
the same thing. The authors flipped, so the baseline flips.
"""
import torch
import torch.nn.functional as F

from src.data.dataset import IMAGENET_MEAN, IMAGENET_STD

__all__ = ["ClipFlip", "ClipColour", "ClipCrop", "ClipTimeJitter",
           "build_transform"]


class ClipFlip:
    """Mirror the whole clip left-right, with probability p."""

    def __init__(self, p: float = 0.5):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be between 0 and 1, got {p}")
        self.p = p

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        if clip.dim() != 4:
            raise ValueError(f"expected (T, C, H, W), got {tuple(clip.shape)}")
        if self.p == 0.0:
            return clip
        if torch.rand(()) >= self.p:
            return clip
        return torch.flip(clip, dims=[-1])

    def __repr__(self) -> str:
        return f"ClipFlip(p={self.p})"


class ClipColour:
    """Brightness, contrast and saturation, each moved by up to +-strength.

    The clip arrives normalised, so it goes back to 0-1 for this and leaves
    normalised again."""

    def __init__(self, strength: float = 0.2):
        if not 0.0 < strength < 1.0:
            raise ValueError(f"strength must be between 0 and 1, got {strength}")
        self.strength = strength
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def _factor(self) -> float:
        return 1.0 + self.strength * (2.0 * torch.rand(()).item() - 1.0)

    def _grey(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        x = clip * self.std + self.mean
        x = x * self._factor()
        # one pivot for the clip, or each frame would drift on its own
        pivot = self._grey(x).mean()
        x = pivot + self._factor() * (x - pivot)
        grey = self._grey(x)
        x = grey + self._factor() * (x - grey)
        x = x.clamp(0.0, 1.0)
        return (x - self.mean) / self.std

    def __repr__(self) -> str:
        return f"ClipColour(strength={self.strength})"


class ClipCrop:
    """One random box for the whole clip, stretched back to full size. Keeps
    at least min_area of the picture, so a bat at the edge usually survives."""

    def __init__(self, min_area: float = 0.9):
        if not 0.0 < min_area < 1.0:
            raise ValueError(f"min_area must be between 0 and 1, got {min_area}")
        self.min_area = min_area

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        h, w = clip.shape[-2:]
        area = self.min_area + (1.0 - self.min_area) * torch.rand(()).item()
        side = area ** 0.5
        box_h = max(1, round(h * side))
        box_w = max(1, round(w * side))
        top = torch.randint(0, h - box_h + 1, ()).item()
        left = torch.randint(0, w - box_w + 1, ()).item()
        box = clip[:, :, top:top + box_h, left:left + box_w]
        return F.interpolate(box, size=(h, w), mode="bilinear",
                             align_corners=False)

    def __repr__(self) -> str:
        return f"ClipCrop(min_area={self.min_area})"


class ClipTimeJitter:
    """Each frame may be swapped for a neighbour up to `reach` away, in order.

    Every stored clip has exactly 15 frames, so there are none spare to pick
    from: this is as far as temporal jitter goes without a rebuild."""

    def __init__(self, reach: int = 1):
        if reach < 1:
            raise ValueError(f"reach must be at least 1, got {reach}")
        self.reach = reach

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        count = clip.shape[0]
        moved = torch.arange(count) + torch.randint(-self.reach, self.reach + 1,
                                                    (count,))
        order, _ = moved.clamp(0, count - 1).sort()
        return clip[order]

    def __repr__(self) -> str:
        return f"ClipTimeJitter(reach={self.reach})"


class _Chain:
    """Several transforms, one after another."""

    def __init__(self, steps):
        self.steps = steps

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        for step in self.steps:
            clip = step(clip)
        return clip

    def __repr__(self) -> str:
        return f"_Chain({self.steps})"


def build_transform(flip: float = 0.0, colour: float = 0.0, crop: float = 0.0,
                    time: int = 0):
    """The training transform, or None when nothing is asked for."""
    steps = []
    if time:
        steps.append(ClipTimeJitter(time))
    if crop:
        steps.append(ClipCrop(crop))
    if flip:
        steps.append(ClipFlip(flip))
    if colour:
        steps.append(ClipColour(colour))
    if not steps:
        return None
    if len(steps) == 1:
        return steps[0]
    return _Chain(steps)
