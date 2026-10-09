"""Clip-level augmentation.

The authors use **horizontal flip only**, so that is all there is here until
Day 7.2 asks for more.

The flip decision is made once per clip, not per frame. Flipping frame 7 and
not frame 8 would invent a camera jump in the middle of a shot, which is
exactly the discontinuity Day 3 spent its time removing.

Worth knowing what a flip means here: it turns a right-handed batter into a
left-handed one. A mirrored Cover Drive is still a Cover Drive, so the label
survives -- but if any class in this dataset is defined partly by which side
of the wicket the ball went, flipping teaches the model that both sides are
the same thing. The authors flipped, so the baseline flips.
"""
import torch

__all__ = ["ClipFlip", "build_transform"]


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


def build_transform(flip: float = 0.0):
    """The training transform, or None when nothing is asked for."""
    if not flip:
        return None
    return ClipFlip(flip)
