"""Any clip length down to a fixed 15 frames.

Two jobs, in this order. **Drop duplicates first**: the 28-frame clips are
24-frame clips with repeats inserted by 25->30fps conversion, and sampling
across them spends slots on identical pictures. Measured over 150 clips:

    24-frame clips   0.4% duplicates
    28-frame clips  15.3%
    49-frame clips   0.0%

The threshold is not a guess. Nothing is bit-identical -- the clips were
re-encoded -- but there is an empty gap in the data: duplicate pairs top out
at a mean absolute difference of 0.927 and real motion starts at 1.045.

**Then sample evenly**, step size taken from the clip. A cricket shot's
information is in the swing, which is in the middle, so taking the first 15
would throw the follow-through away on a 49-frame clip.
"""
from typing import List, Tuple

import numpy as np

__all__ = ["drop_duplicates", "sample_indices", "sample", "TARGET", "DUP_TOL"]

TARGET = 15
DUP_TOL = 1.0       # mean absolute 0-255 difference; the gap is 0.93 to 1.05


def drop_duplicates(clip: np.ndarray, tol: float = DUP_TOL) -> Tuple[np.ndarray, List[int]]:
    """Frames with the repeats removed, plus which source frames survived."""
    if len(clip) < 2:
        return clip, list(range(len(clip)))
    keep = [0]
    for i in range(1, len(clip)):
        last = clip[keep[-1]].astype(np.int16)
        diff = np.abs(clip[i].astype(np.int16) - last).mean()
        if diff >= tol:
            keep.append(i)
    return clip[keep], keep


def sample_indices(n: int, k: int = TARGET) -> List[int]:
    """`k` indices spread over `n` frames, first and last always included.

    Short clips repeat frames rather than fail -- 8% of clips come out of
    cropping with fewer than 15, and a repeated frame is better than no clip.
    """
    if n < 1:
        raise ValueError("no frames to sample")
    if n == 1:
        return [0] * k
    out = []
    for position in np.linspace(0, n - 1, k):
        out.append(int(round(position)))
    return out


def sample(frames: np.ndarray, k: int = TARGET) -> np.ndarray:
    """Exactly `k` frames, evenly spaced."""
    return frames[sample_indices(len(frames), k)]
