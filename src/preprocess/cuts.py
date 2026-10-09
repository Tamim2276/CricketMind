"""Find camera cuts, so a gap in detection can be told from a change of shot.

Measured 2026-10-09 over 60 clips: of the detection gaps 3 frames or longer,
only 21% sit next to a camera cut. The other 79% are the detector failing on
a continuous shot -- 230 frames against 48. Trimming on gap length alone
throws away five frames of good footage for every bad one.

The measure is Day 2's: mean absolute difference between consecutive frames,
greyscale, downscaled. A cut changes every pixel at once; a batter swinging
changes a few hundred.
"""
from typing import List

import cv2
import numpy as np

__all__ = ["find_cuts", "shots", "SMALL", "FLOOR", "RATIO"]

SMALL = (96, 64)    # differences survive downscaling; decoding noise does not
FLOOR = 12.0        # absolute floor, on a 0-255 scale
RATIO = 3.0         # ... or this many times the clip's own typical motion


def frame_diffs(clip) -> np.ndarray:
    """Mean absolute greyscale change between consecutive frames."""
    g = [cv2.cvtColor(cv2.resize(f, SMALL), cv2.COLOR_RGB2GRAY).astype(np.float32)
         for f in clip]
    return np.array([np.abs(g[i + 1] - g[i]).mean() for i in range(len(g) - 1)])


def find_cuts(clip) -> List[int]:
    """Frame indices where a new shot starts. Empty for an uninterrupted clip.

    The threshold is relative to the clip's own motion, because a close-up
    shakes far more than a wide shot and a fixed number would flag one and
    miss the other.
    """
    if len(clip) < 3:
        return []
    d = frame_diffs(clip)
    # the cuts must not set the bar they are measured against, so drop the
    # top decile first; duplicate frames are excluded from the other end
    calm = d[d <= np.percentile(d, 90)]
    moving = calm[calm > 1.0]
    typical = float(np.median(moving)) if len(moving) else 1.0
    thr = max(RATIO * typical, FLOOR)
    return [i + 1 for i, v in enumerate(d) if v > thr]


def shots(clip) -> List[range]:
    """The clip split into continuous shots at its cuts."""
    bounds = [0] + find_cuts(clip) + [len(clip)]
    return [range(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
