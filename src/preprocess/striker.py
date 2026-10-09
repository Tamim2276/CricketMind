"""Pick the facing batter out of the detector's candidates.

Half of all frames return two or more Strikers -- the non-striker, fielders,
the keeper, and station logos the detector mistakes for people. Only one of
them is the shot.

Measured on 36 hand-labelled multi-striker frames from the train split:

    nearest the frame centre   97%        largest box          76%
    nearest the best bat       88%        highest confidence   68%

Centrality wins because the dataset is already built around the batter -- the
true box centre sits between x=0.44 and x=0.54 of the frame width in 80% of
frames. That is a fact about CricShot10k, not about cricket, and the rule will
not transfer to uncropped broadcast footage.
"""
from typing import Optional, Sequence

__all__ = ["pick_striker", "candidates", "plausible",
           "MIN_AREA", "ASPECT", "EDGE_PX"]

MIN_AREA = 0.004        # of the frame; smaller cannot be usefully cropped
ASPECT = (0.15, 2.5)    # width / height -- people are not ribbons
EDGE_PX = 2             # the broadcaster's logo lives against the border


def plausible(det, w: int, h: int) -> bool:
    """Could this box be a person? Rejects logos, scorebars and slivers."""
    x1, y1, x2, y2 = det.box
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return False
    if not ASPECT[0] <= bw / bh <= ASPECT[1]:
        return False
    if x1 <= EDGE_PX or x2 >= w - EDGE_PX:
        return False
    return bw * bh >= MIN_AREA * w * h


def candidates(dets: Sequence, shape) -> list:
    """Every box that could be the batter, most central first.

    crop.py needs the runners-up: one frame in five picks the wrong person,
    and the right one is usually sitting second on this list.
    """
    h, w = shape[:2] if not hasattr(shape, "shape") else shape.shape[:2]

    def off_centre(d):
        x1, y1, x2, y2 = d.box
        return ((x1 + x2) / 2 - w / 2) ** 2 + ((y1 + y2) / 2 - h / 2) ** 2

    return sorted((d for d in dets
                   if d.cls == "Striker" and plausible(d, w, h)), key=off_centre)


def pick_striker(dets: Sequence, shape) -> Optional[object]:
    """The facing batter, or None if nothing in the frame can be one.

    `shape` is the frame, or (h, w). None is a real answer, not a failure --
    9% of frames genuinely contain no batter, mostly after a camera cut.
    """
    c = candidates(dets, shape)
    return c[0] if c else None
