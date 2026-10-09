"""Turn per-frame striker picks into one steady cropped clip.

A clip is not 24 independent frames. Cropping each to its own box makes the
batter jump and pulse, because the detector's box stretches whenever the bat
leaves his outline -- measured width swing within one clip is x2.3.

Four things happen here, in this order, because each changes what the next
sees:

  1. reject rogue boxes   8% of picks sit more than one body-height from the
                          clip's median centre: a fielder in a crowd shot
  2. choose a shot        split the clip at its camera cuts and keep the shot
                          holding the most detections. Gaps inside it are the
                          detector blinking, whatever their length, and get
                          interpolated. Measured over 60 clips: only 21% of
                          gaps of 3+ frames sit next to a cut, so trimming on
                          gap length alone binned 230 good frames to drop 48
  3. smooth               median then mean, so the box follows the batter
                          without vibrating
  4. square and pad       size from the box HEIGHT, smoothed hard, plus 25%
                          padding. Height swings x1.3 within a clip against
                          width's x2.3, because width is the bat leaving his
                          outline and height is the camera zooming. Sizing on
                          height follows a zoom without pulsing on the swing.
                          Measured over 596 frames: the bat stays whole in 84%
                          and the batter fills 67% of the crop
"""
from typing import List, NamedTuple, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.preprocess.cuts import find_cuts
from src.preprocess.striker import candidates, pick_striker

__all__ = ["crop_clip", "track_boxes", "track_picks", "segment",
           "frame_mask",
           "smooth_boxes", "CropResult",
           "OUTLIER", "PAD", "SIZE", "MIN_FRAMES", "MIN_DETECTED",
           "BAT_REACH"]

OUTLIER = 1.0       # body-heights from the clip median; rejects 8% of picks
TRACK_IOU = 0.2     # overlap with the dominant track to count as the batter
BAT_REACH = 1.0     # body-heights; further than this is not his bat
MIN_DETECTED = 0.4  # of a shot must be real detections, or the track is fiction
PAD = 0.25          # each side, as a fraction of the square's side
MEDIAN_W = 5        # spike removal
MEAN_W = 3          # the actual smoothing
SIZE_W = 9          # the crop size is smoothed harder; a pulse is very visible
SIZE = 224
MIN_FRAMES = 8      # below this there is no shot left to classify


class CropResult(NamedTuple):
    frames: np.ndarray          # (n, SIZE, SIZE, 3) uint8, n may be 0
    idxs: List[int]             # which source frames these came from
    note: str                   # why frames were dropped; for the run log


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0.0


def _strays(boxes) -> List[Optional[tuple]]:
    """Drop picks too far from the clip's median centre to be the same person."""
    found = [b for b in boxes if b]
    if len(found) < 3:
        return list(boxes)                 # too few to know what normal is

    cs = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in found])
    mx, my = np.median(cs, axis=0)
    body = float(np.median([b[3] - b[1] for b in found])) or 1.0

    out = []
    for b in boxes:
        if b is None:
            out.append(None)
            continue
        stray = (abs((b[0] + b[2]) / 2 - mx) + abs((b[1] + b[3]) / 2 - my)) / body
        out.append(None if stray > OUTLIER else b)
    return out


def _reference(boxes) -> Optional[List[Optional[tuple]]]:
    """A per-frame guess at where the batter is, from the frames that agree.

    Built from the majority, so a run of frames that picked a fielder is
    outvoted rather than followed.
    """
    found = [b for b in boxes if b]
    if len(found) < 3:
        return None
    med = tuple(np.median(np.array(found), axis=0))
    core = [b if b and _iou(b, med) >= TRACK_IOU else None for b in boxes]
    if not any(core):
        return None
    seen = [i for i, b in enumerate(core) if b]
    # no reference outside the agreed stretch. Holding the end box there lets
    # the bowler running in before the batter is detected pass as the batter,
    # which is every remaining error measured over 60 clips
    ref: List[Optional[tuple]] = [None] * len(core)
    ref[seen[0]:seen[-1] + 1] = _fill(core, seen[0], seen[-1])
    return ref


def track_picks(per_frame, shape) -> List[Optional[object]]:
    """The chosen Detection per frame: the batter, tracked across the clip."""
    h, w = shape[:2]
    cands = [candidates(d, (h, w)) for d in per_frame]
    first = _strays([c[0].box if c else None for c in cands])

    ref = _reference(first)
    if ref is None:
        return [c[0] if c and b else None for c, b in zip(cands, first)]

    # pass 2: the most central box is not always the batter, but the batter is
    # almost always on the list
    out = []
    for c, r in zip(cands, ref):
        if r is None or not c:
            out.append(None)
            continue
        best = max(c, key=lambda d: _iou(d.box, r))
        out.append(best if _iou(best.box, r) >= TRACK_IOU else None)
    return out


def track_boxes(per_frame, shape) -> List[Optional[tuple]]:
    """One box per frame: the batter, tracked across the clip."""
    return [None if p is None else tuple(map(float, p.box))
            for p in track_picks(per_frame, shape)]


def frame_mask(striker, dets=()):
    """The batter and his bat as one mask, or None if he has no silhouette.

    The bat is a separate detection, so keeping only the striker's outline
    would throw away the one object that tells a Sweep from a Pull. Bats more
    than a body-height away belong to somebody else.
    """
    if striker is None or striker.mask is None:
        return None
    m = striker.mask.astype(bool)
    x1, y1, x2, y2 = striker.box
    cx, cy, bh = (x1 + x2) / 2, (y1 + y2) / 2, (y2 - y1) or 1.0
    for d in dets:
        if d.cls != "Bat" or d.mask is None:
            continue
        bx, by = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
        if ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 / bh <= BAT_REACH:
            m |= d.mask.astype(bool)
    return m


def segment(frame, striker, dets=(), mask=None):
    """Black out everything that is not the batter or his bat."""
    m = frame_mask(striker, dets) if mask is None else mask
    if m is None:
        return frame
    out = np.zeros_like(frame)
    out[m] = frame[m]
    return out


def _shift(m, dx: int, dy: int):
    """Translate a mask without wrapping content round the edges."""
    out = np.zeros_like(m)
    h, w = m.shape
    ys, ye = max(0, dy), min(h, h + dy)
    xs, xe = max(0, dx), min(w, w + dx)
    if ye > ys and xe > xs:
        out[ys:ye, xs:xe] = m[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def carried_mask(picks, per_frame, i: int, box):
    """The nearest real silhouette, moved onto `box`.

    An interpolated frame has no mask of its own. Measured on 463 frames that
    do have one, by pretending they do not: carrying the neighbour scores
    IoU 0.80 one frame away and 0.65 at three, against 0.51 for filling the
    bounding box in as a rectangle. So carry, and take whichever side is
    nearer.
    """
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    for d in range(1, len(picks)):
        for j in (i - d, i + d):
            if not 0 <= j < len(picks):
                continue
            m = frame_mask(picks[j], per_frame[j])
            if m is None:
                continue
            q = picks[j].box
            return _shift(m, int(round(cx - (q[0] + q[2]) / 2)),
                          int(round(cy - (q[1] + q[3]) / 2)))
    return None


def _fill(boxes, lo: int, hi: int):
    """Linear interpolation across the blinks inside the span."""
    out = [boxes[i] for i in range(lo, hi + 1)]
    known = [i for i, b in enumerate(out) if b is not None]
    for a, b in zip(known, known[1:]):
        for k in range(a + 1, b):
            t = (k - a) / (b - a)
            out[k] = tuple(out[a][j] + t * (out[b][j] - out[a][j]) for j in range(4))
    return out


def smooth_boxes(boxes: Sequence[tuple]) -> np.ndarray:
    """Median filter to kill spikes, then a mean to take the shake out."""
    a = np.asarray(boxes, dtype=np.float64)
    for width, fn in ((MEDIAN_W, np.median), (MEAN_W, np.mean)):
        r = width // 2
        pad = np.pad(a, ((r, r), (0, 0)), mode="edge")
        a = np.stack([fn(pad[i:i + width], axis=0) for i in range(len(a))])
    return a


def _runmean(v: np.ndarray, width: int) -> np.ndarray:
    r = width // 2
    pad = np.pad(v, (r, r), mode="edge")
    return np.array([pad[i:i + width].mean() for i in range(len(v))])


def _square(box, side: float, fw: int, fh: int):
    """A square of fixed size on the box centre, nudged to stay in frame."""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    s = min(side, fw, fh)                       # cannot be bigger than the frame
    x1 = min(max(cx - s / 2, 0), fw - s)
    y1 = min(max(cy - s / 2, 0), fh - s)
    return int(round(x1)), int(round(y1)), int(round(s))


def crop_clip(clip: np.ndarray, per_frame, size: int = SIZE, pad: float = PAD,
              min_frames: int = MIN_FRAMES, min_detected: float = MIN_DETECTED,
              cuts=None, segmented: bool = False) -> CropResult:
    """Crop a clip to the batter. `per_frame` is detect()'s output for it.

    `segmented=True` blacks out the background first, which needs
    detect(masks=True). It is a variant to compare, not a default -- the
    authors' classifier wants it, but that is their claim, not a measurement.
    """
    fh, fw = clip.shape[1:3]
    picks = track_picks(per_frame, (fh, fw))
    boxes = [None if p is None else tuple(map(float, p.box)) for p in picks]
    rogue = sum(1 for p, b in zip(per_frame, boxes)
                if b is None and pick_striker(p, (fh, fw)) is not None)
    swapped = sum(1 for p, b in zip(per_frame, boxes)
                  if b is not None and (lambda f: f is not None
                                        and _iou(tuple(f.box), b) < TRACK_IOU)
                  (pick_striker(p, (fh, fw))))
    empty = np.empty((0, size, size, 3), np.uint8)

    bounds = [0] + (list(cuts) if cuts is not None else find_cuts(clip)) + [len(clip)]
    shot = max((range(a, b) for a, b in zip(bounds, bounds[1:]) if b > a),
               key=lambda r: sum(boxes[i] is not None for i in r))
    seen = [i for i in shot if boxes[i] is not None]
    if not seen:
        return CropResult(empty, [], "no striker anywhere")

    lo, hi = seen[0], seen[-1]          # do not invent a box before the first
    density = len(seen) / (hi - lo + 1)
    if density < min_detected:
        return CropResult(empty, [], f"only {density:.0%} of the shot detected")
    span = _fill(boxes, lo, hi)
    if len(span) < min_frames:
        return CropResult(empty, [], f"shot only {len(span)} frames")

    sm = smooth_boxes(span)
    # size on height, not on max(w, h): width is the bat, height is the camera
    sides = _runmean(sm[:, 3] - sm[:, 1], SIZE_W) * (1 + 2 * pad)

    out = np.empty((len(span), size, size, 3), np.uint8)
    nomask = carried = 0
    for k, box in enumerate(sm):
        x, y, s = _square(box, float(sides[k]), fw, fh)
        src = clip[lo + k]
        if segmented:
            m = frame_mask(picks[lo + k], per_frame[lo + k])
            if m is None:
                m = carried_mask(picks, per_frame, lo + k, box)
                carried += m is not None
            nomask += m is None
            src = segment(src, None, mask=m)
        patch = src[y:y + s, x:x + s]
        out[k] = cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)

    note = f"kept {len(span)}/{len(clip)}"
    if rogue:
        note += f", {rogue} rogue"
    if swapped:
        note += f", {swapped} swapped"
    if len(span) - len(seen):
        note += f", filled {len(span) - len(seen)}"
    if len(clip) - len(span):
        note += f", trimmed {len(clip) - len(span)}"
    if segmented and carried:
        note += f", {carried} carried"
    if segmented and nomask:
        note += f", {nomask} unsegmented"
    return CropResult(out, list(range(lo, hi + 1)), note)
