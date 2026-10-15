"""Turn per-frame striker picks into one steady cropped clip.

A clip is not 24 independent frames. Cropping each to its own box makes the
batter jump and pulse, because the detector's box stretches whenever the bat
leaves his outline -- measured width swing within one clip is x2.3.

Four things happen here, in this order, because each changes what the next
sees:

  1. track the batter     two passes. Centrality alone picks the wrong person
                          somewhere in 22% of clips, so pass 1 establishes the
                          clip's dominant track and pass 2 re-picks each frame
                          against it, taking the runner-up where the leader
                          disagrees with the clip as a whole. Two passes, not a
                          running chain: chaining off the previous frame lets
                          one bad frame poison every frame after it
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
from typing import List, NamedTuple, Optional, Sequence

import cv2
import numpy as np

from src.preprocess.cuts import find_cuts
from src.preprocess.striker import candidates, pick_striker

__all__ = ["crop_clip", "track_boxes", "track_picks", "segment",
           "frame_mask", "carried_mask", "smooth_boxes", "CropResult",
           "overlay", "blackout", "part_masks", "SEG_MODE", "ALPHA",
           "BODY_TINT", "BAT_TINT", "TINT", "SIDE_FROM",
           "OUTLIER", "PAD", "SIZE", "MIN_FRAMES", "MIN_DETECTED",
           "BAT_REACH"]

OUTLIER = 1.0       # body-heights from the clip median; rejects 8% of picks
TRACK_IOU = 0.2     # overlap with the dominant track to count as the batter
BAT_REACH = 1.0     # body-heights; further than this is not his bat
MIN_DETECTED = 0.4  # of a shot must be real detections, or the track is fiction
PAD = 0.25          # each side, as a fraction of the square's side
SIDE_FROM = "height"   # or "max"; see crop_clip

# The segmentation layer, to the paper's own specification: alpha 0.5, striker
# blue, bat green, striker painted first so the bat stays visible where it
# overlaps. Colours are RGB -- read_clip(rgb=True) is what fills these arrays.
SEG_MODE = "overlay"
ALPHA = 0.5
BODY_TINT = (0, 0, 255)   # blue
BAT_TINT = (0, 255, 0)    # green
TINT = BODY_TINT          # so older imports still resolve

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
    overlap = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def _centre(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _as_box(pick) -> Optional[tuple]:
    """A pick's box as plain floats, or None."""
    if pick is None:
        return None
    box = []
    for v in pick.box:
        box.append(float(v))
    return tuple(box)


def _strays(boxes) -> List[Optional[tuple]]:
    """Drop picks too far from the clip's median centre to be the same person."""
    found = []
    for box in boxes:
        if box:
            found.append(box)
    if len(found) < 3:
        return list(boxes)                 # too few to know what normal is

    centres = []
    heights = []
    for box in found:
        centres.append(_centre(box))
        heights.append(box[3] - box[1])
    mx, my = np.median(np.array(centres), axis=0)
    body = float(np.median(heights))
    if body == 0:
        body = 1.0

    out = []
    for box in boxes:
        if box is None:
            out.append(None)
            continue
        cx, cy = _centre(box)
        stray = (abs(cx - mx) + abs(cy - my)) / body
        if stray > OUTLIER:
            out.append(None)
        else:
            out.append(box)
    return out


def _reference(boxes) -> Optional[List[Optional[tuple]]]:
    """A per-frame guess at where the batter is, from the frames that agree.

    Built from the majority, so a run of frames that picked a fielder is
    outvoted rather than followed.
    """
    found = []
    for box in boxes:
        if box:
            found.append(box)
    if len(found) < 3:
        return None
    median_box = tuple(np.median(np.array(found), axis=0))

    core = []
    for box in boxes:
        if box and _iou(box, median_box) >= TRACK_IOU:
            core.append(box)
        else:
            core.append(None)

    seen = []
    for i, box in enumerate(core):
        if box is not None:
            seen.append(i)
    if not seen:
        return None

    # no reference outside the agreed stretch. Holding the end box there lets
    # the bowler running in before the batter is detected pass as the batter,
    # which is every remaining error measured over 60 clips
    ref: List[Optional[tuple]] = [None] * len(core)
    ref[seen[0]:seen[-1] + 1] = _fill(core, seen[0], seen[-1])
    return ref


def track_picks(per_frame, shape) -> List[Optional[object]]:
    """The chosen Detection per frame: the batter, tracked across the clip."""
    h, w = shape[:2]

    cands = []
    for dets in per_frame:
        cands.append(candidates(dets, (h, w)))

    leaders = []
    for frame_cands in cands:
        if frame_cands:
            leaders.append(frame_cands[0].box)
        else:
            leaders.append(None)
    leaders = _strays(leaders)

    ref = _reference(leaders)
    if ref is None:
        out = []
        for frame_cands, box in zip(cands, leaders):
            if frame_cands and box:
                out.append(frame_cands[0])
            else:
                out.append(None)
        return out

    # pass 2: the most central box is not always the batter, but the batter is
    # almost always on the list
    out = []
    for frame_cands, want in zip(cands, ref):
        if want is None or not frame_cands:
            out.append(None)
            continue
        best = None
        best_iou = -1.0
        for det in frame_cands:
            overlap = _iou(det.box, want)
            if overlap > best_iou:
                best_iou = overlap
                best = det
        if best_iou >= TRACK_IOU:
            out.append(best)
        else:
            out.append(None)
    return out


def track_boxes(per_frame, shape) -> List[Optional[tuple]]:
    """One box per frame: the batter, tracked across the clip."""
    out = []
    for pick in track_picks(per_frame, shape):
        out.append(_as_box(pick))
    return out


def part_masks(striker, dets=()):
    """(striker, bat) silhouettes, kept apart. Either may be None.

    They stay separate because the paper colours them differently and paints
    the bat *second*, "ensuring that the bat remains fully visible even when
    it overlaps the striker". Merging them first throws that away, and the bat
    is the one object that tells a Sweep from a Pull.

    Bats more than a body-height away belong to somebody else.
    """
    if striker is None or striker.mask is None:
        return None, None
    body_mask = striker.mask.astype(bool)

    cx, cy = _centre(striker.box)
    body = striker.box[3] - striker.box[1]
    if body == 0:
        body = 1.0

    bat_mask = None
    for det in dets:
        if det.cls != "Bat" or det.mask is None:
            continue
        bx, by = _centre(det.box)
        gap = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 / body
        if gap <= BAT_REACH:
            if bat_mask is None:
                bat_mask = det.mask.astype(bool)
            else:
                bat_mask = bat_mask | det.mask.astype(bool)
    return body_mask, bat_mask


def frame_mask(striker, dets=()):
    """The batter and his bat as one mask, for the blackout variant."""
    body_mask, bat_mask = part_masks(striker, dets)
    if body_mask is None:
        return None
    if bat_mask is None:
        return body_mask
    return body_mask | bat_mask


def blackout(frame, striker, dets=(), mask=None):
    """Black out everything that is not the batter or his bat.

    This is **not** what the paper describes, and it is kept only so the
    result stays reproducible: the `seg` variant built from it scored 62.37%
    against box's 79.23%, because deleting the pitch and the fielders deletes
    what separates a Lofted Offside from a Lofted Legside.
    """
    if mask is None:
        mask = frame_mask(striker, dets)
    if mask is None:
        return frame
    out = np.zeros_like(frame)
    out[mask] = frame[mask]
    return out


def _blend(out, mask, colour, alpha):
    if mask is None or not mask.any():
        return
    tint = np.array(colour, dtype=np.float32)
    kept = out[mask].astype(np.float32)
    out[mask] = np.rint((1.0 - alpha) * kept + alpha * tint).astype(out.dtype)


def overlay(frame, striker=None, dets=(), masks=None, alpha=ALPHA,
            body_colour=BODY_TINT, bat_colour=BAT_TINT):
    """The paper's segmentation layer, to its own specification.

    Section III-? states it exactly: "the predicted masks were overlaid with
    semi-transparent colors (alpha = 0.5), highlighting the striker in blue
    and the bat in green. Semi-transparent color is chosen instead of full
    solid to avoid the loss of full information such as the position of the
    hands and rest of the body. **The striker is colored first, followed by
    the bat, ensuring that the bat remains fully visible even when it
    overlaps the striker.**"

    So: two colours, not one; and the order matters. Painting the bat second
    means the overlap belongs to the bat, which is the object that names the
    shot. The first version of this merged both masks and painted them one
    grey, and scored 75.51% against plain cropping's 78.14%.

    Colours are **RGB**, because `read_clip(rgb=True)` is what fills these
    arrays. A grey tint hid that; blue and green do not.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if masks is None:
        masks = part_masks(striker, dets)
    body_mask, bat_mask = masks
    if body_mask is None and bat_mask is None:
        return frame
    if alpha == 0.0:
        return frame

    out = frame.copy()
    _blend(out, body_mask, body_colour, alpha)   # striker first
    _blend(out, bat_mask, bat_colour, alpha)     # then the bat, on top
    return out


def _merge(masks):
    """The two parts as one mask, for the blackout variant."""
    if masks is None:
        return None
    body_mask, bat_mask = masks
    if body_mask is None:
        return bat_mask
    if bat_mask is None:
        return body_mask
    return body_mask | bat_mask


def segment(frame, striker=None, dets=(), masks=None, mode=SEG_MODE,
            alpha=ALPHA, body_colour=BODY_TINT, bat_colour=BAT_TINT):
    """The segmentation layer. `mode` picks which one."""
    if mode == "overlay":
        return overlay(frame, striker, dets, masks, alpha=alpha,
                       body_colour=body_colour, bat_colour=bat_colour)
    if mode == "blackout":
        if masks is None:
            return blackout(frame, striker, dets)
        return blackout(frame, striker, dets, _merge(masks))
    raise ValueError(f"unknown segmentation mode: {mode!r}")


def _shift(mask, dx: int, dy: int):
    """Translate a mask without wrapping content round the edges."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    ys, ye = max(0, dy), min(h, h + dy)
    xs, xe = max(0, dx), min(w, w + dx)
    if ye > ys and xe > xs:
        out[ys:ye, xs:xe] = mask[ys - dy:ye - dy, xs - dx:xe - dx]
    return out


def carried_mask(picks, per_frame, i: int, box):
    """The nearest real (striker, bat) silhouettes, moved onto `box`.

    An interpolated frame has no mask of its own. Measured on 463 frames that
    do have one, by pretending they do not: carrying the neighbour scores
    IoU 0.80 one frame away and 0.65 at three, against 0.51 for filling the
    bounding box in as a rectangle. So carry, and take whichever side is
    nearer.
    """
    cx, cy = _centre(box)
    for step in range(1, len(picks)):
        for j in (i - step, i + step):
            if j < 0 or j >= len(picks):
                continue
            body_mask, bat_mask = part_masks(picks[j], per_frame[j])
            if body_mask is None and bat_mask is None:
                continue
            qx, qy = _centre(picks[j].box)
            dx = int(round(cx - qx))
            dy = int(round(cy - qy))
            moved = []
            for one in (body_mask, bat_mask):
                if one is None:
                    moved.append(None)
                else:
                    moved.append(_shift(one, dx, dy))
            return moved[0], moved[1]
    return None, None


def _fill(boxes, lo: int, hi: int):
    """Linear interpolation across the blinks inside the span."""
    out = []
    for i in range(lo, hi + 1):
        out.append(boxes[i])

    known = []
    for i, box in enumerate(out):
        if box is not None:
            known.append(i)

    for a, b in zip(known, known[1:]):
        for k in range(a + 1, b):
            t = (k - a) / (b - a)
            point = []
            for j in range(4):
                point.append(out[a][j] + t * (out[b][j] - out[a][j]))
            out[k] = tuple(point)
    return out


def smooth_boxes(boxes: Sequence[tuple]) -> np.ndarray:
    """Median filter to kill spikes, then a mean to take the shake out."""
    series = np.asarray(boxes, dtype=np.float64)
    for width, reduce in ((MEDIAN_W, np.median), (MEAN_W, np.mean)):
        r = width // 2
        padded = np.pad(series, ((r, r), (0, 0)), mode="edge")
        rows = []
        for i in range(len(series)):
            rows.append(reduce(padded[i:i + width], axis=0))
        series = np.stack(rows)
    return series


def _runmean(values: np.ndarray, width: int) -> np.ndarray:
    r = width // 2
    padded = np.pad(values, (r, r), mode="edge")
    out = []
    for i in range(len(values)):
        out.append(padded[i:i + width].mean())
    return np.array(out)


def _square(box, side: float, fw: int, fh: int):
    """A square of fixed size on the box centre, nudged to stay in frame."""
    cx, cy = _centre(box)
    s = min(side, fw, fh)                       # cannot be bigger than the frame
    x1 = min(max(cx - s / 2, 0), fw - s)
    y1 = min(max(cy - s / 2, 0), fh - s)
    return int(round(x1)), int(round(y1)), int(round(s))


def _count_changes(per_frame, boxes, fh: int, fw: int):
    """How often pass 2 overruled pass 1: dropped the pick, or swapped it."""
    rogue = 0
    swapped = 0
    for dets, box in zip(per_frame, boxes):
        leader = pick_striker(dets, (fh, fw))
        if leader is None:
            continue
        if box is None:
            rogue += 1
        elif _iou(tuple(leader.box), box) < TRACK_IOU:
            swapped += 1
    return rogue, swapped


def _best_shot(clip, boxes, cuts):
    """The stretch between camera cuts holding the most detections."""
    bounds = [0]
    if cuts is None:
        bounds.extend(find_cuts(clip))
    else:
        bounds.extend(cuts)
    bounds.append(len(clip))

    best = range(0, len(clip))
    best_found = -1
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            continue
        found = 0
        for i in range(a, b):
            if boxes[i] is not None:
                found += 1
        if found > best_found:
            best_found = found
            best = range(a, b)
    return best


def crop_clip(clip: np.ndarray, per_frame, size: int = SIZE, pad: float = PAD,
              min_frames: int = MIN_FRAMES, min_detected: float = MIN_DETECTED,
              cuts=None, segmented: bool = False, seg_mode: str = SEG_MODE,
              alpha: float = ALPHA, body_colour=BODY_TINT,
              bat_colour=BAT_TINT, side_from: str = SIDE_FROM) -> CropResult:
    """Crop a clip to the batter. `per_frame` is detect()'s output for it.

    `segmented=True` applies the segmentation layer first, which needs
    detect(masks=True). `seg_mode="overlay"` is the paper's version: tint the
    batter and bat, keep the background. `"blackout"` is the one this project
    built first by misreading them, and it is 16.86 points worse.
    """
    fh, fw = clip.shape[1:3]
    picks = track_picks(per_frame, (fh, fw))

    boxes = []
    for pick in picks:
        boxes.append(_as_box(pick))

    rogue, swapped = _count_changes(per_frame, boxes, fh, fw)
    empty = np.empty((0, size, size, 3), np.uint8)

    shot = _best_shot(clip, boxes, cuts)
    seen = []
    for i in shot:
        if boxes[i] is not None:
            seen.append(i)
    if not seen:
        return CropResult(empty, [], "no striker anywhere")

    lo, hi = seen[0], seen[-1]          # do not invent a box before the first
    density = len(seen) / (hi - lo + 1)
    if density < min_detected:
        return CropResult(empty, [], f"only {density:.0%} of the shot detected")
    span = _fill(boxes, lo, hi)
    if len(span) < min_frames:
        return CropResult(empty, [], f"shot only {len(span)} frames")

    smoothed = smooth_boxes(span)
    # size on height, not on max(w, h): width is the bat, height is the camera.
    # `side_from` exists so tools/crop_sweep.py can put that choice on trial
    # against the authors' own model rather than leaving it a conviction.
    heights = smoothed[:, 3] - smoothed[:, 1]
    if side_from == "height":
        raw = heights
    elif side_from == "max":
        widths = smoothed[:, 2] - smoothed[:, 0]
        raw = np.maximum(heights, widths)
    else:
        raise ValueError(f"unknown side_from: {side_from!r}")
    sides = _runmean(raw, SIZE_W) * (1 + 2 * pad)

    out = np.empty((len(span), size, size, 3), np.uint8)
    nomask = 0
    carried = 0
    for k, box in enumerate(smoothed):
        x, y, s = _square(box, float(sides[k]), fw, fh)
        src = clip[lo + k]
        if segmented:
            masks = part_masks(picks[lo + k], per_frame[lo + k])
            if masks[0] is None and masks[1] is None:
                masks = carried_mask(picks, per_frame, lo + k, box)
                if masks[0] is not None or masks[1] is not None:
                    carried += 1
            if masks[0] is None and masks[1] is None:
                nomask += 1
            src = segment(src, None, masks=masks, mode=seg_mode, alpha=alpha,
                          body_colour=body_colour, bat_colour=bat_colour)
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
