"""Bat and ball tracks for Family W (Day 30), from the authors' detectors.

    bat    on the box crops, so its coordinates are body-relative for free.
           Per frame: found, centre x y, size w h (all 0..1 of the crop), the
           angle of the bat's main axis from the segmentation model's mask
           as (cos 2a, sin 2a), a mask-found flag, and the centre's step
           from the frame before (dx, dy)
    ball   on every raw frame, because the ball is tiny and mostly outside
           the crop. Per frame: found, x y (0..1 of the raw frame), dx dy

The box clips did not record which raw frames they kept, so the ball track
is resampled to the box clip's 15 time points: each point takes the
nearest raw frame, within half a step, that has a ball. Approximate in
time against the bat and pose, and said so.

A missing detection is a 0 in its found column with zeros beside it, never
a guess. An axis has no direction, so the angle is doubled: a bat at 10
and at 190 degrees is the same bat, and (cos 2a, sin 2a) says so without a
jump at 180.

**Mirroring is exact:** x -> 1 - x, dx -> -dx, sin 2a -> -sin 2a.

Measured on 30 clips (2026-10-30) before building: the bat box is found in
81% of frames at 224 px (66% at 640), the bat mask in 75%, the ball in 40%
at 640 px (37% at the plan's 896). The ball's 40% caps W03.

    python -m src.preprocess.tracks             builds data/features/tracks/
    python -m src.preprocess.tracks --coverage  found rates, per class
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from src.utils import progress
from src.utils.checkpoint import append_result, atomic_save, read_results

__all__ = ["BAT_COLUMNS", "BALL_COLUMNS", "bat_row", "bat_track",
           "ball_track", "mirror_bat", "mirror_ball", "axis", "build",
           "coverage", "TRACKS_DIR"]

TRACKS_DIR = os.path.join("data", "features", "tracks")
MODELS = "CricShoot10kModels"
BAT_MODEL = os.path.join(MODELS, "Bat_Detection_Model.pt")
BALL_MODEL = os.path.join(MODELS, "Ball_Detection_Model.pt")
SEG_MODEL = os.path.join(MODELS, "Striker_Bat_Segmentation_Model.pt")
BAT_IMGSZ = 224         # the crops' own size: 81% found, against 66% at 640
BALL_IMGSZ = 640        # its training size: 40% found, against 37% at 896

BAT_COLUMNS = ("found", "x", "y", "w", "h", "angle_cos", "angle_sin",
               "mask_found", "dx", "dy")
BALL_COLUMNS = ("found", "x", "y", "dx", "dy")
MIN_PIXELS = 10         # fewer mask pixels than this has no axis to speak of


def axis(mask):
    """(cos 2a, sin 2a) of a mask's main axis, or None. From the spread of
    its pixels: the direction the bat is longest in."""
    ys, xs = np.nonzero(mask)
    if len(xs) < MIN_PIXELS:
        return None
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    spread = float((xs * xs).mean() - (ys * ys).mean())
    shared = float(2.0 * (xs * ys).mean())
    length = float(np.hypot(spread, shared))
    if length < 1e-9:
        return None                       # round: no main axis
    return spread / length, shared / length


def _best(found, cls=None):
    best = None
    for det in found:
        if cls is not None and det.cls != cls:
            continue
        if best is None or det.conf > best.conf:
            best = det
    return best


def bat_row(box_det, mask_det, size):
    """One frame's bat, without its step: (found, x, y, w, h, cos, sin,
    mask_found) and zeros for dx, dy, filled in by bat_track."""
    row = np.zeros(len(BAT_COLUMNS), dtype=np.float32)
    if box_det is not None:
        x1, y1, x2, y2 = box_det.box
        row[0] = 1.0
        row[1] = (x1 + x2) / 2.0 / size
        row[2] = (y1 + y2) / 2.0 / size
        row[3] = (x2 - x1) / size
        row[4] = (y2 - y1) / size
    if mask_det is not None and mask_det.mask is not None:
        found = axis(mask_det.mask)
        if found is not None:
            row[5], row[6] = found
            row[7] = 1.0
    return row


def _steps(track, found_col, x_col, y_col, dx_col, dy_col):
    """dx, dy from the frame before, where both frames have a detection."""
    for t in range(1, len(track)):
        if track[t, found_col] and track[t - 1, found_col]:
            track[t, dx_col] = track[t, x_col] - track[t - 1, x_col]
            track[t, dy_col] = track[t, y_col] - track[t - 1, y_col]
    return track


def bat_track(box_found, seg_found, size):
    """(T, 10) from each frame's bat detections and segmentation results."""
    rows = []
    for boxes, segs in zip(box_found, seg_found):
        rows.append(bat_row(_best(boxes), _best(segs, "Bat"), size))
    return _steps(np.stack(rows), 0, 1, 2, 8, 9)


def ball_track(raw_found, shape, points):
    """(points, 5): the ball over the whole raw clip, resampled. Each point
    takes the nearest raw frame within half a step that has a ball."""
    height, width = shape
    count = len(raw_found)
    track = np.zeros((points, len(BALL_COLUMNS)), dtype=np.float32)
    if count == 0:
        return track
    centres = np.linspace(0, count - 1, points)
    half = (count - 1) / max(points - 1, 1) / 2.0
    for p, centre in enumerate(centres):
        best = None
        best_gap = None
        for i in range(count):
            gap = abs(i - centre)
            if gap > half + 1e-9:
                continue
            ball = _best(raw_found[i])
            if ball is None:
                continue
            if best is None or gap < best_gap:
                best = ball
                best_gap = gap
        if best is not None:
            x, y = best.centre
            track[p, 0] = 1.0
            track[p, 1] = x / width
            track[p, 2] = y / height
    return _steps(track, 0, 1, 2, 3, 4)


def _copy(track):
    if isinstance(track, torch.Tensor):
        return track.clone()
    return track.copy()


def _flip_x(track, col):
    """x -> 1 - x where something was found; a missing frame stays zeros."""
    found = track[..., 0] > 0
    track[..., col] = (1.0 - track[..., col]) * found


def mirror_bat(track):
    """The same bat, flipped left to right."""
    out = _copy(track)
    _flip_x(out, 1)
    out[..., 6] = -out[..., 6]            # sin 2a
    out[..., 8] = -out[..., 8]            # dx
    return out


def mirror_ball(track):
    """The same ball, flipped left to right."""
    out = _copy(track)
    _flip_x(out, 1)
    out[..., 3] = -out[..., 3]            # dx
    return out


def _manifest(out_dir):
    return os.path.join(out_dir, "manifest.jsonl")


def _finished(manifest):
    done = set()
    for row in read_results(manifest):
        if row.get("ok"):
            done.add(row["clip"])
    return done


def _save(array, out_dir, leaf, rel):
    path = os.path.join(out_dir, leaf, rel.replace("/", os.sep) + ".pt")
    atomic_save(torch.as_tensor(array), path)


def build(processed, raw_root, out_dir=TRACKS_DIR, pose_dir=None, limit=None,
          device=None, quiet=False, ledger=None):
    """Bat and ball tracks for every box clip, and bat + ball + pose joined
    per frame, each with its mirror. Resumable. Returns failures."""
    from src.data.video import read_clip
    from src.extract import clip_list
    from src.preprocess.detect import detect, load_detector
    from src.preprocess.pose import POSE_DIR

    if pose_dir is None:
        pose_dir = POSE_DIR
    bat_model = load_detector(BAT_MODEL)
    seg_model = load_detector(SEG_MODEL)
    ball_model = load_detector(BALL_MODEL)
    root = os.path.join(processed, "box")
    clips = clip_list(root)
    if limit:
        clips = clips[:limit]
    manifest = _manifest(out_dir)
    done = _finished(manifest)
    todo = []
    for rel in clips:
        if rel not in done:
            todo.append(rel)
    if not quiet:
        print(f"tracks: {len(clips)} clips, {len(done)} done, {len(todo)} to go",
              flush=True)

    tty = sys.stdout.isatty() and not quiet
    started = time.time()
    failed = 0
    for n, rel in enumerate(todo, 1):
        stem = rel.replace("/", os.sep)
        try:
            frames = list(np.load(os.path.join(root, stem + ".npy")))
            size = frames[0].shape[1]
            bat = bat_track(
                detect(frames, model=bat_model, device=device,
                       imgsz=BAT_IMGSZ, batch=16),
                detect(frames, model=seg_model, device=device,
                       imgsz=BAT_IMGSZ, batch=16, masks=True), size)
            raw = read_clip(os.path.join(raw_root, stem + ".avi"))
            raw_found = detect(list(raw), model=ball_model, device=device,
                               imgsz=BALL_IMGSZ, batch=8)
            ball = ball_track(raw_found, raw.shape[1:3], len(frames))
            pose = torch.load(os.path.join(pose_dir, "pose", stem + ".pt"))
            pose_mirror = torch.load(os.path.join(pose_dir, "pose_mirror",
                                                  stem + ".pt"))
            if len(pose) != len(frames):
                raise ValueError(f"{len(pose)} pose frames, {len(frames)} box")
            joined = np.concatenate([bat, ball, pose.flatten(1).numpy()], 1)
            joined_mirror = np.concatenate(
                [mirror_bat(bat), mirror_ball(ball),
                 pose_mirror.flatten(1).numpy()], 1)
            _save(bat, out_dir, "bat", rel)
            _save(mirror_bat(bat), out_dir, "bat_mirror", rel)
            _save(ball, out_dir, "ball", rel)
            _save(mirror_ball(ball), out_dir, "ball_mirror", rel)
            _save(joined, out_dir, "bat_ball_pose", rel)
            _save(joined_mirror, out_dir, "bat_ball_pose_mirror", rel)
            raw_hits = 0
            for found in raw_found:
                if found:
                    raw_hits += 1
            row = {"clip": rel, "ok": True, "class": rel.split("/")[0],
                   "frames": len(frames), "bat": int(bat[:, 0].sum()),
                   "bat_mask": int(bat[:, 7].sum()),
                   "ball": int(ball[:, 0].sum()), "raw_frames": len(raw),
                   "raw_ball": raw_hits}
        except (OSError, RuntimeError, ValueError) as err:
            row = {"clip": rel, "ok": False,
                   "reason": f"{type(err).__name__}: {err}"}
            failed += 1
        append_result(row, manifest)
        if tty:
            rate = n / max(time.time() - started, 1e-9)
            left = (len(todo) - n) / max(rate, 1e-9) / 60
            print(progress.line(n, len(todo),
                                f"{rate:5.2f} clips/s  ~{left:5.1f} min left",
                                prefix="  ", tty=True), end="", flush=True)
    if todo and not quiet:
        print(f"\n{failed} failed")
    if todo and ledger:
        secs = time.time() - started
        append_result({"kind": "tracks", "cache": out_dir, "clips": len(todo),
                       "failed": failed, "minutes": round(secs / 60, 2)},
                      ledger)
    return failed


def coverage(out_dir=TRACKS_DIR):
    """{class: {"bat", "bat_mask", "ball": found, "frames": all}}."""
    out = {}
    for row in read_results(_manifest(out_dir)):
        if not row.get("ok"):
            continue
        cls = row["class"]
        if cls not in out:
            out[cls] = {"bat": 0, "bat_mask": 0, "ball": 0, "frames": 0}
        for key in ("bat", "bat_mask", "ball", "frames"):
            out[cls][key] += row[key]
    return out


def main(argv=None):
    from src.extract import CACHE_LEDGER

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coverage", action="store_true",
                    help="how often the bat, its mask and the ball were found")
    ap.add_argument("--processed", default=os.path.join("data", "processed"))
    ap.add_argument("--raw", default=os.path.join("data",
                                                  "CricShoot10kShootDataset"))
    ap.add_argument("--out", default=TRACKS_DIR)
    ap.add_argument("--limit", type=int, help="first N clips, for a smoke test")
    args = ap.parse_args(argv)
    if args.coverage:
        for cls, got in sorted(coverage(args.out).items()):
            frames = got["frames"]
            print(f"{cls:20s} bat {100 * got['bat'] / frames:5.1f}%  "
                  f"mask {100 * got['bat_mask'] / frames:5.1f}%  "
                  f"ball {100 * got['ball'] / frames:5.1f}%")
        return 0
    return build(args.processed, args.raw, args.out, limit=args.limit,
                 ledger=CACHE_LEDGER)


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
