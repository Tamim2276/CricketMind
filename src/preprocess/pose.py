"""Pose for Family E (Day 28): 33 body points a frame, from the box crops.

The box crops already follow the striker (Day 3), so the pose model never
has to choose between players. Per frame MediaPipe gives 33 landmarks, and
we keep (x, y, visibility):

    raw       (33, 3) a frame   x, y in 0..1 of the crop; None: no pose found
    cached    (T, 33, 3)        centred on the clip's mean hip point and
                                scaled by its median torso length. A frame
                                with no pose is all zeros, visibility too,
                                which no pose that was found has

Centred per clip, not per frame: per frame would erase the hips moving
inside the crop, and stepping forward or back is half of a stroke.

**Mirroring is exact:** negate x and swap left and right joints. A pose is
geometry, unlike a CNN feature map (8.7: flipped cached features matched at
cosine 0.41), so the mirror half needs no second detector pass.

    python -m src.extract --pose              writes data/features/pose/
    python -m src.preprocess.pose --coverage  frames with a pose, per class
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from src.utils import progress
from src.utils.checkpoint import append_result, atomic_save, read_results

__all__ = ["JOINTS", "CHANNELS", "LEFT_RIGHT", "EDGES", "adjacency",
           "normalise", "mirror", "build", "coverage", "POSE_DIR"]

JOINTS = 33
CHANNELS = 3                     # x, y, visibility
POSE_DIR = os.path.join("data", "features", "pose")

# MediaPipe's numbering: 0 nose, 1-6 eyes, 7-8 ears, 9-10 mouth,
# 11-12 shoulders, 13-14 elbows, 15-16 wrists, 17-22 hands, 23-24 hips,
# 25-26 knees, 27-28 ankles, 29-30 heels, 31-32 toes. Left first in each pair
LEFT_RIGHT = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
              (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
              (27, 28), (29, 30), (31, 32))

# MediaPipe's own drawing edges, plus nose-shoulders and mouth-nose: without
# them the head is a separate graph and nothing flows from it to the body
EDGES = ((0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
         (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
         (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
         (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
         (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
         (0, 11), (0, 12), (9, 0), (10, 0))

LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12


def adjacency():
    """(33, 33): 1 along every bone, both ways, and 1 on the diagonal so a
    joint keeps its own value when it takes in its neighbours'."""
    a = torch.eye(JOINTS)
    for i, j in EDGES:
        a[i, j] = 1.0
        a[j, i] = 1.0
    return a


def _midpoint(frame, i, j):
    return (frame[i, :2] + frame[j, :2]) / 2.0


def normalise(frames):
    """A clip's raw poses -> (T, 33, 3) float32, as the cache holds them."""
    out = np.zeros((len(frames), JOINTS, CHANNELS), dtype=np.float32)
    hips = []
    torsos = []
    for frame in frames:
        if frame is None:
            continue
        hip = _midpoint(frame, LEFT_HIP, RIGHT_HIP)
        shoulder = _midpoint(frame, LEFT_SHOULDER, RIGHT_SHOULDER)
        hips.append(hip)
        torsos.append(float(np.linalg.norm(shoulder - hip)))
    if not hips:
        return out                          # no pose anywhere: all missing
    centre = np.mean(np.stack(hips), axis=0)
    scale = float(np.median(torsos))
    if scale < 1e-6:
        scale = 1.0
    for t, frame in enumerate(frames):
        if frame is None:
            continue
        out[t, :, :2] = (frame[:, :2] - centre) / scale
        out[t, :, 2] = frame[:, 2]
    return out


def mirror(pose):
    """The same clip flipped left to right: x negated, left and right
    joints swapped. Missing frames stay all zeros."""
    if isinstance(pose, torch.Tensor):
        flipped = pose.clone()
    else:
        flipped = pose.copy()
    for left, right in LEFT_RIGHT:
        flipped[..., left, :] = pose[..., right, :]
        flipped[..., right, :] = pose[..., left, :]
    flipped[..., 0] = -flipped[..., 0]
    return flipped


def _manifest(out_dir):
    return os.path.join(out_dir, "manifest.jsonl")


def _finished(manifest):
    done = set()
    for row in read_results(manifest):
        if row.get("ok"):
            done.add(row["clip"])
    return done


def _save(pose, out_dir, view, rel):
    path = os.path.join(out_dir, view, rel.replace("/", os.sep) + ".pt")
    atomic_save(torch.from_numpy(pose), path)


def build(detector, processed, out_dir=POSE_DIR, limit=None, quiet=False,
          ledger=None):
    """Pose for every box clip, both halves. Resumable. Returns failures.

    `detector(frame)` takes one (H, W, 3) uint8 RGB frame and returns its
    (33, 3) raw pose or None.
    """
    from src.extract import clip_list

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
        print(f"pose: {len(clips)} clips, {len(done)} done, {len(todo)} to go",
              flush=True)

    tty = sys.stdout.isatty() and not quiet
    started = time.time()
    failed = 0
    for n, rel in enumerate(todo, 1):
        try:
            frames = np.load(os.path.join(root, rel.replace("/", os.sep)
                                          + ".npy"))
            raw = []
            found = 0
            for frame in frames:
                pose = detector(frame)
                raw.append(pose)
                if pose is not None:
                    found += 1
            cached = normalise(raw)
            _save(cached, out_dir, "pose", rel)
            _save(mirror(cached), out_dir, "pose_mirror", rel)
            row = {"clip": rel, "ok": True, "frames": len(raw),
                   "found": found, "class": rel.split("/")[0]}
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
        append_result({"kind": "pose", "cache": out_dir, "clips": len(todo),
                       "failed": failed, "minutes": round(secs / 60, 2)},
                      ledger)
    return failed


def coverage(out_dir=POSE_DIR):
    """{class: (frames with a pose, frames)}, from the manifest."""
    found = {}
    for row in read_results(_manifest(out_dir)):
        if not row.get("ok"):
            continue
        cls = row["class"]
        if cls not in found:
            found[cls] = [0, 0]
        found[cls][0] += row["found"]
        found[cls][1] += row["frames"]
    out = {}
    for cls in sorted(found):
        out[cls] = (found[cls][0], found[cls][1])
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coverage", action="store_true",
                    help="frames with a pose, per class")
    ap.add_argument("--out", default=POSE_DIR)
    args = ap.parse_args(argv)
    if args.coverage:
        total_found = 0
        total = 0
        for cls, (got, frames) in coverage(args.out).items():
            total_found += got
            total += frames
            print(f"{cls:20s} {got:6d} / {frames:6d}  {100 * got / frames:5.1f}%")
        if total:
            print(f"{'all':20s} {total_found:6d} / {total:6d}  "
                  f"{100 * total_found / total:5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
