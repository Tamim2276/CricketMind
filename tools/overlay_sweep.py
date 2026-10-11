"""Render the segmentation overlay at several alphas, side by side.

The paper says "a semi-transparent overlay is applied to the batter and bat"
and states neither the weight nor the colour. So: look at it before spending
4.5 hours rebuilding 9,880 clips with a guess.

    python -m tools.overlay_sweep                       # 4 clips, default tint
    python -m tools.overlay_sweep --clips 6 --tint 0,170,0
    python -m tools.overlay_sweep --alphas 0.2,0.3,0.4,0.5,0.6

Writes notes/figures/overlay_sweep.png -- one row per clip, one column per
alpha, with blackout on the end for comparison.
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np

from src.data.video import read_clip
from src.preprocess.crop import TINT, blackout, frame_mask, overlay
from src.preprocess.detect import detect, load_detector
from src.preprocess.striker import pick_striker
from src.utils.device import get_device

OUT = os.path.join("notes", "figures", "overlay_sweep.png")
ROOT = os.path.join("data", "CricShoot10kShootDataset")
PANEL = 260          # each tile, square
LABEL = 26           # strip above each row of tiles


def _clip_paths(root, n, seed):
    found = []
    for folder, _, files in os.walk(root):
        for name in files:
            if name.lower().endswith((".avi", ".mp4")):
                found.append(os.path.join(folder, name))
    if not found:
        raise SystemExit(f"no clips under {root}")
    found.sort()
    random.Random(seed).shuffle(found)
    return found[:n]


def _best_frame(clip, per_frame):
    """The frame whose striker has the largest mask -- worth looking at.

    The striker is chosen by `pick_striker`, the same centrality rule the
    real pipeline uses. Taking "whichever Striker detection came last" would
    tint the non-striker or an umpire and make the sweep a lie.
    """
    shape = clip.shape[1:3]
    best = 0
    area = -1
    for i, dets in enumerate(per_frame):
        chosen = pick_striker(dets, shape)
        if chosen is None or chosen.mask is None:
            continue
        got = int(chosen.mask.sum())
        if got > area:
            area = got
            best = i
    return best


def _tile(frame, mask, label, alpha=None, tint=TINT):
    if alpha is None:
        shown = blackout(frame, None, mask=mask)
    else:
        shown = overlay(frame, None, mask=mask, alpha=alpha, colour=tint)
    small = cv2.resize(shown, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    strip = np.full((LABEL, PANEL, 3), 252, np.uint8)
    cv2.putText(strip, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (11, 11, 11), 1, cv2.LINE_AA)
    return np.vstack([strip, small])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--alphas", default="0.2,0.3,0.4,0.5,0.6")
    ap.add_argument("--tint", default=",".join(str(c) for c in TINT),
                    help="BGR, the order cv2 uses")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    alphas = []
    for piece in args.alphas.split(","):
        alphas.append(float(piece))
    tint = []
    for piece in args.tint.split(","):
        tint.append(int(piece))
    tint = tuple(tint)

    device = get_device()
    model = load_detector()
    print(f"device={device}  tint(BGR)={tint}  alphas={alphas}", flush=True)

    rows = []
    for path in _clip_paths(args.root, args.clips, args.seed):
        clip = read_clip(path)
        per_frame = detect(list(clip), model=model, device=device,
                           masks=True)
        i = _best_frame(clip, per_frame)
        striker = pick_striker(per_frame[i], clip.shape[1:3])
        mask = frame_mask(striker, per_frame[i])
        if mask is None:
            print(f"  no mask: {os.path.basename(path)}", flush=True)
            continue

        tiles = [_tile(clip[i], mask, "original", alpha=0.0)]
        for a in alphas:
            tiles.append(_tile(clip[i], mask, f"alpha {a:g}", alpha=a,
                               tint=tint))
        tiles.append(_tile(clip[i], mask, "blackout (ours, -16.9)"))
        rows.append(np.hstack(tiles))
        print(f"  {os.path.basename(path)}  frame {i}", flush=True)

    if not rows:
        raise SystemExit("nothing had a mask; try --clips higher")

    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, sheet)
    print(f"\nwrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
