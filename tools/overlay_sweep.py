"""Look at the segmentation overlay before rebuilding 9,880 clips with it.

The paper specifies it exactly: alpha 0.5, striker blue, bat green, striker
painted first so the bat stays visible where it overlaps. This renders that,
plus neighbouring alphas, plus the blackout variant for contrast.

    python -m tools.overlay_sweep                       # 4 clips
    python -m tools.overlay_sweep --clips 6 --alphas 0.4,0.5,0.6

Writes notes/figures/overlay_sweep.png -- one row per clip.

The frames are **RGB** (read_clip(rgb=True)) and cv2.imwrite wants BGR, so
they are converted on the way out. Without that the preview lies about every
colour in it, which it did until this was noticed.
"""
import argparse
import os
import random
import sys

import cv2
import numpy as np

from src.data.video import read_clip
from src.preprocess.build_dataset import ROOT
from src.preprocess.crop import (ALPHA, BAT_TINT, BODY_TINT, blackout,
                                 part_masks, overlay)
from src.preprocess.detect import detect, load_detector
from src.preprocess.striker import pick_striker
from src.utils.device import get_device

OUT = os.path.join("notes", "figures", "overlay_sweep.png")
PANEL = 260
LABEL = 26


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
    """The frame whose striker has the largest mask, by the pipeline's own
    centrality rule -- not whichever Striker detection happens to come last."""
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


def _tile(frame, masks, label, alpha=None):
    if alpha is None:
        body_mask, bat_mask = masks
        merged = body_mask
        if bat_mask is not None:
            if merged is None:
                merged = bat_mask
            else:
                merged = merged | bat_mask
        shown = blackout(frame, None, mask=merged)
    else:
        shown = overlay(frame, masks=masks, alpha=alpha)

    small = cv2.resize(shown, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    small = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)      # cv2 writes BGR
    strip = np.full((LABEL, PANEL, 3), 252, np.uint8)
    cv2.putText(strip, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (11, 11, 11), 1, cv2.LINE_AA)
    return np.vstack([strip, small])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--alphas", default="0.3,0.4,0.5,0.6")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    alphas = []
    for piece in args.alphas.split(","):
        alphas.append(float(piece))

    device = get_device()
    model = load_detector()
    print(f"device={device}  striker={BODY_TINT} bat={BAT_TINT} (RGB)  "
          f"paper alpha={ALPHA}", flush=True)

    rows = []
    for path in _clip_paths(args.root, args.clips, args.seed):
        clip = read_clip(path)
        per_frame = detect(list(clip), model=model, device=device, masks=True)
        i = _best_frame(clip, per_frame)
        striker = pick_striker(per_frame[i], clip.shape[1:3])
        masks = part_masks(striker, per_frame[i])
        if masks[0] is None and masks[1] is None:
            print(f"  no mask: {os.path.basename(path)}", flush=True)
            continue

        has_bat = masks[1] is not None
        tiles = [_tile(clip[i], masks, "original", alpha=0.0)]
        for a in alphas:
            mark = f"alpha {a:g}"
            if a == ALPHA:
                mark += "  <- paper"
            tiles.append(_tile(clip[i], masks, mark, alpha=a))
        tiles.append(_tile(clip[i], masks, "blackout (old, -16.9)"))
        rows.append(np.hstack(tiles))
        print(f"  {os.path.basename(path)}  frame {i}  "
              f"bat={'yes' if has_bat else 'NO'}", flush=True)

    if not rows:
        raise SystemExit("nothing had a mask; try --clips higher")

    sheet = np.vstack(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    cv2.imwrite(args.out, sheet)
    print(f"\nwrote {args.out}  ({sheet.shape[1]}x{sheet.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
