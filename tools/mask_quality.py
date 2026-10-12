"""How much of the batter does the segmentation mask actually cover?

The overlay sweep showed silhouettes stopping around the pad line on several
clips. That is an impression from six pictures. This turns it into a number,
because the answer decides an order of work: if the masks are badly
incomplete, fine-tuning YOLO comes *before* the 4.5-hour rebuild, since the
rebuild bakes the masks into 9,880 .npy files.

    python -m tools.mask_quality                  # 120 clips, across classes
    python -m tools.mask_quality --clips 300

Reports coverage of the striker's box, a top-to-bottom profile of that
coverage, and how often the bat is masked at all. Writes a figure next to the
numbers.
"""
import argparse
import csv
import os
import random
import sys

import numpy as np

from src.data.video import read_clip
from src.preprocess.build_dataset import ROOT
from src.preprocess.crop import BAT_REACH, _centre
from src.preprocess.detect import detect, load_detector
from src.preprocess.sample import drop_duplicates
from src.preprocess.striker import pick_striker
from src.utils import progress
from src.utils.device import get_device

SPLIT = os.path.join("data", "splits_author", "train.csv")
OUT = os.path.join("notes", "figures", "mask_quality.png")
BANDS = 10          # vertical slices of the striker's box, head to feet


def _clips_across_classes(split_csv, n, seed):
    """Sampled per class. A class-ordered head() would measure one class."""
    by_class = {}
    with open(split_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_class.setdefault(row["class"], []).append(row["clip"])

    rng = random.Random(seed)
    for name in by_class:
        rng.shuffle(by_class[name])

    picked = []
    names = sorted(by_class)
    i = 0
    while len(picked) < n:
        took = False
        for name in names:
            if i < len(by_class[name]) and len(picked) < n:
                picked.append(by_class[name][i])
                took = True
        if not took:
            break
        i += 1
    return picked


def _box_pixels(box, shape):
    h, w = shape
    x1 = max(0, int(round(box[0])))
    y1 = max(0, int(round(box[1])))
    x2 = min(w, int(round(box[2])))
    y2 = min(h, int(round(box[3])))
    return x1, y1, x2, y2


def measure_frame(dets, shape):
    """Coverage and a vertical profile for one frame, or None."""
    striker = pick_striker(dets, shape)
    if striker is None or striker.mask is None:
        return None

    x1, y1, x2, y2 = _box_pixels(striker.box, shape)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    inside = striker.mask.astype(bool)[y1:y2, x1:x2]
    area = inside.size
    if area == 0:
        return None

    bands = []
    height = y2 - y1
    for k in range(BANDS):
        a = int(round(k * height / BANDS))
        b = int(round((k + 1) * height / BANDS))
        if b <= a:
            bands.append(0.0)
        else:
            bands.append(float(inside[a:b].mean()))

    rows = inside.any(axis=1)
    if rows.any():
        lowest = int(np.nonzero(rows)[0].max())
        reach = (lowest + 1) / height      # 1.0 means the mask reaches the feet
    else:
        reach = 0.0

    bat = False
    cx, cy = _centre(striker.box)
    body = striker.box[3] - striker.box[1]
    if body == 0:
        body = 1.0
    for det in dets:
        if det.cls != "Bat" or det.mask is None:
            continue
        bx, by = _centre(det.box)
        if ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5 / body <= BAT_REACH:
            bat = True

    return {"coverage": float(inside.mean()), "bands": bands,
            "reach": reach, "bat": bat}


def _pct(values, q):
    return float(np.percentile(values, q))


def report(rows, frames_seen, no_striker, no_mask):
    cover = []
    reach = []
    bands = []
    bat = 0
    for r in rows:
        cover.append(r["coverage"])
        reach.append(r["reach"])
        bands.append(r["bands"])
        if r["bat"]:
            bat += 1
    bands = np.array(bands)

    print()
    print(f"frames with a striker and a mask   {len(rows)} of {frames_seen}")
    print(f"  no striker picked                {no_striker}")
    print(f"  striker picked but no mask       {no_mask}")
    print()
    print("mask coverage of the striker's box")
    print("  p10 %5.1f%%   median %5.1f%%   p90 %5.1f%%"
          % (100 * _pct(cover, 10), 100 * _pct(cover, 50), 100 * _pct(cover, 90)))
    print()
    print("how far down the box the mask reaches (1.00 = to the feet)")
    print("  p10 %5.2f    median %5.2f    p90 %5.2f"
          % (_pct(reach, 10), _pct(reach, 50), _pct(reach, 90)))
    short = 0
    for value in reach:
        if value < 0.9:
            short += 1
    print("  stops above 90%% of the box in %d of %d frames (%.1f%%)"
          % (short, len(reach), 100 * short / max(len(reach), 1)))
    print()
    print("coverage by vertical band, head (1) to feet (%d)" % BANDS)
    for k in range(BANDS):
        mean = float(bands[:, k].mean())
        bar = chr(0x2588) * int(round(40 * mean))
        print("  %2d  %5.1f%%  %s" % (k + 1, 100 * mean, bar))
    print()
    print("bat masked within reach of the striker   %d of %d (%.1f%%)"
          % (bat, len(rows), 100 * bat / max(len(rows), 1)))
    return bands, reach, bat / max(len(rows), 1)


def draw(bands, reach, bat_rate, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.plot import THEMES

    ink = THEMES["light"]
    blue = ink["series"][0]
    mean = bands.mean(axis=0)
    lo = np.percentile(bands, 25, axis=0)
    hi = np.percentile(bands, 75, axis=0)
    y = np.arange(1, BANDS + 1)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    fig.patch.set_facecolor(ink["surface"])
    ax.set_facecolor(ink["surface"])
    ax.fill_betweenx(y, 100 * lo, 100 * hi, color=blue, alpha=0.18, linewidth=0)
    ax.plot(100 * mean, y, color=blue, linewidth=1.8, marker="o",
            markersize=5, markeredgecolor=ink["surface"], markeredgewidth=1.2)

    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_yticks(y)
    ax.set_ylabel("band of the striker's box, head to feet",
                  color=ink["secondary"], fontsize=9)
    ax.set_xlabel("mask coverage (%)", color=ink["secondary"], fontsize=9)
    ax.grid(True, color=ink["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(ink["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=ink["muted"], labelsize=8)

    ax.annotate(f"{100 * mean[-1]:.0f}%", xy=(100 * mean[-1], BANDS),
                xytext=(12, 0), textcoords="offset points", va="center",
                color=ink["primary"], fontsize=9)

    fig.text(0.06, 0.975, "Segmentation mask coverage, top to bottom",
             color=ink["primary"], fontsize=12, ha="left", va="top")

    reaches = 0
    for value in reach:
        if value >= 0.9:
            reaches += 1
    lines = [
        "%d frames. The mask reaches the bottom of the box in %.0f%% of them,"
        % (bands.shape[0], 100 * reaches / max(len(reach), 1)),
        "so the fall towards the feet is mostly legs being narrower than a"
        " torso, not mask failure.",
        "The real gap is the bat: masked in only %.0f%% of frames."
        % (100 * bat_rate),
    ]
    note = "\n".join(lines)
    fig.text(0.06, 0.935, note, color=ink["muted"], fontsize=8.5,
             ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(left=0.13, right=0.95, top=0.80, bottom=0.10)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=ink["surface"])
    plt.close(fig)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", type=int, default=120)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    device = get_device()
    model = load_detector()
    clips = _clips_across_classes(args.split, args.clips, args.seed)
    print(f"device={device}  {len(clips)} clips across classes", flush=True)

    rows = []
    frames_seen = 0
    no_striker = 0
    no_mask = 0
    unreadable = 0

    for n, rel in enumerate(clips):
        try:
            clip = read_clip(os.path.join(args.root, rel))
            clip, _ = drop_duplicates(clip)
            per = detect(list(clip), model=model, device=device, masks=True)
        except Exception:
            unreadable += 1
            continue

        shape = clip.shape[1:3]
        for dets in per:
            frames_seen += 1
            got = measure_frame(dets, shape)
            if got is None:
                striker = pick_striker(dets, shape)
                if striker is None:
                    no_striker += 1
                else:
                    no_mask += 1
            else:
                rows.append(got)

        if progress.is_tty():
            sys.stdout.write(progress.line(n + 1, len(clips),
                                           f"{len(rows)} frames measured"))
            sys.stdout.flush()

    if progress.is_tty():
        sys.stdout.write(chr(13) + " " * 90 + chr(13))
    if unreadable:
        print(f"{unreadable} clips could not be read")
    if not rows:
        raise SystemExit("no frame had a striker with a mask")

    bands, reach, bat_rate = report(rows, frames_seen, no_striker,
                                    no_mask)
    draw(bands, reach, bat_rate, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
