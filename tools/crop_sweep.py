"""Use the authors' frozen model as a ruler for our cropping.

Their trained classifier scores 34.46% on our crops where our own model scores
78.14% on the identical files. The signal is there; the framing is not theirs.
So vary one cropping parameter at a time and re-score *their* model: whichever
setting makes their score climb is the setting that matches their pipeline.

No training, no label noise, no convergence to wait for. Minutes per setting
instead of a six-hour run.

Two stages, because **detection does not depend on the crop settings**:

    stage 1   read each clip once, detect once, then emit one 15-frame array
              per setting from those same detections
    stage 2   score each setting's arrays with the authors' model

Stage 1 is the expensive one and it is paid once for the whole sweep.

The two stages need **different interpreters**: stage 1 imports torch and
ultralytics, stage 2 needs Keras 3.8.0 (the version that wrote their file,
per metadata.json). So they are run separately, with the same arguments so
the clip sampling matches:

    python -m tools.crop_sweep --clips 200 --stage 1          # project venv
    <keras38>/python -m tools.crop_sweep --clips 200 --stage 2  # scratch venv
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time

import numpy as np

SPLITS = os.path.join("data", "splits_author")
WORK = os.path.join("experiments", "crop_sweep")
KERAS = os.path.join("CricShoot10kModels",
                     "Efficientnetv2-s_GRU_128_NEEDS_CROPPED_SEGMENTED_SHOTS.keras")


class Setting(dict):
    """One point in the sweep. `name` is also its directory."""

    @property
    def name(self):
        bits = ["pad%g" % self["pad"], self["side_from"]]
        if not self["dedup"]:
            bits.append("nodedup")
        return "_".join(bits)


def settings_from(pads, sides, dedups):
    out = []
    for pad in pads:
        for side in sides:
            for dedup in dedups:
                out.append(Setting(pad=pad, side_from=side, dedup=dedup))
    return out


def balanced(rows, n, seed=0):
    by = {}
    for row in rows:
        by.setdefault(row["class"], []).append(row)
    rng = random.Random(seed)
    for name in by:
        rng.shuffle(by[name])
    out = []
    i = 0
    while len(out) < n:
        took = False
        for name in sorted(by):
            if i < len(by[name]) and len(out) < n:
                out.append(by[name][i])
                took = True
        if not took:
            break
        i += 1
    return out


def read_split(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---- stage 1: crops ------------------------------------------------------

def build_crops(rows, settings, work, root, quiet=False):
    """One read and one detection per clip; one array per setting."""
    from src.data.video import read_clip
    from src.preprocess.crop import crop_clip
    from src.preprocess.detect import detect, load_detector
    from src.preprocess.sample import drop_duplicates, sample
    from src.utils.device import get_device

    model = load_detector()
    device = str(get_device())
    for s in settings:
        os.makedirs(os.path.join(work, s.name), exist_ok=True)

    started = time.time()
    made = {s.name: 0 for s in settings}

    for n, row in enumerate(rows, 1):
        try:
            raw = read_clip(os.path.join(root, row["clip"]))
        except Exception:
            continue

        # detections are computed on the de-duplicated clip, as the pipeline
        # does; the nodedup setting re-indexes rather than detecting twice
        clip, keep = drop_duplicates(raw)
        per = detect(list(clip), model=model, device=device, masks=False)

        for s in settings:
            if s["dedup"]:
                frames, dets = clip, per
            else:
                frames, dets = raw, _spread(per, keep, len(raw))
            try:
                got = crop_clip(frames, dets, pad=s["pad"],
                                side_from=s["side_from"])
            except Exception:
                continue
            if got.frames.shape[0] == 0:
                continue
            out = os.path.join(work, s.name,
                               row["clip"].replace("/", os.sep)[:-4] + ".npy")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out, sample(got.frames, 15))
            made[s.name] += 1

        if not quiet and (n % 20 == 0 or n == len(rows)):
            gone = time.time() - started
            print("  %4d/%4d clips  %.1f min, ~%.1f min left"
                  % (n, len(rows), gone / 60, gone / n * (len(rows) - n) / 60),
                  flush=True)

    return made


def _spread(per, keep, total):
    """Detections for the de-duplicated frames, re-indexed onto the raw clip.

    A duplicate frame is by definition the same picture as the one kept
    before it, so reusing that frame's detections is exact rather than an
    approximation.
    """
    out = []
    at = -1
    for i in range(total):
        if at + 1 < len(keep) and keep[at + 1] == i:
            at += 1
        if at < 0:
            out.append([])
        else:
            out.append(per[min(at, len(per) - 1)])
    return out


# ---- stage 2: score ------------------------------------------------------

def score_dir(model, rows, folder, classes, mapping=None):
    """Returns (truth, their output units) for clips present in `folder`."""
    truth = []
    units = []
    for row in rows:
        path = os.path.join(folder,
                            row["clip"].replace("/", os.sep)[:-4] + ".npy")
        if not os.path.exists(path):
            continue
        clip = np.load(path).astype("float32")
        if clip.shape != (15, 224, 224, 3):
            continue
        units.append(int(model.predict(clip[None], verbose=0)[0].argmax()))
        truth.append(classes.index(row["class"]))
    return np.array(truth), np.array(units)


def best_mapping(truth, units, n=15):
    counts = np.zeros((n, n), dtype=int)
    for t, u in zip(truth, units):
        counts[u, t] += 1
    from scipy.optimize import linear_sum_assignment
    rows_, cols = linear_sum_assignment(-counts)
    return {int(u): int(t) for u, t in zip(rows_, cols)}


def accuracy(truth, units, mapping):
    if len(truth) == 0:
        return 0.0
    hit = 0
    for t, u in zip(truth, units):
        if mapping.get(int(u), -1) == t:
            hit += 1
    return hit / len(truth)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clips", type=int, default=200)
    ap.add_argument("--pads", default="0,0.125,0.25,0.375,0.5")
    ap.add_argument("--sides", default="height,max")
    ap.add_argument("--dedup", default="1,0")
    ap.add_argument("--stage", type=int, choices=(1, 2), required=True,
                    help="1 crops (project venv), 2 scores (keras 3.8)")
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--root")
    ap.add_argument("--keep", action="store_true",
                    help="do not delete the crops afterwards")
    args = ap.parse_args(argv)

    pads = [float(p) for p in args.pads.split(",")]
    sides = args.sides.split(",")
    dedups = [bool(int(d)) for d in args.dedup.split(",")]
    settings = settings_from(pads, sides, dedups)

    with open(os.path.join(args.splits, "classes.json"), encoding="utf-8") as fh:
        classes = json.load(fh)

    fit_rows = balanced(read_split(os.path.join(args.splits, "train.csv")),
                        max(args.clips // 2, 120))
    test_rows = balanced(read_split(os.path.join(args.splits, "test.csv")),
                         args.clips, seed=1)
    rows = fit_rows + test_rows

    print(f"{len(settings)} settings x {len(rows)} clips")
    for s in settings:
        print("   " + s.name)

    if args.stage == 1:
        root = args.root
        if root is None:
            from src.preprocess.build_dataset import ROOT as root
        print("\nstage 1: crops")
        made = build_crops(rows, settings, args.work, root)
        for s in settings:
            print("   %-24s %d clips" % (s.name, made[s.name]))

    if args.stage == 1:
        return 0

    print("\nstage 2: scoring with the authors' model")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import keras
    model = keras.saving.load_model(KERAS, compile=False)
    print(f"  loaded, keras {keras.__version__}")

    results = []
    for s in settings:
        folder = os.path.join(args.work, s.name)
        ft, fu = score_dir(model, fit_rows, folder, classes)
        if len(ft) == 0:
            print("   %-24s no crops" % s.name)
            continue
        mapping = best_mapping(ft, fu)
        tt, tu = score_dir(model, test_rows, folder, classes)
        acc = accuracy(tt, tu, mapping)
        results.append((acc, s.name, len(tt)))
        print("   %-24s test top-1 %6.2f%%  (%d clips)"
              % (s.name, 100 * acc, len(tt)))

    results.sort(reverse=True)
    print()
    print("=" * 56)
    print("  best settings by the authors' own model")
    for acc, name, n in results[:5]:
        print("    %-24s %6.2f%%" % (name, 100 * acc))
    print("=" * 56)
    print("  for reference: our current pipeline scores 34.46%")
    print("  our own model on our current crops scores 78.14%")

    out = os.path.join(args.work, "results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump([{"setting": n, "test_top1": a, "clips": c}
                   for a, n, c in results], fh, indent=1)
    print(f"\nwritten to {out}")

    if not args.keep:
        for s in settings:
            shutil.rmtree(os.path.join(args.work, s.name), ignore_errors=True)
        print("crops removed; pass --keep to retain them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
