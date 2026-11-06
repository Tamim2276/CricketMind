"""G03: show a model each clip a few ways and average its answers (Day 41).

    python -m tools.tta                 what is already scored; no GPU
    python -m tools.tta --score         first scores what is missing, on val
    python -m tools.tta --exp C06_mvitv2_s T05_c06_ls005

The plan asks for a flip and two temporal crops. Every stored clip has exactly
15 frames, so there are none spare to crop from. "early" and "late" slide the
clip by SHIFT frames and repeat the edge frame, which is as close as it gets
without a rebuild. Whether they help is for val to say, so each set is scored.

Nothing retrains. Each view is one full pass over val, about four minutes per
run on the GPU, so use --score when nothing is training. Test stays closed
(rule 5).
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CricShotDataset
from src.evaluate import _restore, run_eval
from src.experiments import registry as R
from src.train import _ticker
from src.utils.checkpoint import RESULTS_PATH, read_results
from src.utils.device import get_device
from tools.tuning import SPLIT, _ledger, base_names, clear_line

__all__ = ["VIEWS", "VIEW_SETS", "View", "view_probs", "tta_scores", "averaged",
           "tta_part", "top1", "rows_for", "render", "main"]

VIEWS = ("plain", "flip", "early", "late")
VIEW_SETS = {"flip": ("plain", "flip"),
             "time": ("plain", "early", "late"),
             "all": VIEWS}
EXPS = ["C06_mvitv2_s", "F02_multiscale_tf"]
SHIFT = 2
CACHE = "val_tta.npz"


def _slid(count, view):
    """Frame numbers for a clip slid by SHIFT, its edge frame repeated. The
    count stays the same, as F02's position table stops at 15."""
    order = []
    for i in range(count):
        if view == "early":
            order.append(min(i, count - 1 - SHIFT))
        else:
            order.append(max(i, SHIFT))
    return order


class View:
    """One way of showing a clip, (T, C, H, W). Fits the dataset's transform."""

    def __init__(self, name):
        if name not in VIEWS:
            raise ValueError(f"unknown view {name!r}; have {VIEWS}")
        self.name = name

    def __call__(self, clip):
        if self.name == "plain":
            return clip
        if self.name == "flip":
            return torch.flip(clip, dims=[-1])
        return clip[_slid(clip.shape[0], self.name)]

    def __repr__(self):
        return f"View({self.name!r})"


def view_probs(ckpt, split_csv, batch_size=4, device=None):
    """Per clip, the model's probabilities under every view."""
    device = device or get_device()
    _, data, model, _ = _restore(ckpt, split_csv, device=device)
    if not isinstance(data, CricShotDataset):
        raise SystemExit(f"{ckpt}: this needs clips, not stored features")
    clips = []
    for item in data.items:
        clips.append(item[2])
    run = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
    views = {}
    labels = None
    for name in VIEWS:
        data.transform = View(name)
        loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                            num_workers=0)
        stats = run_eval(model, loader, device=device, collect=True,
                         on_batch=_ticker(f"  {run} {name:<5} ", False))
        clear_line()
        views[name] = torch.softmax(torch.from_numpy(stats.logits), dim=1).numpy()
        labels = stats.labels
    return {"clips": clips, "labels": labels, "classes": list(data.classes),
            "views": views}


def _save(found, cache):
    arrays = {"clips": np.asarray(found["clips"]),
              "labels": np.asarray(found["labels"]),
              "classes": np.asarray(found["classes"])}
    for name in VIEWS:
        arrays[f"probs_{name}"] = found["views"][name]
    # a power cut must not leave half a file that looks finished
    temp = cache[:-4] + ".tmp.npz"
    np.savez(temp, **arrays)
    os.replace(temp, cache)


def _load(cache):
    saved = np.load(cache)
    views = {}
    for name in VIEWS:
        views[name] = saved[f"probs_{name}"]
    return {"clips": list(saved["clips"]), "labels": saved["labels"],
            "classes": list(saved["classes"]), "views": views}


def tta_scores(name, out, score=False, split=SPLIT, device=None):
    """A run's answers per view, kept beside its checkpoint. None if they are
    not there yet and score is off."""
    folder = os.path.join(out, name)
    cache = os.path.join(folder, CACHE)
    ckpt = os.path.join(folder, "best.pt")
    fresh = os.path.exists(cache)
    if fresh and os.path.exists(ckpt):
        # a retrained run must not be read from the old one's answers
        fresh = os.path.getmtime(cache) >= os.path.getmtime(ckpt)
    if not fresh:
        if not score:
            return None
        print(f"scoring {name}: {len(VIEWS)} views")
        found = view_probs(ckpt, os.path.join(R.SPLITS[split], "val.csv"),
                           device=device)
        _save(found, cache)
    return _load(cache)


def averaged(scores, views):
    """The mean of some views' probabilities."""
    total = None
    for name in views:
        if total is None:
            total = scores["views"][name].astype(np.float64)
        else:
            total = total + scores["views"][name]
    return total / len(views)


def tta_part(name, out, view_set, score=False, split=SPLIT, device=None):
    """A run's averaged answers, shaped like tuning.val_probs. None if absent."""
    scores = tta_scores(name, out, score, split, device)
    if scores is None:
        return None
    return {"clips": scores["clips"], "labels": scores["labels"],
            "classes": scores["classes"],
            "probs": averaged(scores, VIEW_SETS[view_set])}


def top1(probs, labels):
    return float(np.mean(np.argmax(probs, axis=1) == labels))


def rows_for(exp_id, split, latest, out, score=False, device=None):
    """(one row per run, runs not scored yet) for one model."""
    names = base_names(exp_id, latest, split)
    if not names:
        raise SystemExit(f"{exp_id} has no finished run on {split}")
    rows = []
    missing = []
    for name in names:
        scores = tta_scores(name, out, score, split, device)
        if scores is None:
            missing.append(name)
            continue
        row = {"name": name, "ledger": 100 * latest[name]["best_val_top1"],
               "plain": 100 * top1(scores["views"]["plain"], scores["labels"])}
        for view_set in VIEW_SETS:
            probs = averaged(scores, VIEW_SETS[view_set])
            row[view_set] = 100 * top1(probs, scores["labels"])
        rows.append(row)
    return rows, missing


def render(exp_id, rows):
    columns = ["ledger", "plain"]
    for view_set in VIEW_SETS:
        columns.append(view_set)
    header = f"  {'':<36}"
    for column in columns:
        header += f"{column:>8}"
    lines = [header]
    for row in rows:
        line = f"  {row['name']:<36}"
        for column in columns:
            line += f"{row[column]:>8.2f}"
        lines.append(line)
    line = f"  {exp_id + ' mean':<36}"
    for column in columns:
        total = 0.0
        for row in rows:
            total += row[column]
        line += f"{total / len(rows):>8.2f}"
    lines.append(line)
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", nargs="+", default=EXPS,
                    help="experiment IDs; all their seeds are scored")
    ap.add_argument("--split", choices=list(R.SPLITS), default=SPLIT)
    ap.add_argument("--score", action="store_true",
                    help="score what is missing on val; needs the GPU")
    ap.add_argument("--device", help="cpu scores beside training, slowly")
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args(argv)

    device = None
    if args.device:
        device = torch.device(args.device)
    latest = _ledger(read_results(args.results))
    done = []
    missing = []
    for exp_id in args.exp:
        rows, absent = rows_for(exp_id, args.split, latest, args.out, args.score,
                                device)
        done.append((exp_id, rows))
        for name in absent:
            missing.append(name)
    if missing:
        print("not scored yet: " + ", ".join(missing))
        print("When nothing is training: python -m tools.tta --score")
        return 1

    print(f"{args.split} val; plain should match the ledger")
    for exp_id, rows in done:
        for line in render(exp_id, rows):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
