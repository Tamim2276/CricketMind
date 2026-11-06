"""Family G: combine finished runs, train nothing (Day 41).

    python -m tools.ensemble            what is already scored; no GPU
    python -m tools.ensemble --score    first scores what is missing, on val
    python -m tools.ensemble --stream C06_mvitv2_s --stream T05_c06_ls005
    python -m tools.ensemble --tta all  each run's answers come from tools.tta

A stream is one model. Its ID brings in all its seeds, and they are averaged
before anything else, so a model with more seeds gets no extra votes.

    G01   equal vote over the streams
    G02   a weight per stream, fitted on val
    G04   stacking: a logistic head over the streams' probabilities

G02 and G04 learn from val, so a score on the clips they learned from flatters
them. They are also scored held out: five folds, each match whole in one fold.

Val only. Test stays closed (rule 5). Use --score when nothing is training.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from src.experiments import registry as R
from src.methods.fusion import _average
from src.utils.checkpoint import RESULTS_PATH, read_results
from tools.tta import VIEW_SETS, top1, tta_part
from tools.tuning import SPLIT, _ledger, base_names, val_probs

__all__ = ["load_stream", "align", "matches_of", "vote", "top1", "loss_of",
           "weight_grid", "fit_weights", "match_folds", "fit_stack",
           "stack_probs", "held_out_weights", "held_out_stack", "score_all",
           "render", "main"]

STREAMS = ["C06_mvitv2_s", "F02_multiscale_tf"]
STEP = 0.05
FOLDS = 5


def load_stream(exp_id, split, latest, out, score=False, tta=None, device=None):
    """(stream, missing). One model's runs averaged, or the runs not scored."""
    names = base_names(exp_id, latest, split)
    if not names:
        raise SystemExit(f"{exp_id} has no finished run on {split}")
    parts = []
    missing = []
    for name in names:
        if tta:
            part = tta_part(name, out, tta, score, split, device)
        else:
            part = val_probs(name, out, score, split, device)
        if part is None:
            missing.append(name)
        else:
            parts.append(part)
    if missing:
        return None, missing
    clips, labels, probs = _average(parts)
    stream = {"id": exp_id, "runs": len(names), "clips": clips,
              "labels": labels, "probs": probs,
              "classes": parts[0]["classes"]}
    return stream, []


def align(streams):
    """(clips, labels, one array per stream), all in one clip order."""
    # _average already checks the classes and labels agree
    clips, labels, _ = _average(streams)
    stacks = []
    for stream in streams:
        where = {}
        for i, clip in enumerate(stream["clips"]):
            where[clip] = i
        rows = []
        for clip in clips:
            rows.append(where[clip])
        stacks.append(stream["probs"][rows])
    return clips, labels, stacks


def matches_of(clips, split):
    """Each clip's match, from the split's own file."""
    where = {}
    path = os.path.join(R.SPLITS[split], "val.csv")
    with open(path, encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            where[row["clip"]] = row["match"]
    found = []
    for clip in clips:
        found.append(where[clip])
    return found


def vote(stacks, weights=None):
    """The streams' probabilities, averaged. No weights means equal ones."""
    if weights is None:
        weights = [1 / len(stacks)] * len(stacks)
    total = np.zeros_like(stacks[0], dtype=np.float64)
    for stack, weight in zip(stacks, weights):
        total = total + weight * stack
    return total


def loss_of(probs, labels):
    picked = probs[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(picked, 1e-12, None))))


def _shares(slots, left):
    """Every way to hand out `left` equal shares over `slots`."""
    if slots == 1:
        return [[left]]
    found = []
    for share in range(left + 1):
        for rest in _shares(slots - 1, left - share):
            found.append([share] + rest)
    return found


def weight_grid(n, step=STEP):
    """Every set of n weights that adds to 1, in steps of `step`."""
    parts = int(round(1 / step))
    grid = []
    for shares in _shares(n, parts):
        weights = []
        for share in shares:
            weights.append(share / parts)
        grid.append(weights)
    return grid


def fit_weights(stacks, labels, step=STEP):
    """The best weights by top-1. Many tie on top-1, so loss breaks a tie."""
    best = None
    best_top1 = -1.0
    best_loss = None
    for weights in weight_grid(len(stacks), step):
        probs = vote(stacks, weights)
        score = top1(probs, labels)
        if score < best_top1:
            continue
        loss = loss_of(probs, labels)
        if score > best_top1 or loss < best_loss:
            best = weights
            best_top1 = score
            best_loss = loss
    return best


def match_folds(matches, folds=FOLDS):
    """A fold number per clip. A match stays whole, biggest match first, each
    into the smallest fold so far, so the answer is the same every time."""
    counts = {}
    for match in matches:
        counts[match] = counts.get(match, 0) + 1
    if len(counts) < folds:
        raise ValueError(f"{len(counts)} matches cannot fill {folds} folds")
    order = []
    for match in counts:
        order.append((-counts[match], match))
    order.sort()
    sizes = [0] * folds
    fold_of = {}
    for negative, match in order:
        fold = int(np.argmin(sizes))
        fold_of[match] = fold
        sizes[fold] += -negative
    found = []
    for match in matches:
        found.append(fold_of[match])
    return np.asarray(found)


def fit_stack(stacks, labels):
    """A logistic head over all the streams' probabilities, side by side."""
    model = LogisticRegression(max_iter=1000)
    model.fit(np.hstack(stacks), labels)
    return model


def stack_probs(model, stacks):
    """The head's answers, with a column for every class. A fold can be short
    of a rare class, and the head then has no column for it."""
    known = model.predict_proba(np.hstack(stacks))
    probs = np.zeros_like(stacks[0], dtype=np.float64)
    probs[:, model.classes_] = known
    return probs


def _take(stacks, rows):
    taken = []
    for stack in stacks:
        taken.append(stack[rows])
    return taken


def held_out_weights(stacks, labels, folds):
    """Each fold's answers, from weights fitted on the other folds."""
    probs = np.zeros_like(stacks[0], dtype=np.float64)
    for fold in np.unique(folds):
        held = folds == fold
        weights = fit_weights(_take(stacks, ~held), labels[~held])
        probs[held] = vote(_take(stacks, held), weights)
    return probs


def held_out_stack(stacks, labels, folds):
    """Each fold's answers, from a head fitted on the other folds."""
    probs = np.zeros_like(stacks[0], dtype=np.float64)
    for fold in np.unique(folds):
        held = folds == fold
        model = fit_stack(_take(stacks, ~held), labels[~held])
        probs[held] = stack_probs(model, _take(stacks, held))
    return probs


def score_all(labels, stacks, folds):
    """Every stream and every way of combining them, by top-1."""
    members = []
    for stack in stacks:
        members.append(top1(stack, labels))
    weights = fit_weights(stacks, labels)
    return {"members": members,
            "equal": top1(vote(stacks), labels),
            "weights": weights,
            "weighted": top1(vote(stacks, weights), labels),
            "weighted_held": top1(held_out_weights(stacks, labels, folds),
                                  labels),
            "stack_held": top1(held_out_stack(stacks, labels, folds), labels)}


def render(streams, scores, split, clips):
    lines = [f"{split} val, {clips} clips"]
    for stream, score in zip(streams, scores["members"]):
        lines.append(f"  {stream['id']:<24}{stream['runs']} runs"
                     f"{100 * score:>9.2f}")
    best = max(scores["members"])
    shown = []
    for weight in scores["weights"]:
        shown.append(f"{weight:.2f}")
    rows = [("G01 equal vote", scores["equal"], ""),
            ("G02 weights, held out", scores["weighted_held"], ""),
            ("G02 weights, all val", scores["weighted"],
             f"  flatters; weights {' '.join(shown)}"),
            ("G04 stacking, held out", scores["stack_held"], "")]
    lines.append(f"  {'':<31}{'val':>9}{'+-best':>8}")
    for label, score, note in rows:
        lines.append(f"  {label:<31}{100 * score:>9.2f}"
                     f"{100 * (score - best):>+8.2f}{note}")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stream", action="append",
                    help="an experiment ID; give it once per model")
    ap.add_argument("--split", choices=list(R.SPLITS), default=SPLIT)
    ap.add_argument("--tta", choices=list(VIEW_SETS),
                    help="average each run over these views first")
    ap.add_argument("--score", action="store_true",
                    help="score what is missing on val; needs the GPU")
    ap.add_argument("--device", help="cpu scores beside training, slowly")
    ap.add_argument("--results", default=RESULTS_PATH)
    ap.add_argument("--out", default="experiments")
    args = ap.parse_args(argv)

    ids = args.stream or STREAMS
    if len(ids) < 2:
        raise SystemExit("an ensemble needs at least two streams")
    device = None
    if args.device:
        device = torch.device(args.device)
    latest = _ledger(read_results(args.results))
    streams = []
    missing = []
    for exp_id in ids:
        stream, absent = load_stream(exp_id, args.split, latest, args.out,
                                     args.score, args.tta, device)
        if stream is not None:
            streams.append(stream)
        for name in absent:
            missing.append(name)
    if missing:
        hint = "python -m tools.ensemble --score"
        if args.tta:
            hint += f" --tta {args.tta}"
        print("not scored yet: " + ", ".join(missing))
        print("When nothing is training: " + hint)
        return 1

    clips, labels, stacks = align(streams)
    folds = match_folds(matches_of(clips, args.split))
    scores = score_all(labels, stacks, folds)
    if args.tta:
        print(f"each run averaged over: {' '.join(VIEW_SETS[args.tta])}")
    for line in render(streams, scores, args.split, len(clips)):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
