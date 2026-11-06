"""Day 41's last step: freeze the ensemble chosen on val, then read test once.

    python -m tools.final save G02
    python -m tools.final save G01 --stream C06_mvitv2_s --stream T05_c06_ls005 --tta all --name tuned
    python -m tools.final test G02_final__grouped

save fits the chosen way of combining on all of val and writes
experiments/<G..>_<name>__<split>/report_choice.json, with every
checkpoint's SHA-256. It refuses an ensemble that does not beat its best
stream on val, scored held out for G02 and G04 (Day 41's check), and one
whose test has already been read. It reads the val answers ensemble.py keeps,
so run `python -m tools.ensemble --score` (with the same --tta) first.

test scores every run on test, combines them with the saved recipe unchanged
and writes report_test.json beside it. It runs once: a second try names the
report and stops, so nothing is ever tuned on test. A checkpoint that changed
since save stops it too. Use the GPU: about a minute per run, four with
test-time augmentation. A cut-off test resumes from the runs already scored.
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import torch

from src.evaluate import file_fingerprint, probabilities, scores_report
from src.experiments import registry as R
from src.methods.fusion import _average
from src.train import _clear_line, _ticker
from src.utils.checkpoint import RESULTS_PATH, atomic_write_json, read_results
from tools.ensemble import (STREAMS, align, fit_stack, fit_weights,
                            load_stream, match_folds, matches_of, score_all,
                            vote)
from tools.tta import VIEW_SETS, VIEWS, averaged, top1, view_probs
from tools.tuning import SPLIT, _ledger, base_names

__all__ = ["METHODS", "honest", "recipe_of", "apply", "save", "test_part",
           "test", "main"]

METHODS = ("G01", "G02", "G04")
CHOICE = "report_choice.json"
REPORT = "report_test.json"
NAME = "final"


def honest(method, scores):
    """The val score Day 41's check uses: held out wherever val was fitted."""
    if method == "G01":
        return scores["equal"]
    if method == "G02":
        return scores["weighted_held"]
    return scores["stack_held"]


def recipe_of(method, stacks, labels):
    """What the method learns on all of val, as plain lists for JSON."""
    if method == "G01":
        weights = []
        for _ in stacks:
            weights.append(1 / len(stacks))
        return {"weights": weights}
    if method == "G02":
        return {"weights": fit_weights(stacks, labels)}
    model = fit_stack(stacks, labels)
    return {"coef": model.coef_.tolist(), "intercept": model.intercept_.tolist(),
            "classes": model.classes_.tolist()}


def apply(recipe, stacks):
    """A saved recipe on new answers."""
    if "weights" in recipe:
        return vote(stacks, recipe["weights"])
    # the logistic head by hand: a softmax over its scores, as sklearn does
    scores = np.hstack(stacks) @ np.asarray(recipe["coef"]).T
    scores = scores + np.asarray(recipe["intercept"])
    scores = scores - scores.max(axis=1, keepdims=True)
    known = np.exp(scores)
    known = known / known.sum(axis=1, keepdims=True)
    probs = np.zeros_like(stacks[0], dtype=np.float64)
    probs[:, recipe["classes"]] = known
    return probs


def _folder(out, method, name, split):
    return os.path.join(out, f"{method}_{name}__{split}")


def save(method, ids, split, tta, name, out, results):
    """Fit the method on val and write report_choice.json. 0, or 1 if a run
    has no val answers yet."""
    if not re.fullmatch(r"[a-z0-9]+(_[a-z0-9]+)*", name):
        raise SystemExit(f"name {name!r}: lower case, digits and single _ only")
    if len(ids) < 2:
        raise SystemExit("an ensemble needs at least two streams")
    folder = _folder(out, method, name, split)
    if os.path.exists(os.path.join(folder, REPORT)):
        raise SystemExit(f"{folder}: its test is already read, so it cannot "
                         f"change now. Save under another --name.")

    latest = _ledger(read_results(results))
    streams = []
    missing = []
    for exp_id in ids:
        stream, absent = load_stream(exp_id, split, latest, out, False, tta)
        if stream is not None:
            streams.append(stream)
        for run in absent:
            missing.append(run)
    if missing:
        hint = "python -m tools.ensemble --score"
        if tta:
            hint += f" --tta {tta}"
        print("no val answers yet: " + ", ".join(missing))
        print("First, when nothing is training: " + hint)
        return 1

    clips, labels, stacks = align(streams)
    scores = score_all(labels, stacks, match_folds(matches_of(clips, split)))
    best = max(scores["members"])
    chosen = honest(method, scores)
    if chosen <= best:
        raise SystemExit(f"{method} scores {100 * chosen:.2f} on val, its best "
                         f"stream {100 * best:.2f}. Day 41's check fails: "
                         f"report the best stream alone.")

    saved_streams = []
    members = {}
    for stream, score in zip(streams, scores["members"]):
        runs = []
        for run in base_names(stream["id"], latest, split):
            ckpt = os.path.join(out, run, "best.pt")
            runs.append({"name": run, "checkpoint": file_fingerprint(ckpt)})
        saved_streams.append({"id": stream["id"], "runs": runs})
        members[stream["id"]] = score
    choice = {"method": method, "split": split, "tta": tta,
              "streams": saved_streams,
              "classes": list(streams[0]["classes"]),
              "recipe": recipe_of(method, stacks, labels),
              "val": {"clips": len(clips), "members": members,
                      "equal": scores["equal"],
                      "weighted_held": scores["weighted_held"],
                      "stack_held": scores["stack_held"], "chosen": chosen},
              "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    # answers kept for an older recipe would not be this one's
    shutil.rmtree(os.path.join(folder, "test_probs"), ignore_errors=True)
    atomic_write_json(choice, os.path.join(folder, CHOICE))
    print(f"saved {os.path.join(folder, CHOICE)}")
    print(f"{method} on {split} val: {100 * chosen:.2f}, best stream "
          f"{100 * best:.2f}, {len(clips)} clips")
    print(f"To read test once: python -m tools.final test "
          f"{os.path.basename(folder)}")
    return 0


def _store(arrays, cache):
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    # a power cut must not leave half a file that looks finished
    temp = cache[:-4] + ".tmp.npz"
    np.savez(temp, **arrays)
    os.replace(temp, cache)


def test_part(ckpt, sha256, split_csv, tta, cache, device=None):
    """A run's test answers, kept in the ensemble's folder so a cut-off test
    resumes. Averaged over the saved views when there are any."""
    fresh = os.path.exists(cache)
    if fresh:
        # Windows will not replace a file that is still open
        with np.load(cache) as saved:
            fresh = str(saved["sha256"]) == sha256
    if not fresh:
        if tta:
            found = view_probs(ckpt, split_csv, device=device)
            arrays = {"clips": np.asarray(found["clips"]),
                      "labels": np.asarray(found["labels"]),
                      "classes": np.asarray(found["classes"]),
                      "sha256": np.asarray(sha256)}
            for view in VIEWS:
                arrays[f"probs_{view}"] = found["views"][view]
        else:
            run = os.path.basename(os.path.dirname(os.path.abspath(ckpt)))
            found = probabilities(ckpt, split_csv, batch_size=4, device=device,
                                  on_batch=_ticker(f"  {run} ", False))
            _clear_line()
            arrays = {"clips": np.asarray(found["clips"]),
                      "labels": np.asarray(found["labels"]),
                      "classes": np.asarray(found["classes"]),
                      "sha256": np.asarray(sha256),
                      "probs": found["probs"]}
        _store(arrays, cache)
    with np.load(cache) as saved:
        part = {"clips": list(saved["clips"]), "labels": saved["labels"],
                "classes": list(saved["classes"])}
        if tta:
            views = {}
            for view in VIEWS:
                views[view] = saved[f"probs_{view}"]
            part["probs"] = averaged({"views": views}, VIEW_SETS[tta])
        else:
            part["probs"] = saved["probs"]
    return part


def test(folder_name, out, device=None):
    """Score the saved ensemble on test, once. 0, or 1 if already read."""
    folder = os.path.join(out, folder_name)
    choice_path = os.path.join(folder, CHOICE)
    if not os.path.exists(choice_path):
        raise SystemExit(f"no saved ensemble at {choice_path}; save one first")
    report_path = os.path.join(folder, REPORT)
    if os.path.exists(report_path):
        print(f"test was read once already: {report_path}")
        return 1
    with open(choice_path, encoding="utf-8") as handle:
        choice = json.load(handle)

    split_csv = os.path.join(R.SPLITS[choice["split"]], "test.csv")
    # every checkpoint first, so a changed one stops the test before it starts
    for stream in choice["streams"]:
        for run in stream["runs"]:
            ckpt = os.path.join(out, run["name"], "best.pt")
            if not os.path.exists(ckpt):
                raise SystemExit(f"{ckpt} is gone")
            if file_fingerprint(ckpt)["sha256"] != run["checkpoint"]["sha256"]:
                raise SystemExit(f"{ckpt} changed since the ensemble was "
                                 f"saved. Save it again on val first.")

    streams = []
    for stream in choice["streams"]:
        parts = []
        for run in stream["runs"]:
            print(f"scoring {run['name']} on test")
            cache = os.path.join(folder, "test_probs", run["name"] + ".npz")
            parts.append(test_part(os.path.join(out, run["name"], "best.pt"),
                                   run["checkpoint"]["sha256"], split_csv,
                                   choice["tta"], cache, device))
        clips, labels, probs = _average(parts)
        streams.append({"id": stream["id"], "clips": clips, "labels": labels,
                        "probs": probs, "classes": parts[0]["classes"]})
    clips, labels, stacks = align(streams)
    report = scores_report(apply(choice["recipe"], stacks), labels,
                           choice["classes"])
    members = {}
    for stream, stack in zip(streams, stacks):
        members[stream["id"]] = top1(stack, labels)
    report.update({"method": choice["method"], "split": choice["split"],
                   "tta": choice["tta"], "split_csv": os.path.abspath(split_csv),
                   "members_top1": members, "streams": choice["streams"],
                   "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    atomic_write_json(report, report_path)

    print(f"{folder_name} on {choice['split']} test, {len(clips)} clips: "
          f"top-1 {100 * report['top1']:.2f}, macro-F1 {report['macro_f1']:.3f}")
    for stream_id in members:
        print(f"  {stream_id} alone: {100 * members[stream_id]:.2f}")
    print(f"wrote {report_path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="action", required=True)
    keep = sub.add_parser("save", help="fit on val and freeze the recipe")
    keep.add_argument("method", choices=METHODS)
    keep.add_argument("--stream", action="append",
                      help="an experiment ID; give it once per model")
    keep.add_argument("--split", choices=list(R.SPLITS), default=SPLIT)
    keep.add_argument("--tta", choices=list(VIEW_SETS))
    keep.add_argument("--name", default=NAME)
    keep.add_argument("--results", default=RESULTS_PATH)
    keep.add_argument("--out", default="experiments")
    read = sub.add_parser("test", help="score the saved recipe on test, once")
    read.add_argument("folder", help="e.g. G02_final__grouped")
    read.add_argument("--device")
    read.add_argument("--out", default="experiments")
    args = ap.parse_args(argv)

    if args.action == "save":
        return save(args.method, args.stream or STREAMS, args.split, args.tta,
                    args.name, args.out, args.results)
    device = None
    if args.device:
        device = torch.device(args.device)
    return test(args.folder, args.out, device)


if __name__ == "__main__":
    sys.exit(main())
