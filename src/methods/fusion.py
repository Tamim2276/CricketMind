"""Late fusion: average finished runs' probabilities, train nothing (29.1).

    fuse: [["A09_tf4l8h"], ["E01_pose_gru", "E02_stgcn"]]

Each inner list is one stream. From each, the run with the best val top-1
on this split is used, so "the better of E01 and E02" is decided on val,
never on test. Every chosen run's own checkpoint scores the split, and each
clip's probabilities are averaged over the streams. F03 and Phase 4's
ensembles reuse it.

A clip counts only when every stream has it: the streams read different
caches, and an average of one stream is not a fusion.
"""
import json
import os
import time

import numpy as np

from src.utils.checkpoint import append_result, atomic_write_json, read_results

NAME = "fusion"

READS = ("fuse",)

CHECKPOINT = "fusion.json"


def _streams_problem(fuse):
    if not isinstance(fuse, list) or not fuse:
        return "fusion needs fuse: a list of streams, each a list of IDs"
    for group in fuse:
        if not isinstance(group, list) or not group:
            return "every stream in fuse must be a non-empty list of IDs"
    return None


def _last_row(name, ledger):
    found = None
    for row in read_results(ledger):
        if row.get("name") == name:
            found = row
    return found


def _choose(cfg):
    """(chosen runs, None), or (None, why not). One run per stream: the
    best on val among those that have finished on this split."""
    from src.experiments import registry as R
    from src.train import split_name

    ledger = os.path.join(cfg["out"], "results.jsonl")
    split = split_name(cfg["split"])
    chosen = []
    for group in cfg["fuse"]:
        best = None
        waiting = []
        for exp_id in group:
            try:
                R.get(exp_id)
            except KeyError:
                return None, f"fuse names {exp_id}, which is not an experiment"
            name = f"{exp_id}__{split}"
            row = _last_row(name, ledger)
            if row is None:
                waiting.append(exp_id)
                continue
            ckpt = row.get("checkpoint") or os.path.join(cfg["out"], name,
                                                         "best.pt")
            if not ckpt.endswith(".pt"):
                return None, f"{exp_id} has no best.pt; fusion reads trained runs"
            if best is None or row["best_val_top1"] > best["val_top1"]:
                best = {"id": exp_id, "name": name, "checkpoint": ckpt,
                        "val_top1": row["best_val_top1"]}
        if best is None:
            return None, f"needs {' or '.join(waiting)} on {split} first"
        chosen.append(best)
    return chosen, None


def check(cfg):
    """Why this fusion cannot run yet, or None."""
    problem = _streams_problem(cfg["fuse"])
    if problem:
        return problem
    _, problem = _choose(cfg)
    return problem


def _average(parts):
    """(clips, labels, mean probabilities) over the clips every stream has."""
    shared = set(parts[0]["clips"])
    for part in parts[1:]:
        if part["classes"] != parts[0]["classes"]:
            raise ValueError("the streams were trained on different classes")
        shared = shared & set(part["clips"])
    order = []
    for clip in parts[0]["clips"]:
        if clip in shared:
            order.append(clip)
    total = None
    labels = None
    for part in parts:
        where = {}
        for i, clip in enumerate(part["clips"]):
            where[clip] = i
        rows = []
        for clip in order:
            rows.append(where[clip])
        if labels is None:
            labels = np.asarray(part["labels"])[rows]
        elif not np.array_equal(labels, np.asarray(part["labels"])[rows]):
            raise ValueError("the streams disagree about a clip's label")
        if total is None:
            total = part["probs"][rows].astype(np.float64)
        else:
            total = total + part["probs"][rows]
    return order, labels, total / len(parts)


def _fused(sources, split_csv, batch_size, workers):
    from src.evaluate import probabilities

    parts = []
    for source in sources:
        parts.append(probabilities(source["checkpoint"], split_csv,
                                   batch_size=batch_size, workers=workers))
    clips, labels, probs = _average(parts)
    return clips, labels, probs, parts[0]["classes"]


def fit(cfg, quiet=False):
    """Choose the runs on val, fuse them on val, save fusion.json, append
    the ledger row. Nothing is trained."""
    from src.evaluate import scores_report
    from src.train import run_dir, split_name

    sources, problem = _choose(cfg)
    if problem:
        raise SystemExit(f"{cfg['name']}: {problem}")
    started = time.time()
    clips, labels, probs, classes = _fused(
        sources, os.path.join(cfg["split"], "val.csv"), cfg["batch_size"],
        cfg["workers"])
    val_top1 = scores_report(probs, labels, classes)["top1"]
    out = run_dir(cfg)
    os.makedirs(out, exist_ok=True)
    atomic_write_json({"cfg": cfg, "sources": sources, "classes": classes,
                       "val_top1": val_top1}, os.path.join(out, CHECKPOINT))
    if not quiet:
        names = []
        for source in sources:
            names.append(f"{source['id']} ({source['val_top1']:.2%})")
        print(f"{cfg['name']} [fusion]: {' + '.join(names)} on "
              f"{len(clips)} val clips -> {val_top1:.2%}")

    row = {"name": cfg["name"], "method": cfg["method"],
           "split": cfg["split"], "split_name": split_name(cfg["split"]),
           "variant": cfg["variant"], "features": None, "view": None,
           "backbone": None, "frozen": True, "best_val_top1": val_top1,
           "epochs_run": 0, "stopped": "no training: fused",
           "minutes": (time.time() - started) / 60, "resumed": False,
           "deterministic": True, "device": None, "sources": sources,
           "cfg": cfg}
    append_result(row, os.path.join(cfg["out"], "results.jsonl"))
    return row


def score(ckpt_path, split_csv):
    """The same report evaluate.py writes, for the fused runs."""
    from src.evaluate import file_fingerprint, scores_report

    with open(ckpt_path, encoding="utf-8") as fh:
        saved = json.load(fh)
    cfg = saved["cfg"]
    clips, labels, probs, classes = _fused(saved["sources"], split_csv,
                                           cfg["batch_size"], cfg["workers"])
    report = {"checkpoint": file_fingerprint(ckpt_path),
              "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "split_csv": os.path.abspath(split_csv),
              "method": cfg["method"], "variant": cfg["variant"],
              "features": None, "view": None, "dropped": None,
              "sources": saved["sources"]}
    report.update(scores_report(probs, labels, classes))
    report["trained_epoch"] = 0
    report["trained_val_top1"] = saved["val_top1"]
    return report
