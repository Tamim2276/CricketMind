"""Score a checkpoint, and make a stale report impossible.

This project has already lost a day to a stale evaluation: a 23.67% result was
read as a class-order bug when the real cause was that the report had been
written 6h43m *before* the checkpoint it claimed to measure. Nothing in the
report said so.

So every report here carries the checkpoint's **SHA-256**, size and mtime, and
`load_report` re-hashes the file and refuses if it has changed.

A hash rather than a timestamp, because an mtime proves very little -- a file
can be rewritten with the same one, clocks drift, and copying a file moves it.
The hash answers the only question that matters: are these numbers from *this*
file?

And it refuses rather than warns. A warning inside a six-hour log is a warning
nobody reads.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import CricShotDataset
from src.engine import evaluate as run_eval
from src.models.model import build_model
from src.utils.checkpoint import atomic_write_json
from src.utils.device import get_device

__all__ = ["file_fingerprint", "StaleReport", "evaluate_checkpoint",
           "load_report", "confusion", "per_class"]


class StaleReport(RuntimeError):
    """The report does not describe the checkpoint it was asked about."""


def file_fingerprint(path: str) -> dict:
    """SHA-256, size and mtime. The hash is what the guard actually checks."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            digest.update(block)
    info = os.stat(path)
    return {"path": os.path.abspath(path),
            "sha256": digest.hexdigest(),
            "bytes": info.st_size,
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                                   time.localtime(info.st_mtime))}


def confusion(labels, preds, n_classes: int) -> np.ndarray:
    """Rows are the truth, columns the prediction."""
    out = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(labels, preds):
        out[int(true), int(pred)] += 1
    return out


def per_class(matrix: np.ndarray) -> dict:
    """Precision, recall and F1 for each class, straight from the matrix."""
    precision = []
    recall = []
    f1 = []
    support = []
    for i in range(matrix.shape[0]):
        tp = int(matrix[i, i])
        predicted = int(matrix[:, i].sum())
        actual = int(matrix[i, :].sum())

        if predicted:
            p = tp / predicted
        else:
            p = 0.0
        if actual:
            r = tp / actual
        else:
            r = 0.0
        if p + r:
            f = 2 * p * r / (p + r)
        else:
            f = 0.0

        precision.append(p)
        recall.append(r)
        f1.append(f)
        support.append(actual)
    return {"precision": precision, "recall": recall, "f1": f1,
            "support": support}


def evaluate_checkpoint(ckpt_path: str, split_csv: str, variant=None,
                        processed=None, batch_size: int = 6, device=None,
                        workers: int = 0) -> dict:
    """Load a checkpoint, score it on one split, return the report."""
    device = device or get_device()
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = saved.get("cfg") or {}
    classes = saved.get("classes")

    if variant is None:
        variant = cfg.get("variant", "box")
    if processed is None:
        processed = cfg.get("processed", os.path.join("data", "processed"))

    data = CricShotDataset(split_csv, variant=variant, processed=processed,
                           classes=classes)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False,
                        num_workers=workers)

    model = build_model(num_classes=len(data.classes),
                        backbone=cfg.get("backbone", "efficientnet_v2_s"),
                        pretrained=False, hidden=cfg.get("hidden", 128),
                        dense=cfg.get("dense", 1024),
                        dropout=cfg.get("dropout", 0.0),
                        pool=cfg.get("pool", "avg")).to(device)
    model.load_state_dict(saved["model_state"])

    stats = run_eval(model, loader, device=device, collect=True)
    matrix = confusion(stats.labels, stats.preds, len(data.classes))
    scores = per_class(matrix)

    return {
        "checkpoint": file_fingerprint(ckpt_path),
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split_csv": os.path.abspath(split_csv),
        "variant": variant,
        "clips": stats.clips,
        "dropped": len(data.missing),
        "classes": list(data.classes),
        "loss": stats.loss,
        "top1": stats.top1,
        "top2": stats.top2,
        "top3": stats.top3,
        "macro_f1": float(np.mean(scores["f1"])),
        "macro_precision": float(np.mean(scores["precision"])),
        "macro_recall": float(np.mean(scores["recall"])),
        "per_class": scores,
        "confusion": matrix.tolist(),
        "trained_epoch": saved.get("epoch"),
        "trained_val_top1": saved.get("val_top1"),
    }


def load_report(report_path: str, ckpt_path=None) -> dict:
    """Read a report, refusing it if its checkpoint has changed since."""
    with open(report_path, encoding="utf-8") as fh:
        report = json.load(fh)

    recorded = report.get("checkpoint") or {}
    if ckpt_path is None:
        ckpt_path = recorded.get("path")
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise StaleReport(
            f"{report_path} describes {recorded.get('path')!r}, which is not "
            f"there now")

    now = file_fingerprint(ckpt_path)
    if now["sha256"] != recorded.get("sha256"):
        raise StaleReport(
            f"{report_path} was written for a different file.\n"
            f"  report : {recorded.get('sha256', '?')[:16]}  "
            f"{recorded.get('mtime')}\n"
            f"  on disk: {now['sha256'][:16]}  {now['mtime']}\n"
            f"Re-run the evaluation; do not read these numbers.")
    return report


def format_report(report: dict, top_confusions: int = 6) -> str:
    """The human-readable version, for the terminal and the journal."""
    lines = []
    lines.append(f"checkpoint  {report['checkpoint']['path']}")
    lines.append(f"            sha256 {report['checkpoint']['sha256'][:16]}  "
                 f"{report['checkpoint']['mtime']}")
    lines.append(f"split       {report['split_csv']}  "
                 f"({report['clips']} clips, {report['dropped']} dropped)")
    lines.append("")
    lines.append(f"  top-1 {report['top1']:7.2%}    top-2 {report['top2']:7.2%}"
                 f"    top-3 {report['top3']:7.2%}")
    lines.append(f"  macro-F1 {report['macro_f1']:.4f}   "
                 f"precision {report['macro_precision']:.4f}   "
                 f"recall {report['macro_recall']:.4f}   "
                 f"loss {report['loss']:.4f}")
    lines.append("")
    lines.append(f"  {'class':18s} {'prec':>6s} {'rec':>6s} {'F1':>6s} {'n':>5s}")

    order = np.argsort(report["per_class"]["f1"])
    for i in order:
        lines.append(f"  {report['classes'][i]:18s} "
                     f"{report['per_class']['precision'][i]:6.3f} "
                     f"{report['per_class']['recall'][i]:6.3f} "
                     f"{report['per_class']['f1'][i]:6.3f} "
                     f"{report['per_class']['support'][i]:5d}")

    matrix = np.array(report["confusion"])
    mistakes = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and matrix[i, j] > 0:
                mistakes.append((int(matrix[i, j]), i, j))
    mistakes.sort(reverse=True)

    lines.append("")
    lines.append("  most common confusions (true -> predicted)")
    for count, i, j in mistakes[:top_confusions]:
        lines.append(f"    {report['classes'][i]:18s} -> "
                     f"{report['classes'][j]:18s} {count:4d}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("split_csv")
    ap.add_argument("--out", help="where to write the JSON report")
    ap.add_argument("--variant", choices=("box", "seg"))
    ap.add_argument("--processed")
    ap.add_argument("--batch_size", type=int, default=6)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args(argv)

    report = evaluate_checkpoint(args.checkpoint, args.split_csv,
                                 variant=args.variant,
                                 processed=args.processed,
                                 batch_size=args.batch_size,
                                 workers=args.workers)
    print(format_report(report))
    if args.out:
        atomic_write_json(report, args.out)
        print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
