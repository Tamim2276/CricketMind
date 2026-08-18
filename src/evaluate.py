"""
Evaluate a trained checkpoint on the test split.

Usage:
    python src/evaluate.py --checkpoint experiments/baseline/checkpoints/best.pt
    python src/evaluate.py --checkpoint /kaggle/working/checkpoints/baseline/best.pt --split val
"""
import os
import sys
import json
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.device import get_device, get_amp_settings
from src.utils.metrics import topk_accuracy, per_class_report, get_confusion_matrix, mcnemar_test
from src.utils.viz import save_confusion_matrix
from src.datasets.cricshot_dataset import CricShotDataset, CLASS_NAMES
from torch.utils.data import DataLoader


def build_model_from_cfg(cfg):
    model_type  = cfg.get("model_type", "baseline")
    num_classes = cfg.get("num_classes", 15)
    if model_type == "baseline":
        from src.models.baseline_model import BaselineModel
        return BaselineModel(num_classes=num_classes, pretrained=False)
    raise ValueError(f"Unknown model_type: {model_type!r}")


@torch.no_grad()
def run_eval(model, loader, device, amp_dtype):
    model.eval()
    all_logits, all_labels = [], []
    for clips, labels in loader:
        clips = clips.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(amp_dtype is not None),
        ):
            logits = model(clips)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="test", choices=["train", "val", "test"])
    parser.add_argument("--out_dir",    default=None,
                        help="directory to save confusion matrix / report (default: checkpoint dir)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg  = ckpt["cfg"]

    device    = get_device()
    amp_dtype, _ = get_amp_settings(device)

    model = build_model_from_cfg(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint (epoch {ckpt['epoch']}, val top-1 {ckpt['val_top1']:.2f}%)")

    csv_path = os.path.join(cfg["splits_dir"], f"{args.split}.csv")
    ds = CricShotDataset(
        csv_path=csv_path,
        processed_root=cfg["processed_root"],
        split=args.split,
        num_frames=cfg.get("frames_per_clip", 15),
    )
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 8),
                        shuffle=False, num_workers=4, pin_memory=True)

    print(f"Evaluating on {args.split} set ({len(ds)} clips)...")
    logits, labels = run_eval(model, loader, device, amp_dtype)
    preds = logits.argmax(dim=1)

    # ── accuracy ─────────────────────────────────────────────────────────────
    acc = topk_accuracy(logits, labels, topk=(1, 2, 3))
    print(f"\nTop-1: {acc['top1']:.2f}%  Top-2: {acc['top2']:.2f}%  Top-3: {acc['top3']:.2f}%")

    # ── per-class report ──────────────────────────────────────────────────────
    report = per_class_report(labels.numpy(), preds.numpy(), CLASS_NAMES)
    print("\nPer-class report:")
    for cls in CLASS_NAMES:
        r = report.get(cls, {})
        print(f"  {cls:<20s}  P={r.get('precision', 0):.3f}  "
              f"R={r.get('recall', 0):.3f}  F1={r.get('f1-score', 0):.3f}  "
              f"n={r.get('support', 0)}")

    # ── confusion matrix ──────────────────────────────────────────────────────
    out_dir = args.out_dir or os.path.dirname(args.checkpoint)
    cm, _ = get_confusion_matrix(labels.numpy(), preds.numpy(), CLASS_NAMES)
    save_confusion_matrix(
        cm, CLASS_NAMES,
        save_path=os.path.join(out_dir, f"confusion_matrix_{args.split}.png"),
        title=f"{cfg.get('model_type','model')} — {args.split} set",
    )

    # ── save preds for McNemar test later ─────────────────────────────────────
    # Cast to float32 first — bfloat16 (XPU AMP) is not supported by numpy.
    preds_path = os.path.join(out_dir, f"preds_{args.split}.npz")
    np.savez(preds_path,
             logits=logits.float().numpy(),
             preds=preds.numpy(),
             labels=labels.numpy())
    print(f"Predictions saved → {preds_path}")

    # ── save report JSON ──────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, f"report_{args.split}.json")
    report["_topk"] = acc
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
