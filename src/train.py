"""
Training script for CricShot10k models.

Usage — local smoke test (1 epoch, 50 clips, Arc B580):
    python src/train.py --config configs/baseline.yaml --smoke_test

Usage — exact author mimic (target: reproduce 89%):
    python src/train.py --config configs/mimic_author.yaml

Usage — improved baseline:
    python src/train.py --config configs/baseline_kaggle.yaml

KEY DIFFERENCES vs AUTHOR (mimic_author=True restores author behaviour):
  - dropout=0.0 in GRUHead (author has NO Dropout)
  - ReduceLROnPlateau scheduler (author: factor=0.1, patience=4)
  - Single LR for all params (backbone_lr_multiplier=1.0)
  - weight_decay=0.0 (Keras Adam default)
  - label_smoothing=0.0
  - No MixUp / ColorJitter / RandomErasing (horizontal flip only, offline)
  - batch_size=4 (author notebook, not 8 as stated in paper)
  - epochs=100 (author notebook, not 50 as stated in paper)
"""
import os
import sys
import json
import argparse
import time

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.device import get_device, get_amp_settings
from src.utils.metrics import topk_accuracy, per_class_report, get_confusion_matrix
from src.utils.viz import save_confusion_matrix, save_accuracy_bar
from src.datasets.cricshot_dataset import CricShotDataset, CLASS_NAMES


#model registry
def build_model(model_type: str, num_classes: int, pretrained: bool,
                mimic_author: bool = False):
    """
    mimic_author=True -> dropout=0.0 in GRUHead (matches author's Keras arch).
    mimic_author=False -> dropout=0.5 for improved regularisation.
    """
    dropout = 0.0 if mimic_author else 0.5

    if model_type == "baseline":
        from src.models.baseline_model import BaselineModel
        return BaselineModel(num_classes=num_classes, pretrained=pretrained,
                             dropout=dropout)
    if model_type == "multiscale":
        from src.models.multiscale_model import MultiScaleModel
        return MultiScaleModel(num_classes=num_classes, pretrained=pretrained)
    if model_type == "transformer":
        from src.models.transformer_model import TransformerModel
        return TransformerModel(num_classes=num_classes, pretrained=pretrained)
    if model_type == "combined":
        from src.models.combined_model import CombinedModel
        return CombinedModel(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model_type: {model_type!r}. "
                     f"Choices: baseline | multiscale | transformer | combined")


# helpers
def load_config(path: str) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_dataloader(cfg, split, smoke_test):
    csv_path    = os.path.join(cfg["splits_dir"], f"{split}.csv")
    num_workers = cfg.get("num_workers", 4)
    cache = cfg.get("cache", False) and (split == "train") and not smoke_test
    ds = CricShotDataset(
        csv_path=csv_path,
        processed_root=cfg["processed_root"],
        split=split,
        num_frames=cfg.get("frames_per_clip", 15),
        smoke_test=smoke_test,
        cache=cache,
        mimic_author=cfg.get("mimic_author", False),
    )
    shuffle = (split == "train")
    extra = {}
    if num_workers > 0:
        extra["persistent_workers"] = True
        extra["prefetch_factor"]     = cfg.get("prefetch_factor", 2)
    return DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
        **extra,
    )


#scheduler factory 
def build_scheduler(cfg, optimizer, epochs):
    """
    Returns the LR scheduler requested by config.

    scheduler: "reduce_lr_on_plateau"  -> author's exact scheduler
               "cosine"                -> improved variant (default)
               anything else           -> cosine
    """
    scheduler_type = cfg.get("scheduler", "cosine").lower()

    if scheduler_type == "reduce_lr_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.get("lr_decay_factor", 0.1),
            patience=cfg.get("lr_decay_patience", 4),
        ), "plateau"

    # Default: CosineAnnealingLR
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-6,
    ), "cosine"


# mixup helpers
import numpy as np

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# train one epoch
def train_epoch(model, loader, criterion, optimizer, scaler, device, amp_dtype,
                grad_accum_steps=1, mimic_author=False):
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []

    optimizer.zero_grad()
    bar = tqdm(enumerate(loader), total=len(loader),
               desc="  train", unit="batch", leave=False, dynamic_ncols=True)

    for step, (clips, labels) in bar:
        clips  = clips.to(device)
        labels = labels.to(device)

        # MixUp disabled when mimicking author (author uses no online augmentation)
        if not mimic_author:
            clips, targets_a, targets_b, lam = mixup_data(clips, labels, alpha=0.2)
        else:
            targets_a, targets_b, lam = labels, labels, 1.0

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(amp_dtype is not None),
        ):
            logits = model(clips)
            if not mimic_author:
                loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            else:
                loss = criterion(logits, labels)
            loss = loss / grad_accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.cpu())
        bar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}")

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc = topk_accuracy(all_logits, all_labels, topk=(1, 2, 3))
    return total_loss / len(loader), acc


# evaluate
@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_dtype):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    bar = tqdm(loader, desc="    val", unit="batch", leave=False, dynamic_ncols=True)
    for clips, labels in bar:
        clips  = clips.to(device)
        labels = labels.to(device)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=(amp_dtype is not None),
        ):
            logits = model(clips)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    acc = topk_accuracy(all_logits, all_labels, topk=(1, 2, 3))
    return total_loss / len(loader), acc, all_logits, all_labels


# main 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, help="path to YAML config")
    parser.add_argument("--smoke_test", action="store_true",
                        help="50-clip subset, 1 epoch — just confirm code runs")
    parser.add_argument("--model_type", default=None,
                        help="override model_type from config")
    parser.add_argument("--epochs",     type=int, default=None,
                        help="override epochs from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_type:
        cfg["model_type"] = args.model_type
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.smoke_test:
        cfg["epochs"] = 1  # force 1 epoch in smoke test regardless of config

    mimic_author = cfg.get("mimic_author", False)

    device    = get_device()
    amp_dtype, use_scaler = get_amp_settings(device)
    scaler    = torch.amp.GradScaler(device=device.type, enabled=use_scaler)

    print(f"Device     : {device}  |  AMP dtype: {amp_dtype}  |  Scaler: {use_scaler}")
    print(f"Mimic mode : {mimic_author}")

    model_type  = cfg.get("model_type", "baseline")
    num_classes = cfg.get("num_classes", 15)
    pretrained  = cfg.get("pretrained", True)

    model = build_model(model_type, num_classes, pretrained,
                        mimic_author=mimic_author).to(device)
    print(f"Model      : {model_type}  |  params: "
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if cfg.get("compile", False) and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:
            print(f"torch.compile: skipped ({e})")

    smoke = args.smoke_test
    train_loader = make_dataloader(cfg, "train", smoke_test=smoke)
    val_loader   = make_dataloader(cfg, "val",   smoke_test=False)

    if len(train_loader.dataset) == 0:
        print("ERROR: Training dataset is empty. "
              "Run: python -m src.preprocess.build_dataset")
        sys.exit(1)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.get("label_smoothing", 0.0 if mimic_author else 0.1)
    )

    base_lr                = cfg.get("lr", 1e-4)
    backbone_lr_multiplier = cfg.get("backbone_lr_multiplier", 1.0 if mimic_author else 0.1)
    weight_decay           = cfg.get("weight_decay", 0.0 if mimic_author else 5e-4)

    # Discriminative LRs (or single LR in mimic mode)
    if hasattr(model, "encoder") and hasattr(model, "head") and backbone_lr_multiplier != 1.0:
        param_groups = [
            {"params": model.encoder.parameters(), "lr": base_lr * backbone_lr_multiplier},
            {"params": model.head.parameters(),    "lr": base_lr},
        ]
        print(f"LR         : backbone={base_lr * backbone_lr_multiplier:.2e}, "
              f"head={base_lr:.2e}")
    else:
        param_groups = [{"params": model.parameters(), "lr": base_lr}]
        print(f"LR         : {base_lr:.2e} (all layers)")

    optimizer = torch.optim.Adam(
        param_groups,
        weight_decay=weight_decay,
    )

    epochs = cfg.get("epochs", 100 if mimic_author else 60)

    # Scheduler
    scheduler, scheduler_type = build_scheduler(cfg, optimizer, epochs)
    print(f"Scheduler  : {scheduler_type}  |  weight_decay={weight_decay}")

    checkpoint_dir  = cfg.get("checkpoint_dir", "experiments/baseline/checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    early_stop_pat = cfg.get("early_stop_patience", 10)
    grad_accum     = cfg.get("grad_accum_steps", 1)

    best_val_top1 = 0.0
    no_improve    = 0
    history       = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp_dtype,
            grad_accum_steps=grad_accum, mimic_author=mimic_author,
        )
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, amp_dtype)

        # Step scheduler
        if scheduler_type == "plateau":
            scheduler.step(val_loss)   # ReduceLROnPlateau needs the metric
        else:
            scheduler.step()           # CosineAnnealing steps every epoch

        elapsed = time.time() - t0

        history["train_acc"].append(tr_acc["top1"])
        history["val_acc"].append(val_acc["top1"])
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {tr_loss:.4f} top1 {tr_acc['top1']:.1f}% | "
            f"val loss {val_loss:.4f} top1 {val_acc['top1']:.1f}% "
            f"top2 {val_acc['top2']:.1f}% top3 {val_acc['top3']:.1f}% | "
            f"{elapsed:.1f}s"
        )

        if val_acc["top1"] > best_val_top1:
            best_val_top1 = val_acc["top1"]
            no_improve = 0
            ckpt_path = os.path.join(checkpoint_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_top1": best_val_top1,
                "cfg": cfg,
            }, ckpt_path)
            print(f"  ✓ New best val top-1: {best_val_top1:.2f}% → {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= early_stop_pat and not smoke:
                print(f"Early stopping: {no_improve} epochs without improvement.")
                break

    # final checkpoint
    last_path = os.path.join(checkpoint_dir, "last.pt")
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "val_top1": val_acc["top1"],
        "cfg": cfg,
    }, last_path)

    # save history + figures
    history_path = os.path.join(checkpoint_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    fig_dir = cfg.get("figure_dir", os.path.join(checkpoint_dir, "figures"))
    save_accuracy_bar(
        history,
        os.path.join(fig_dir, "accuracy_curve.png"),
        title=f"{model_type}{'[mimic]' if mimic_author else ''} — Accuracy",
    )

    print(f"\nDone. Best val top-1: {best_val_top1:.2f}%")
    print(f"Run: python src/evaluate.py --checkpoint {os.path.join(checkpoint_dir, 'best.pt')}")


if __name__ == "__main__":
    main()
