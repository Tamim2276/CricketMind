"""The epoch loop, built so an interrupted run costs one epoch.

Three decisions worth stating, because all three are about the power going
off rather than about accuracy:

**`save_resume` happens before any `break`.** Early stopping decides to stop
*after* an epoch whose state is already on disk, so the last thing that
happened is always recorded.

**`resume.pt` is deleted when the run finishes.** Its presence therefore means
exactly one thing -- this run was interrupted -- and never "a finished run
left a file behind".

**Resuming is the default, not a flag.** When the power comes back you want to
press up-arrow and hit enter, not remember `--resume`. `--fresh` is there for
the rarer case of starting over on purpose.
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.augment import build_transform
from src.data.dataset import CricShotDataset
from src.engine import evaluate, make_scaler, train_epoch
from src.models.encoder import DEFAULT_BACKBONE
from src.models.model import build_model
from src.utils.checkpoint import (append_result, atomic_save,
                                  atomic_write_json, load_resume, save_resume)
from src.utils import progress
from src.utils.device import describe, get_device
from src.utils.seed import make_generator, seed_worker, set_seed

__all__ = ["train", "DEFAULTS", "load_config", "run_dir"]

DEFAULTS = {
    "name": "run",
    "split": os.path.join("data", "splits_grouped"),
    "variant": "box",
    "processed": os.path.join("data", "processed"),
    "backbone": DEFAULT_BACKBONE,
    "pretrained": True,
    "freeze": False,
    "hidden": 128,
    "dense": 1024,
    "dropout": 0.0,
    "pool": "avg",            # "flatten" is what the authors did
    "flip": 0.0,              # horizontal flip probability on train clips
    "optimizer": "adam",
    "lr": 1e-4,
    "weight_decay": 0.0,
    "scheduler": "none",      # or "plateau"
    "lr_factor": 0.1,
    "lr_patience": 4,
    "min_lr": 0.0,
    "batch_size": 6,          # 6.82 GB in bf16, measured; 8 is an OOM
    "accum": 1,
    "epochs": 30,
    "patience": 8,
    "workers": 0,
    "seed": 42,
    "limit_clips": None,      # a smoke-test subset, taken across all classes
    "max_train_batches": None,
    "max_val_batches": None,
    "out": "experiments",
}


def load_config(path: str) -> dict:
    """A YAML config on top of the defaults."""
    import yaml
    with open(path, encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        loaded = {}
    unknown = []
    for key in loaded:
        if key not in DEFAULTS:
            unknown.append(key)
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return loaded


def run_dir(cfg: dict) -> str:
    return os.path.join(cfg["out"], cfg["name"])


def _subset_across_classes(dataset, n: int):
    """A smoke-test subset with every class in it.

    The split CSVs are ordered by class, so `range(n)` would be one class and
    100% accuracy on it would mean nothing. Found the hard way in 6.1.
    """
    by_class = {}
    for i, item in enumerate(dataset.items):
        label = item[1]
        by_class.setdefault(label, []).append(i)

    per_class = max(1, n // max(1, len(by_class)))
    chosen = []
    for label in sorted(by_class):
        chosen.extend(by_class[label][:per_class])
    return Subset(dataset, chosen)


def _make_loaders(cfg):
    train_set = CricShotDataset(os.path.join(cfg["split"], "train.csv"),
                                variant=cfg["variant"],
                                processed=cfg["processed"],
                                transform=build_transform(cfg["flip"]))
    val_set = CricShotDataset(os.path.join(cfg["split"], "val.csv"),
                              variant=cfg["variant"],
                              processed=cfg["processed"])
    classes = train_set.classes

    if cfg["limit_clips"]:
        train_set = _subset_across_classes(train_set, cfg["limit_clips"])
        val_set = _subset_across_classes(val_set, cfg["limit_clips"])

    # drop_last, because BatchNorm1d raises on a batch of one in train mode
    train_loader = DataLoader(train_set, batch_size=cfg["batch_size"],
                              shuffle=True, drop_last=True,
                              num_workers=cfg["workers"],
                              worker_init_fn=seed_worker,
                              generator=make_generator(cfg["seed"]))
    val_loader = DataLoader(val_set, batch_size=cfg["batch_size"],
                            shuffle=False, num_workers=cfg["workers"])
    return train_loader, val_loader, classes


def _make_optimizer(cfg, model):
    name = cfg["optimizer"].lower()
    params = []
    for p in model.parameters():
        if p.requires_grad:
            params.append(p)
    if name == "adam":
        return torch.optim.Adam(params, lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(params, lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg["weight_decay"])
    raise ValueError(f"unknown optimizer: {cfg['optimizer']}")


def _make_scheduler(cfg, optimizer):
    """Epoch-level only. ReduceLROnPlateau steps on the validation metric, not
    on the optimizer, so it lives in the loop rather than inside train_epoch."""
    name = str(cfg["scheduler"]).lower()
    if name in ("none", "", "null"):
        return None
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg["lr_factor"],
            patience=cfg["lr_patience"], min_lr=cfg["min_lr"])
    raise ValueError(f"unknown scheduler: {cfg['scheduler']}")


def _ticker(label: str, quiet: bool):
    """A callback for train_epoch/evaluate that redraws one progress line."""
    if quiet or not progress.is_tty():
        return None

    state = {"start": time.time(), "loss": 0.0, "n": 0}

    def tick(i, total, loss):
        state["loss"] += loss
        state["n"] += 1
        done = i + 1
        gone = time.time() - state["start"]
        rate = done / max(gone, 1e-9)
        left = (total - done) / max(rate, 1e-9)
        tail = (f"loss {state['loss'] / state['n']:6.4f}  "
                f"~{left / 60:4.1f} min left")
        sys.stdout.write(progress.line(done, total, tail, prefix=label))
        sys.stdout.flush()

    return tick


def _clear_line():
    if progress.is_tty():
        sys.stdout.write(chr(13) + " " * 100 + chr(13))


def train(config=None, fresh: bool = False, quiet: bool = False) -> dict:
    """Run one experiment. Returns the ledger row that was recorded."""
    cfg = dict(DEFAULTS)
    if config:
        cfg.update(config)

    set_seed(cfg["seed"])
    device = get_device()
    out = run_dir(cfg)
    os.makedirs(out, exist_ok=True)
    resume_path = os.path.join(out, "resume.pt")
    best_path = os.path.join(out, "best.pt")

    train_loader, val_loader, classes = _make_loaders(cfg)
    model = build_model(num_classes=len(classes), backbone=cfg["backbone"],
                        pretrained=cfg["pretrained"], freeze=cfg["freeze"],
                        hidden=cfg["hidden"], dense=cfg["dense"],
                        dropout=cfg["dropout"], pool=cfg["pool"]).to(device)
    optimizer = _make_optimizer(cfg, model)
    scheduler = _make_scheduler(cfg, optimizer)
    scaler = make_scaler(device)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_top1 = 0.0
    no_improve = 0
    start_epoch = 1
    resumed = False

    if fresh and os.path.exists(resume_path):
        os.remove(resume_path)
    if os.path.exists(resume_path):
        state = load_resume(resume_path, model, optimizer, scheduler,
                            scaler=scaler)
        start_epoch = state["epoch"] + 1
        history = state.get("history") or []
        best_top1 = state.get("best_val_top1", 0.0)
        no_improve = state.get("no_improve", 0)
        resumed = True

    if not quiet:
        print(describe())
        print(f"{cfg['name']}: {len(train_loader.dataset)} train, "
              f"{len(val_loader.dataset)} val, {len(classes)} classes, "
              f"batch {cfg['batch_size']} x accum {cfg['accum']}")
        if resumed:
            print(f"RESUMED from {resume_path} -- starting at epoch "
                  f"{start_epoch} with best {best_top1:.2%}")
        print()

    started = time.time()
    stopped = "completed"
    epoch = start_epoch - 1

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        tick = time.time()
        try:
            tr = train_epoch(model, train_loader, optimizer, criterion, device,
                             scaler=scaler, accum=cfg["accum"],
                             max_batches=cfg["max_train_batches"],
                             on_batch=_ticker(f"  epoch {epoch:3d} train ", quiet))
            va = evaluate(model, val_loader, criterion, device,
                          max_batches=cfg["max_val_batches"],
                          on_batch=_ticker(f"  epoch {epoch:3d}   val ", quiet))
            _clear_line()
        except torch.OutOfMemoryError:
            # an XPU OOM poisons the process: every later allocation fails
            # too, so save what we have and let the next run resume
            save_resume(resume_path, epoch=epoch - 1, model=model,
                        optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                        history=history, best_val_top1=best_top1,
                        no_improve=no_improve, cfg=cfg)
            raise

        if scheduler is not None:
            scheduler.step(va.top1)

        improved = va.top1 > best_top1
        if improved:
            best_top1 = va.top1
            no_improve = 0
            atomic_save({"model_state": model.state_dict(), "cfg": cfg,
                         "epoch": epoch, "val_top1": va.top1,
                         "classes": classes}, best_path)
        else:
            no_improve += 1

        history.append({"epoch": epoch, "train_loss": tr.loss,
                        "train_top1": tr.top1, "val_loss": va.loss,
                        "val_top1": va.top1, "val_top3": va.top3,
                        "lr": optimizer.param_groups[0]["lr"],
                        "secs": time.time() - tick})

        # before the break, always: an early stop must not lose the epoch
        save_resume(resume_path, epoch=epoch, model=model,
                    optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    history=history, best_val_top1=best_top1,
                    no_improve=no_improve, cfg=cfg)

        if not quiet:
            mark = "  *" if improved else ""
            lr = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch:3d}  train {tr.loss:6.4f}/{tr.top1:6.2%}  "
                  f"val {va.loss:6.4f}/{va.top1:6.2%}  top3 {va.top3:6.2%}  "
                  f"lr {lr:.1e}  {time.time() - tick:5.1f}s{mark}", flush=True)

        if no_improve >= cfg["patience"]:
            stopped = f"early stop after {no_improve} epochs without a gain"
            break

    atomic_write_json(history, os.path.join(out, "history.json"))

    row = {"name": cfg["name"], "split": cfg["split"], "variant": cfg["variant"],
           "backbone": cfg["backbone"], "frozen": cfg["freeze"],
           "best_val_top1": best_top1, "epochs_run": epoch,
           "stopped": stopped, "minutes": (time.time() - started) / 60,
           "device": str(device), "cfg": cfg}
    append_result(row, os.path.join(cfg["out"], "results.jsonl"))

    # last, and only on a clean finish: resume.pt existing means interrupted
    if os.path.exists(resume_path):
        os.remove(resume_path)

    if not quiet:
        print(f"\n  {stopped}; best val top-1 {best_top1:.2%} "
              f"in {row['minutes']:.1f} min")
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", help="YAML, applied on top of the defaults")
    ap.add_argument("--name")
    ap.add_argument("--split")
    ap.add_argument("--variant", choices=("box", "seg"))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch_size", type=int)
    ap.add_argument("--accum", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--limit_clips", type=int)
    ap.add_argument("--pool", choices=("avg", "flatten"))
    ap.add_argument("--flip", type=float)
    ap.add_argument("--freeze", action="store_true", default=None)
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any resume.pt and start over")
    args = ap.parse_args(argv)

    cfg = {}
    if args.config:
        cfg.update(load_config(args.config))
    for key, value in vars(args).items():
        if key in ("config", "fresh"):
            continue
        if value is not None:
            cfg[key] = value

    train(cfg, fresh=args.fresh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
