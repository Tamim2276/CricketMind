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

from src import methods
from src.engine import evaluate, make_scaler, train_epoch
from src.models.encoder import DEFAULT_BACKBONE
from src.plot import FIGURES, plot_run
from src.utils.checkpoint import (append_result, atomic_save,
                                  atomic_write_json, load_resume, save_resume)
from src.utils import progress
from src.utils.device import describe, get_device
from src.utils.seed import make_generator, seed_worker, set_seed

__all__ = ["train", "DEFAULTS", "LOOP_READS", "load_config", "run_dir",
           "split_name"]

DEFAULTS = {
    "name": "run",
    "method": "video_gru",    # a file in src/methods/; see that package

    "split": os.path.join("data", "splits_grouped"),
    "variant": "box",
    "processed": os.path.join("data", "processed"),
    "features": None,         # a cache path reads tensors; None reads video
    "view": "flat",           # which cached view; "flat" is pool="flatten"
    "backbone": DEFAULT_BACKBONE,
    "pretrained": True,
    "freeze": False,
    "hidden": 128,
    "dense": 1024,
    "dropout": 0.0,
    "keras_dense": False,     # Keras' Dense init in the GRU head (7.7)
    "head": "gru",            # the head on cached features; Families A, D, V
    "bidirectional": False,
    "layers": 1,
    "attn_heads": 4,          # transformer heads; only when head is one
    "feature_norm": "none",   # or "layer", LayerNorm before the head (A13)
    "pool": "avg",            # "flatten" is what the authors did
    "flip": 0.0,              # horizontal flip probability on train clips
    "optimizer": "adam",
    "lr": 1e-4,
    "backbone_lr": None,      # None gives the backbone the same lr as the head
    "weight_decay": 0.0,
    "clip_grad": 0.0,         # 0 is off; when on, the epoch line says how often
    "scheduler": "none",      # or "plateau"
    "monitor": "top1",        # or "loss": the authors' callbacks watch val loss (7.7)
    "adam_eps": None,         # None is torch's 1e-8; Keras' Adam uses 1e-7
    "lr_factor": 0.1,
    "lr_patience": 4,
    "min_lr": 0.0,
    "batch_size": 6,          # 6.82 GB in bf16, measured; 8 is an OOM
    "accum": 1,
    "epochs": 30,
    "patience": 8,
    "workers": 0,
    "seed": 42,
    "deterministic": True,    # same seed, same run, bit for bit; off end to end (17.3)
    "limit_clips": None,      # a smoke-test subset, taken across all classes
    "max_train_batches": None,
    "max_val_batches": None,
    "out": "experiments",
    "figures": FIGURES,   # the copy git keeps; null to skip
}

# Keys the loop itself acts on. Every other key must be read by the method,
# or changing it changes nothing but the ledger row.
LOOP_READS = ("name", "method", "split", "seed", "optimizer", "lr",
              "backbone_lr", "weight_decay", "clip_grad", "scheduler",
              "monitor", "adam_eps",
              "lr_factor", "lr_patience", "min_lr", "batch_size", "accum",
              "epochs", "patience", "workers", "deterministic", "limit_clips",
              "max_train_batches", "max_val_batches", "out", "figures")


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


def split_name(path: str) -> str:
    """'data/splits_author' -> 'author'."""
    leaf = os.path.basename(os.path.normpath(path))
    if leaf.startswith("splits_"):
        return leaf[len("splits_"):]
    return leaf


def _row_backbone(cfg: dict) -> str:
    """What made the features. A cached run's own backbone key is unused."""
    if cfg["features"]:
        source = os.path.join(cfg["features"], "source.json")
        if os.path.exists(source):
            with open(source, encoding="utf-8") as fh:
                return json.load(fh).get("backbone", cfg["backbone"])
    return cfg["backbone"]


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


def _make_loaders(cfg, method):
    """The loader plumbing, which is the same for every method.

    The method says *what* the data is; this says how it is batched, shuffled
    and seeded. Nothing here knows whether it is reading video or tensors.
    """
    train_set, val_set = method.datasets(cfg)
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
    return train_loader, val_loader, classes, train_set


def _trainable(module):
    found = []
    for p in module.parameters():
        if p.requires_grad:
            found.append(p)
    return found


def _param_groups(cfg, model):
    """One group, or a smaller lr for the pretrained half.

    The backbone is 20M ImageNet parameters; the head is 65M new ones. One lr
    for both moved the backbone a median 18.65% off its ImageNet weights in
    five epochs -- one bias by 65x its own norm -- and the blown-up features
    saturated 20 of the GRU's 128 units. A 10x smaller lr on the pretrained
    half is the standard fix.

    `backbone_lr: null` keeps the single group, so every earlier result still
    reproduces exactly.
    """
    groups = [{"params": None, "lr": cfg["lr"], "name": "all"}]
    backbone_lr = cfg.get("backbone_lr")
    encoder = getattr(model, "encoder", None)
    head = getattr(model, "head", None)

    if backbone_lr is None or encoder is None or head is None:
        groups[0]["params"] = _trainable(model)
        return groups

    groups = [{"params": _trainable(head), "lr": cfg["lr"], "name": "head"}]
    backbone = _trainable(encoder)
    if backbone:
        groups.append({"params": backbone, "lr": backbone_lr,
                       "name": "backbone"})
    return groups


def _make_optimizer(cfg, model):
    name = cfg["optimizer"].lower()
    groups = _param_groups(cfg, model)
    if name == "adam":
        if cfg.get("adam_eps") is None:
            return torch.optim.Adam(groups, lr=cfg["lr"],
                                    weight_decay=cfg["weight_decay"])
        return torch.optim.Adam(groups, lr=cfg["lr"], eps=cfg["adam_eps"],
                                weight_decay=cfg["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    if name == "sgd":
        return torch.optim.SGD(groups, lr=cfg["lr"], momentum=0.9,
                               weight_decay=cfg["weight_decay"])
    raise ValueError(f"unknown optimizer: {cfg['optimizer']}")


def _saturation(model):
    """(pinned, dead) fractions of the head's hidden units, or (None, None).

    Watched every epoch because it is the failure that cost this project two
    runs: the loss column cannot see it -- epoch 5 of the second run had a
    *rising* top-1 of 55.11% while 20 units sat dead at tanh's rail.
    """
    # a cached run has no encoder, so the head is the model itself
    head = getattr(model, "head", model)
    if not hasattr(head, "pinned_fraction"):
        return None, None
    return head.pinned_fraction(), head.dead_fraction()


def _lr_text(optimizer):
    """The lr column: one number, or head/backbone when they differ."""
    rates = []
    for group in optimizer.param_groups:
        rates.append(group["lr"])
    if len(rates) == 1 or rates[0] == rates[1]:
        return f"{rates[0]:.0e}"
    parts = []
    for rate in rates:
        parts.append(f"{rate:.0e}")
    return "/".join(parts)


def _make_scheduler(cfg, optimizer):
    """Epoch-level only. ReduceLROnPlateau steps on the validation metric, not
    on the optimizer, so it lives in the loop rather than inside train_epoch."""
    name = str(cfg["scheduler"]).lower()
    if name in ("none", "", "null"):
        return None
    if name == "plateau":
        if _watch(cfg) == "loss":
            # Keras' ReduceLROnPlateau: lower is better, by 1e-4 absolute
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=cfg["lr_factor"],
                patience=cfg["lr_patience"], min_lr=cfg["min_lr"],
                threshold=1e-4, threshold_mode="abs")
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=cfg["lr_factor"],
            patience=cfg["lr_patience"], min_lr=cfg["min_lr"])
    raise ValueError(f"unknown scheduler: {cfg['scheduler']}")


def _watch(cfg):
    """What decides the best epoch, the lr drops and the early stop."""
    monitor = str(cfg.get("monitor", "top1")).lower()
    if monitor not in ("top1", "loss"):
        raise ValueError(f"monitor must be top1 or loss, got {monitor!r}")
    return monitor


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

    # deterministic kernels: the same seed gives the same run, bit for bit
    set_seed(cfg["seed"], deterministic=cfg["deterministic"])
    device = get_device()
    out = run_dir(cfg)
    os.makedirs(out, exist_ok=True)
    resume_path = os.path.join(out, "resume.pt")
    best_path = os.path.join(out, "best.pt")
    history_path = os.path.join(out, "history.json")

    method = methods.load(cfg["method"])
    train_loader, val_loader, classes, train_set = _make_loaders(cfg, method)
    model = method.model(cfg, train_set, len(classes)).to(device)
    optimizer = _make_optimizer(cfg, model)
    scheduler = _make_scheduler(cfg, optimizer)
    scaler = make_scaler(device)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_top1 = 0.0                 # of the epoch kept as best.pt
    best_loss = float("inf")        # what monitor: loss compares against
    monitor = _watch(cfg)
    no_improve = 0
    start_epoch = 1
    resumed = False

    if fresh and os.path.exists(resume_path):
        os.remove(resume_path)
    if os.path.exists(resume_path):
        state = load_resume(resume_path, model, optimizer, scheduler,
                            scaler=scaler)
        # the shuffle order too, or epoch N+1 sees epoch 1's order again.
        # Resume files from before 2026-10-20 have none; they keep that
        if state.get("loader_rng") is not None:
            train_loader.generator.set_state(state["loader_rng"])
        start_epoch = state["epoch"] + 1
        history = state.get("history") or []
        best_top1 = state.get("best_val_top1", 0.0)
        best_loss = state.get("best_val_loss", float("inf"))
        no_improve = state.get("no_improve", 0)
        resumed = True

    if not quiet:
        print(describe())
        print(f"{cfg['name']} [{cfg['method']}]: "
              f"{len(train_loader.dataset)} train, "
              f"{len(val_loader.dataset)} val, {len(classes)} classes, "
              f"batch {cfg['batch_size']} x accum {cfg['accum']}")
        if resumed:
            print(f"RESUMED from {resume_path} -- starting at epoch "
                  f"{start_epoch} with best {best_top1:.2%}")
        print()

    stopped = "completed"
    epoch = start_epoch - 1

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        tick = time.time()
        # before this epoch draws its shuffle: what a save of epoch - 1 needs
        epoch_start_rng = train_loader.generator.get_state()
        try:
            tr = train_epoch(model, train_loader, optimizer, criterion, device,
                             scaler=scaler, accum=cfg["accum"],
                             clip_grad=cfg["clip_grad"],
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
                        no_improve=no_improve, cfg=cfg,
                        loader_rng=epoch_start_rng, best_val_loss=best_loss)
            raise

        pinned, dead = _saturation(model)

        if monitor == "loss":
            metric = va.loss
            # Keras' EarlyStopping: any drop counts, min_delta 0
            improved = va.loss < best_loss
        else:
            metric = va.top1
            improved = va.top1 > best_top1
        if scheduler is not None:
            scheduler.step(metric)

        if improved:
            best_top1 = va.top1
            best_loss = va.loss
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
                        "lrs": _lr_text(optimizer),
                        "pinned": pinned, "dead": dead,
                        "grad_norm": tr.grad_norm, "clipped": tr.clipped,
                        "steps": tr.steps,
                        "secs": time.time() - tick})

        # before the break, always: an early stop must not lose the epoch
        save_resume(resume_path, epoch=epoch, model=model,
                    optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    history=history, best_val_top1=best_top1,
                    no_improve=no_improve, cfg=cfg,
                    loader_rng=train_loader.generator.get_state(),
                    best_val_loss=best_loss)
        # and every epoch, not just at the end: a run killed at epoch 13 used
        # to leave its twelve finished epochs only in the terminal scrollback
        atomic_write_json(history, history_path)

        if not quiet:
            if improved:
                mark = "  *"
            else:
                mark = ""
            if pinned is None:
                watch = ""
            else:
                watch = f"pin {pinned:.0%} "
            watch += f"gn {tr.grad_norm:.3g} "
            if tr.clipped > 0:
                watch += f"clip {tr.clipped / max(tr.steps, 1):.0%} "
            print(f"  epoch {epoch:3d}  "
                  f"train loss {tr.loss:6.4f} top1 {tr.top1:6.2%}   "
                  f"val loss {va.loss:6.4f} top1 {va.top1:6.2%} "
                  f"top3 {va.top3:6.2%}   "
                  f"{watch} lr {_lr_text(optimizer)}  "
                  f"{(time.time() - tick) / 60:4.1f}m{mark}",
                  flush=True)

        if no_improve >= cfg["patience"]:
            stopped = f"early stop after {no_improve} epochs without a gain"
            break

    atomic_write_json(history, history_path)

    # every epoch's own time, so a resumed run still counts the part before
    # the cut; wall clock since this process started would lose it
    epoch_secs = 0.0
    for past in history:
        epoch_secs += past.get("secs", 0.0)

    row = {"name": cfg["name"], "method": cfg["method"],
           "split": cfg["split"], "split_name": split_name(cfg["split"]),
           "variant": cfg["variant"],
           "features": cfg["features"], "view": cfg["view"],
           "backbone": _row_backbone(cfg), "frozen": cfg["freeze"],
           "best_val_top1": best_top1, "best_val_loss": best_loss,
           "monitor": monitor, "epochs_run": epoch,
           "stopped": stopped, "minutes": epoch_secs / 60,
           "resumed": resumed,
           "deterministic": torch.are_deterministic_algorithms_enabled(),
           "device": str(device), "cfg": cfg}
    append_result(row, os.path.join(cfg["out"], "results.jsonl"))

    # the figure, before the resume file goes: a run you cannot see is a run
    # you will re-read as a wall of numbers
    try:
        plot_run(out, name=cfg["name"], mirror=cfg["figures"])
    except Exception as err:
        print(f"  (no figure: {err})", flush=True)

    # last, and only on a clean finish: resume.pt existing means interrupted
    if os.path.exists(resume_path):
        os.remove(resume_path)

    if not quiet:
        kept = ""
        if monitor == "loss":
            kept = f" (the epoch with the lowest val loss, {best_loss:.4f})"
        print(f"\n  {stopped}; best val top-1 {best_top1:.2%}{kept} "
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
    ap.add_argument("--backbone_lr", type=float,
                    help="a smaller lr for the pretrained half")
    ap.add_argument("--clip_grad", type=float)
    ap.add_argument("--limit_clips", type=int)
    ap.add_argument("--pool", choices=("avg", "flatten"))
    ap.add_argument("--method", help=f"one of {', '.join(methods.KNOWN)}")
    ap.add_argument("--features", help="a cache path; reads tensors not video")
    ap.add_argument("--view", choices=("flat", "grid", "avg"))
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
