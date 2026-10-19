"""Will a run fit on the card, and how long will an epoch take? (17.2)

Builds the run exactly as train.py does (same config, loaders, model,
optimizer, train_epoch) and trains a few real batches. Nothing is saved and
no row is written.

    python -m tools.memprobe --exp B07_regnety_16gf
    python -m tools.memprobe --family B

One warm-up batch first: the first step compiles kernels and would make the
timing look worse than it is. Peak memory counts the warm-up too.
"""
import argparse
import time

import torch
import torch.nn as nn

from src import methods
from src.engine import make_scaler, train_epoch
from src.experiments import registry as R
from src.train import _make_loaders, _make_optimizer
from src.utils.device import get_device
from src.utils.seed import set_seed

GB = 1024 ** 3
# share of the card the allocator may reserve. Measured 2026-10-19: B07 at
# batch 4 reserved 10.1 of 11.6 GB (87%), and with the desktop's share on
# top the Arc spilled into system memory: 7.0 s a clip against 0.17 at
# batch 2. 80% leaves room for what else is on the card
FITS = 0.8


def _memory(device):
    """torch.cuda or torch.xpu, whichever this device is; None on the CPU."""
    if device.type in ("cuda", "xpu"):
        return getattr(torch, device.type)
    return None


def _out_of_memory(err):
    return "out of memory" in str(err).lower()


def probe(entry, split="author", batches=3, overrides=None, device=None):
    """One run's peak memory and seconds per batch, as a dict."""
    cfg = R.resolve(entry, split, overrides)
    set_seed(cfg["seed"], deterministic=True)
    device = device or get_device()
    mem = _memory(device)
    method = methods.load(cfg["method"])
    train_loader, _, classes, train_set = _make_loaders(cfg, method)

    found = {"id": entry["id"], "batch": cfg["batch_size"],
             "accum": cfg["accum"], "steps": len(train_loader),
             "peak_gb": None, "reserved_gb": None, "total_gb": None,
             "sec_per_batch": None, "fits": None}
    model = None
    optimizer = None
    try:
        if mem is not None:
            # the XPU refuses to reset its counters before its first
            # allocation ("Invalid device argument"); one number wakes it
            torch.zeros(1, device=device)
            mem.empty_cache()
            mem.reset_peak_memory_stats(device)
        model = method.model(cfg, train_set, len(classes)).to(device)
        found["params_m"] = sum(p.numel() for p in model.parameters()) / 1e6
        optimizer = _make_optimizer(cfg, model)
        scaler = make_scaler(device)
        criterion = nn.CrossEntropyLoss()

        train_epoch(model, train_loader, optimizer, criterion, device,
                    scaler=scaler, accum=cfg["accum"], max_batches=1)
        if mem is not None:
            mem.synchronize(device)
        started = time.time()
        train_epoch(model, train_loader, optimizer, criterion, device,
                    scaler=scaler, accum=cfg["accum"], max_batches=batches)
        if mem is not None:
            mem.synchronize(device)
        found["sec_per_batch"] = (time.time() - started) / batches
        found["fits"] = True
        if mem is not None:
            found["peak_gb"] = mem.max_memory_allocated(device) / GB
            found["reserved_gb"] = mem.max_memory_reserved(device) / GB
            total = mem.get_device_properties(device).total_memory / GB
            found["total_gb"] = total
            found["fits"] = found["reserved_gb"] <= FITS * total
    except (RuntimeError, MemoryError) as err:
        if not _out_of_memory(err):
            raise
        found["fits"] = False
        found["error"] = str(err).splitlines()[0][:120]
    finally:
        del model
        del optimizer
        if mem is not None:
            mem.empty_cache()
    return found


def describe(f):
    """One line a person can read."""
    if f["fits"] is False and f["peak_gb"] is None:
        return (f"{f['id']:20s} DOES NOT FIT at batch {f['batch']}: "
                f"{f.get('error', '')}  -> try batch 2 x accum 2")
    parts = [f"{f['id']:20s}"]
    if f["peak_gb"] is not None:
        parts.append(f"peak {f['peak_gb']:5.1f} / {f['total_gb']:.1f} GB "
                     f"(reserved {f['reserved_gb']:.1f})")
    epoch = f["sec_per_batch"] * f["steps"] / 60
    parts.append(f"{f['sec_per_batch']:5.2f} s/batch  ~{epoch:4.1f} min/epoch "
                 f"training")
    parts.append(f"{f['params_m']:5.1f}M params")
    if f["fits"]:
        parts.append("fits")
    else:
        parts.append("TOO CLOSE: try batch 2 x accum 2")
    return "  ".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", nargs="+", default=[], help="IDs to probe")
    ap.add_argument("--family", help="every ID in one family")
    ap.add_argument("--split", default="author", choices=tuple(R.SPLITS))
    ap.add_argument("--batches", type=int, default=3)
    args = ap.parse_args(argv)

    ids = list(args.exp)
    if args.family:
        for entry in R.entries(args.family):
            ids.append(entry["id"])
    if not ids:
        ap.error("give --exp or --family")
    for exp_id in ids:
        entry = R.get(exp_id)
        problem = R.status(entry)
        if problem != "ready":
            print(f"{exp_id:20s} not ready: {problem}")
            continue
        print(describe(probe(entry, args.split, args.batches)), flush=True)


if __name__ == "__main__":
    main()
