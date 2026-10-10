"""Crash-safe checkpointing, for training through load shedding.

The desktop has no battery, so a power cut kills the process mid-instruction.
Three things to get right: saves that can't be left half-written, checkpoints
you can actually resume from, and results that don't live only in the terminal.
"""
import json
import os
import random
import time

import torch

__all__ = [
    "atomic_save",
    "atomic_write_json",
    "save_resume",
    "load_resume",
    "append_result",
    "read_results",
]


# atomic primitives

def _replace_atomically(tmp_path: str, path: str) -> None:
    os.replace(tmp_path, path)
    # also fsync the directory so the rename itself lands; Windows won't let
    # you open one, hence the guard
    try:
        fd = os.open(os.path.dirname(path) or ".", os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_save(obj, path: str) -> str:
    """torch.save that can't leave a truncated file behind."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"          # same directory, or os.replace isn't atomic
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    _replace_atomically(tmp, path)
    return path


def atomic_write_json(obj, path: str, indent: int = 2) -> str:
    """Same guarantee for history, metrics, configs."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent)
        fh.flush()
        os.fsync(fh.fileno())
    _replace_atomically(tmp, path)
    return path


# resumable training state

def _rng_state() -> dict:
    state = {"python": random.getstate(), "torch": torch.get_rng_state()}
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    return state


def _restore_rng(state) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu")
                            else state["torch"])
    if "numpy" in state:
        try:
            import numpy as np
            np.random.set_state(state["numpy"])
        except ImportError:
            pass


def _state_of(thing):
    """An optimizer/scheduler/scaler's state, or None if there isn't one."""
    if thing is None:
        return None
    return thing.state_dict()


def save_resume(
    path: str,
    *,
    epoch: int,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    history=None,
    best_val_top1: float = 0.0,
    no_improve: int = 0,
    cfg=None,
    **extra,
):
    """Everything needed to continue this run. `epoch` is the last one finished.

    Weights alone aren't enough: without Adam's moments, the scheduler position
    and the early-stop counter you get a different run that happens to start
    warm.
    """
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": _state_of(optimizer),
        "scheduler_state": _state_of(scheduler),
        "scaler_state": _state_of(scaler),
        "history": history or {},
        "best_val_top1": best_val_top1,
        "no_improve": no_improve,
        "rng": _rng_state(),
        "cfg": cfg,
        "saved_at": time.time(),
        **extra,
    }
    return atomic_save(payload, path)


def load_resume(path: str, model=None, optimizer=None, scheduler=None,
                scaler=None, map_location="cpu", restore_rng: bool = True):
    """Load a save_resume checkpoint, restoring whatever is passed in.

    Returns None if there's nothing to resume, so callers can write
    `load_resume(p, ...) or {}` and treat first and resumed runs alike.
    """
    if not path or not os.path.exists(path):
        return None
    # weights_only=False: the payload holds the cfg dict and RNG tuples, which
    # the restricted unpickler rejects. Written by this run, on this machine.
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and payload.get("model_state") is not None:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        saved = len(payload["optimizer_state"].get("param_groups") or [])
        live = len(optimizer.param_groups)
        if saved != live:
            # changing backbone_lr splits one group into two; torch's own
            # error for this does not say which knob caused it
            raise ValueError(
                f"{path} was saved with {saved} parameter group(s) and this "
                f"run builds {live}. The optimizer config changed "
                f"(backbone_lr); this run cannot resume that one. Use --fresh.")
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng:
        _restore_rng(payload.get("rng"))
    return payload


# the results registry

RESULTS_PATH = os.path.join("experiments", "results.jsonl")


def append_result(row: dict, path: str = RESULTS_PATH) -> str:
    """Append one finished run, durably.

    JSON Lines, not a JSON array: appending needs no read of the file, so a cut
    can damage at most the last line. An array would be rewritten whole every
    time, which is the operation a power cut destroys.
    """
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {"recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    with open(path, "ab+") as fh:
        # terminate a line a cut left unfinished, or this result gets glued
        # onto the damaged one and both are lost
        fh.seek(0, os.SEEK_END)
        if fh.tell():
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b"\n":
                fh.write(b"\n")
        fh.write((json.dumps(row, default=str) + "\n").encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
    return path


def read_results(path: str = RESULTS_PATH) -> list:
    """Every recorded run. Skips a line a power cut cut in half."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
