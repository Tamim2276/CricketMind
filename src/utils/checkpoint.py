"""Crash-safe checkpointing, for training through load shedding.

The problem this solves: a desktop has no battery, so a power cut kills the
process instantly, at an arbitrary instruction. Three things go wrong if you
just call ``torch.save`` in the training loop:

1. **A half-written file.** ``torch.save`` streams hundreds of MB. Lose power
   in the middle and ``best.pt`` is a truncated file that will not load -- so
   the cut costs you not only the current epoch but the best model you had
   already earned. ``atomic_save`` writes to a temporary file in the same
   directory and then calls ``os.replace``, which is atomic on NTFS and POSIX
   alike: the old file is intact until the new one is complete, so the
   checkpoint on disk is always loadable.

2. **Nothing to resume from.** A checkpoint holding only ``model_state`` cannot
   continue training: Adam's moment estimates, the LR scheduler's position and
   the early-stopping counter are all gone, and restarting from epoch 1 with a
   warm model is not the same run. ``save_resume`` stores every piece of
   mutable training state, so the restarted run is a continuation rather than
   a new experiment the supervisor cannot compare.

3. **Results that live only in the terminal.** Over 30+ experiments, a number
   you measured and did not write down is a number you have to measure again.
   ``append_result`` appends one JSON object per finished run and fsyncs it, so
   a result survives the moment it is recorded.

Nothing here is specific to a model or a dataset, and nothing here imports a
device: every function takes what it needs. See ``load_resume`` for the one
subtlety, which is ``weights_only``.
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


# ── atomic primitives ────────────────────────────────────────────────────────

def _replace_atomically(tmp_path: str, path: str) -> None:
    """fsync the temp file, then move it into place. Never leaves a partial."""
    os.replace(tmp_path, path)
    # Also fsync the directory, so the rename itself reaches the disk. Best
    # effort: Windows does not allow opening a directory, hence the guard.
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
    """``torch.save`` that cannot leave a truncated file behind.

    The write goes to ``<path>.tmp`` in the same directory -- same directory
    matters, because ``os.replace`` is only atomic within one filesystem.
    """
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    _replace_atomically(tmp, path)
    return path


def atomic_write_json(obj, path: str, indent: int = 2) -> str:
    """Same guarantee for JSON: history, metrics, configs."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent)
        fh.flush()
        os.fsync(fh.fileno())
    _replace_atomically(tmp, path)
    return path


# ── resumable training state ─────────────────────────────────────────────────

def _rng_state() -> dict:
    """Capture every random stream that affects training."""
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
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
    """Write everything needed to continue this exact run after a power cut.

    ``epoch`` is the last epoch that *finished*, so the resumed run starts at
    ``epoch + 1``.
    """
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
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
    """Load a ``save_resume`` checkpoint, restoring any object passed in.

    Returns the raw payload, or ``None`` if there is nothing to resume from --
    so the caller can write ``ckpt = load_resume(p, ...) or {}`` and treat a
    first run and a resumed run the same way.

    ``weights_only=False`` is required and safe here: the payload holds the
    config dict and RNG tuples, which the restricted unpickler rejects, and the
    file was written by this training run on this machine.
    """
    if not path or not os.path.exists(path):
        return None
    payload = torch.load(path, map_location=map_location, weights_only=False)

    if model is not None and payload.get("model_state") is not None:
        model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng:
        _restore_rng(payload.get("rng"))
    return payload


# ── the results registry ─────────────────────────────────────────────────────

RESULTS_PATH = os.path.join("experiments", "results.jsonl")


def append_result(row: dict, path: str = RESULTS_PATH) -> str:
    """Append one finished run to the registry, durably.

    JSON Lines, not CSV or a single JSON array: appending a line needs no read
    of what is already there, so two runs cannot clobber each other and a cut
    mid-append can at worst leave one damaged final line -- ``read_results``
    skips it. A JSON array would have to be rewritten whole every time, which
    is exactly the operation a power cut destroys.

    The one trap, which the test for this caught: a cut mid-append leaves a
    final line with no newline, and a naive append would then glue the *next*
    result onto the damaged one, losing a good number as well as a bad one. So
    terminate the file first if it does not end in a newline. Binary mode,
    because seeking from the end is only well defined there.
    """
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    row = {"recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"), **row}
    with open(path, "ab+") as fh:
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
    """Every recorded run. Silently drops a line a power cut cut in half."""
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
                continue   # truncated final line from an interrupted append
    return rows
