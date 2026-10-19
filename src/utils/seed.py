"""One seed for every random number generator in play.

Three libraries roll their own dice -- python's `random`, numpy, and torch --
and seeding one does nothing to the others. Weight init comes from torch, most
augmentation from numpy, and shuffling from whichever the code happened to use.
"""
import os
import random
from typing import Optional

import numpy as np
import torch

__all__ = ["set_seed", "seed_worker", "make_generator", "DEFAULT_SEED"]

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = False) -> int:
    """Seed python, numpy and torch. Returns the seed, to log it.

    `deterministic=True` also forces deterministic kernels; an op with no
    deterministic version warns instead of failing. Without it the flat GRU
    gave three different losses from one seed on the Arc (11.1), and on the
    cached heads it costs nothing. End to end it costs 1.5-1.7x (17.3), so
    those bases turn it off.

    `False` switches it off, rather than leaving it alone: one process
    trains many runs, and one run's choice must not leak into the next.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)        # covers CPU, CUDA and XPU

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = False    # benchmark picks kernels by timing

    if deterministic:
        # cuBLAS needs this to be deterministic on the RTX; harmless elsewhere
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.use_deterministic_algorithms(False)

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn`. Each worker is a separate process.

    Torch gives workers distinct seeds already, but numpy and `random` don't
    get the same treatment -- so without this, every worker can produce the
    same "random" augmentations.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = DEFAULT_SEED) -> torch.Generator:
    """Generator for a DataLoader's `generator=`, so shuffling is repeatable."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def hashseed_note() -> Optional[str]:
    """Warn if PYTHONHASHSEED isn't pinned. Only matters if you iterate sets."""
    if os.environ.get("PYTHONHASHSEED") is None:
        return ("PYTHONHASHSEED unset: set iteration order varies between "
                "processes. Harmless unless you build label maps from a set.")
    return None
