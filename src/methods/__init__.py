"""One file per method. `train.py` holds the loop and never learns about them.

A **method** is an answer to two questions and nothing else:

    datasets(cfg)                 -> (train_set, val_set)
    model(cfg, train_set, n_cls)  -> an nn.Module

plus `eval_set(cfg, csv, classes)` so `evaluate.py` can score any method,
`READS`, the config keys it acts on, and optionally `check(cfg)`, which says
why a config cannot start yet.

Everything that is the same for every run -- the epoch loop, resume, the
ledger row, early stopping, the figure, the seed, the saturation watch -- lives
once in `train.py`. Adding a method means adding a ~30-line file here and
naming it in a config. It must never mean editing `train.py`.

**Why it is split this way and not the other way.** Copying `train.py` per
method would put the resume logic and the early-stopping rule in 46 places.
When the `channels_last` bug turned up on 2026-10-15, one fix covered
everything; with 46 copies that is 46 fixes and no way to know which was
missed. It would also quietly break rule 2: two methods could differ in their
stopping rule without anyone intending it, and the ledger would read as though
they differed in the thing under test.

So: **separated per method, shared per concern.**
"""
import importlib

__all__ = ["load", "KNOWN"]

KNOWN = ("video_gru", "cached_gru", "cached_head")


def load(name):
    """The method module called `name`, with a readable error if it is not there."""
    if not name:
        raise ValueError(
            f"a config must name a method. Known: {', '.join(KNOWN)}")
    try:
        return importlib.import_module(f"src.methods.{name}")
    except ImportError as err:
        raise ValueError(
            f"no method {name!r} in src/methods/. Known: "
            f"{', '.join(KNOWN)}. ({err})")
