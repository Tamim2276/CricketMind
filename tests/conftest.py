"""Keep the tests out of the project tree, and out of each other's way.

`train()` writes its figure into `notes/figures/runs` by default, and that
directory is one git keeps. Without this, every test that calls `train()`
drops a PNG into the repo -- which is exactly what happened the first time.

`train()` also switches torch to deterministic kernels for the whole
process. Each test gets torch back the way it found it.
"""
import pytest
import torch

from src import train as T


@pytest.fixture(autouse=True)
def figures_go_to_tmp(tmp_path_factory, monkeypatch):
    kept = tmp_path_factory.mktemp("figures")
    monkeypatch.setitem(T.DEFAULTS, "figures", str(kept))
    return str(kept)


@pytest.fixture(autouse=True)
def deterministic_mode_is_reset():
    was = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    yield
    torch.use_deterministic_algorithms(was, warn_only=warn)
