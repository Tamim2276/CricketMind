"""Keep the tests out of the project tree.

`train()` writes its figure into `notes/figures/runs` by default, and that
directory is one git keeps. Without this, every test that calls `train()`
drops a PNG into the repo -- which is exactly what happened the first time.
"""
import pytest

from src import train as T


@pytest.fixture(autouse=True)
def figures_go_to_tmp(tmp_path_factory, monkeypatch):
    kept = tmp_path_factory.mktemp("figures")
    monkeypatch.setitem(T.DEFAULTS, "figures", str(kept))
    return str(kept)
