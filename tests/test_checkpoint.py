"""Does a run interrupted by a power cut really continue, or just restart warm?

The load-shedding claim is strong -- that a resumed run *is* the run that was
interrupted -- so the test is correspondingly strong: train one model straight
through, train another in two halves with a simulated kill in between, and
require the weights to match exactly. Anything forgotten in the checkpoint
(Adam's moments, the scheduler's position, the RNG) shows up as a difference.
"""
import json
import os
import random

import pytest

torch = pytest.importorskip("torch")

from src.utils.checkpoint import (          # noqa: E402
    append_result,
    atomic_save,
    atomic_write_json,
    load_resume,
    read_results,
    save_resume,
)

EPOCHS = 6
CUT_AFTER = 3


def _fresh(seed=0):
    """A tiny deterministic training setup: model, optimizer, scheduler, data."""
    torch.manual_seed(seed)
    random.seed(seed)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(32, 8, generator=g)
    y = torch.randint(0, 3, (32,), generator=g)
    return model, opt, sched, x, y


def _train_epochs(model, opt, sched, x, y, n):
    """n epochs, with a dropout-free model but an RNG-dependent batch order."""
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(n):
        order = torch.randperm(len(x))        # consumes global RNG on purpose
        for i in range(0, len(x), 8):
            idx = order[i:i + 8]
            opt.zero_grad()
            loss_fn(model(x[idx]), y[idx]).backward()
            opt.step()
        sched.step()
    return model


def _flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def test_resumed_run_equals_uninterrupted_run(tmp_path):
    # (a) the run that never loses power
    model, opt, sched, x, y = _fresh()
    _train_epochs(model, opt, sched, x, y, EPOCHS)
    expected = _flat(model)

    # (b) the same run, killed after CUT_AFTER epochs
    model, opt, sched, x, y = _fresh()
    _train_epochs(model, opt, sched, x, y, CUT_AFTER)
    ckpt = tmp_path / "resume.pt"
    save_resume(str(ckpt), epoch=CUT_AFTER, model=model, optimizer=opt,
                scheduler=sched, history={"val_acc": [1.0, 2.0, 3.0]},
                best_val_top1=3.0, no_improve=1, cfg={"model_type": "test"})
    del model, opt, sched                      # the process is gone

    # (c) power is back: brand-new objects, a different seed, then resume
    model, opt, sched, x, y = _fresh(seed=999)
    payload = load_resume(str(ckpt), model=model, optimizer=opt, scheduler=sched)
    assert payload["epoch"] == CUT_AFTER
    assert payload["best_val_top1"] == 3.0
    assert payload["no_improve"] == 1
    assert payload["history"]["val_acc"] == [1.0, 2.0, 3.0]
    assert payload["cfg"]["model_type"] == "test"

    _train_epochs(model, opt, sched, x, y, EPOCHS - CUT_AFTER)

    torch.testing.assert_close(_flat(model), expected, rtol=0, atol=0)


def test_without_optimizer_state_the_run_would_diverge(tmp_path):
    """The guard on the test above: prove it can fail.

    If this passed too, the first test would prove nothing -- it would be
    measuring something insensitive to the state we bothered to save.
    """
    model, opt, sched, x, y = _fresh()
    _train_epochs(model, opt, sched, x, y, EPOCHS)
    expected = _flat(model)

    model, opt, sched, x, y = _fresh()
    _train_epochs(model, opt, sched, x, y, CUT_AFTER)
    ckpt = tmp_path / "weights_only.pt"
    atomic_save({"model_state": model.state_dict()}, str(ckpt))

    # resume the old way: weights, nothing else
    model, opt, sched, x, y = _fresh(seed=999)
    model.load_state_dict(torch.load(str(ckpt), weights_only=True)["model_state"])
    _train_epochs(model, opt, sched, x, y, EPOCHS - CUT_AFTER)

    assert not torch.allclose(_flat(model), expected, rtol=1e-4, atol=1e-6), \
        "weights-only resume matched, so this test cannot detect lost state"


def test_a_cut_during_the_write_leaves_the_old_checkpoint_loadable(tmp_path, monkeypatch):
    ckpt = tmp_path / "best.pt"
    atomic_save({"epoch": 7, "val_top1": 88.5}, str(ckpt))

    def die_midway(*_a, **_k):
        raise KeyboardInterrupt("power cut")

    monkeypatch.setattr(torch, "save", die_midway)
    with pytest.raises(KeyboardInterrupt):
        atomic_save({"epoch": 8, "val_top1": 89.1}, str(ckpt))

    # the good checkpoint is untouched, and still loads
    survived = torch.load(str(ckpt), weights_only=True)
    assert survived["epoch"] == 7
    assert survived["val_top1"] == 88.5


def test_load_resume_returns_none_when_there_is_nothing_to_resume(tmp_path):
    assert load_resume(str(tmp_path / "nope.pt")) is None
    assert load_resume("") is None
    assert (load_resume(str(tmp_path / "nope.pt")) or {}) == {}


def test_history_json_is_rewritten_atomically(tmp_path):
    path = tmp_path / "history.json"
    atomic_write_json({"val_acc": [10.0]}, str(path))
    atomic_write_json({"val_acc": [10.0, 20.0]}, str(path))
    assert json.loads(path.read_text())["val_acc"] == [10.0, 20.0]
    assert not os.path.exists(str(path) + ".tmp")


def test_results_registry_survives_a_truncated_append(tmp_path):
    path = tmp_path / "results.jsonl"
    append_result({"run": "mimic_author", "test_top1": 89.09}, str(path))
    append_result({"run": "transformer", "test_top1": 75.93}, str(path))

    # a power cut in the middle of the third append
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"run": "3d_cnn", "test_top')

    rows = read_results(str(path))
    assert [r["run"] for r in rows] == ["mimic_author", "transformer"]
    assert rows[0]["test_top1"] == 89.09
    assert "recorded_at" in rows[0]

    # and the next run still appends cleanly on top of the damaged line
    append_result({"run": "swin3d", "test_top1": 91.2}, str(path))
    assert [r["run"] for r in read_results(str(path))] == [
        "mimic_author", "transformer", "swin3d"
    ]


def test_read_results_on_a_missing_file_is_empty_not_an_error(tmp_path):
    assert read_results(str(tmp_path / "never_written.jsonl")) == []
