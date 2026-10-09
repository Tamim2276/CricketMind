"""Mostly about interruption. The accuracy path is 6.1 and 6.2's job; what
this file checks is that a run which dies halfway costs one epoch.
"""
import json
import os

import numpy as np
import pytest
import torch

from src import train as T

CLASSES = ["Cover Drive", "Sweep", "Pull"]


@pytest.fixture
def workspace(tmp_path):
    """A tiny real dataset on disk, and a model small enough to train fast."""
    proc = tmp_path / "processed"
    split = tmp_path / "splits"
    split.mkdir()
    rng = np.random.default_rng(0)

    rows = []
    for label, cls in enumerate(CLASSES):
        for k in range(6):
            clip = f"{cls}/vid{label}{k}_0.avi"
            rows.append((clip, cls, label))
            for variant in ("box", "seg"):
                p = proc / variant / clip.replace("/", os.sep)[:-4]
                p.parent.mkdir(parents=True, exist_ok=True)
                # each class gets its own brightness, so it is learnable
                base = 40 + 70 * label
                frames = np.clip(base + rng.integers(0, 20, (4, 8, 8, 3)),
                                 0, 255).astype(np.uint8)
                np.save(str(p) + ".npy", frames)

    for name in ("train", "val"):
        text = "clip,class,label,match\n"
        for clip, cls, label in rows:
            text += f"{clip},{cls},{label},vid{label}\n"
        (split / f"{name}.csv").write_text(text, encoding="utf-8")
    (split / "classes.json").write_text(json.dumps(CLASSES), encoding="utf-8")

    return str(split), str(tmp_path / "exp"), str(proc)


def base_cfg(workspace, **over):
    split, out, proc = workspace
    cfg = {"name": "t", "split": split, "out": out, "processed": proc,
           "pretrained": False, "backbone": "resnet18", "batch_size": 3,
           "epochs": 2, "patience": 99, "lr": 1e-3, "workers": 0}
    cfg.update(over)
    return cfg


# ---- it runs ------------------------------------------------------------

def test_a_run_produces_a_ledger_row_and_a_history(workspace):
    row = T.train(base_cfg(workspace), quiet=True)
    out = T.run_dir(base_cfg(workspace))
    assert row["epochs_run"] == 2
    assert os.path.exists(os.path.join(out, "history.json"))
    assert len(json.load(open(os.path.join(out, "history.json")))) == 2


def test_the_ledger_is_append_only(workspace):
    T.train(base_cfg(workspace, name="a"), quiet=True)
    T.train(base_cfg(workspace, name="b"), quiet=True)
    ledger = os.path.join(workspace[1], "results.jsonl")
    rows = []
    for line in open(ledger, encoding="utf-8"):
        rows.append(json.loads(line))
    assert [r["name"] for r in rows] == ["a", "b"]


def test_the_best_checkpoint_is_written(workspace):
    T.train(base_cfg(workspace), quiet=True)
    best = os.path.join(T.run_dir(base_cfg(workspace)), "best.pt")
    saved = torch.load(best, map_location="cpu", weights_only=False)
    assert saved["classes"] == CLASSES
    assert "model_state" in saved


# ---- the interruption contract -----------------------------------------

def test_a_finished_run_leaves_no_resume_file(workspace):
    """Its presence must mean 'interrupted' and nothing else."""
    T.train(base_cfg(workspace), quiet=True)
    assert not os.path.exists(
        os.path.join(T.run_dir(base_cfg(workspace)), "resume.pt"))


def test_an_interrupted_run_leaves_one_and_resumes_from_it(workspace):
    cfg = base_cfg(workspace, epochs=4)
    out = T.run_dir(cfg)

    real = T.evaluate
    calls = {"n": 0}

    def die_during_epoch_2(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(*a, **k)

    T.evaluate = die_during_epoch_2
    try:
        with pytest.raises(KeyboardInterrupt):
            T.train(cfg, quiet=True)
    finally:
        T.evaluate = real

    assert os.path.exists(os.path.join(out, "resume.pt"))
    state = torch.load(os.path.join(out, "resume.pt"), map_location="cpu",
                       weights_only=False)
    assert state["epoch"] == 1                 # epoch 1 finished and was saved

    row = T.train(cfg, quiet=True)             # same command again
    assert row["epochs_run"] == 4
    history = json.load(open(os.path.join(out, "history.json")))
    assert [h["epoch"] for h in history] == [1, 2, 3, 4]


def test_fresh_throws_the_resume_away(workspace):
    cfg = base_cfg(workspace, epochs=1)
    T.train(cfg, quiet=True)
    resume = os.path.join(T.run_dir(cfg), "resume.pt")
    torch.save({"epoch": 99}, resume)          # pretend an interrupted run
    row = T.train(cfg, fresh=True, quiet=True)
    assert row["epochs_run"] == 1


def test_early_stopping_saves_before_it_breaks(workspace):
    """The stopping epoch must be on disk, not lost to the break."""
    cfg = base_cfg(workspace, epochs=10, patience=1)
    row = T.train(cfg, quiet=True)
    assert "early stop" in row["stopped"]
    history = json.load(open(os.path.join(T.run_dir(cfg), "history.json")))
    assert history[-1]["epoch"] == row["epochs_run"]


# ---- the config ---------------------------------------------------------

def test_a_subset_covers_every_class(workspace):
    """6.1's lesson: range(n) over a class-ordered CSV is one class."""
    from src.data.dataset import CricShotDataset
    split, _, proc = workspace
    full = CricShotDataset(os.path.join(split, "train.csv"), processed=proc)
    subset = T._subset_across_classes(full, 6)
    labels = set()
    for i in range(len(subset)):
        labels.add(subset[i][1])
    assert labels == {0, 1, 2}


def test_an_unknown_config_key_is_refused(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("learning_rate: 0.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        T.load_config(str(bad))


def test_a_config_loads_onto_the_defaults(tmp_path):
    good = tmp_path / "good.yaml"
    good.write_text("name: mimic\nlr: 0.001\nepochs: 5\n", encoding="utf-8")
    cfg = T.load_config(str(good))
    assert cfg == {"name": "mimic", "lr": 0.001, "epochs": 5}


def test_an_unknown_optimizer_is_refused():
    with pytest.raises(ValueError, match="unknown optimizer"):
        T._make_optimizer({"optimizer": "rmsprop", "lr": 1e-3,
                           "weight_decay": 0.0}, torch.nn.Linear(2, 2))


# ---- 6.4: the authors' settings ----------------------------------------

def test_the_mimic_config_loads_and_every_key_is_known():
    cfg = T.load_config(os.path.join("configs", "mimic_author.yaml"))
    for key in cfg:
        assert key in T.DEFAULTS, key


def test_the_mimic_config_matches_the_shipped_keras_model():
    """Each of these was read out of their .keras file, not out of the paper."""
    cfg = T.load_config(os.path.join("configs", "mimic_author.yaml"))
    assert cfg["pool"] == "flatten"        # GRU kernel is (62720, 384)
    assert cfg["hidden"] == 128
    assert cfg["dense"] == 1024
    assert cfg["dropout"] == 0.0           # dropout and recurrent_dropout both 0
    assert cfg["freeze"] is False          # trainable: true
    assert cfg["optimizer"] == "adam"
    assert cfg["weight_decay"] == 0.0      # weight_decay: null
    assert cfg["scheduler"] == "plateau"
    assert cfg["lr_factor"] == 0.1


def test_a_plateau_scheduler_is_built_and_an_unknown_one_is_refused():
    model = torch.nn.Linear(2, 2)
    cfg = dict(T.DEFAULTS)
    cfg["scheduler"] = "plateau"
    opt = T._make_optimizer(cfg, model)
    sched = T._make_scheduler(cfg, opt)
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert T._make_scheduler(dict(T.DEFAULTS), opt) is None
    cfg["scheduler"] = "cosine"
    with pytest.raises(ValueError, match="unknown scheduler"):
        T._make_scheduler(cfg, opt)


def test_the_plateau_scheduler_drops_the_rate_and_is_recorded(workspace):
    """mode='max' on val top-1, so no improvement means a cut."""
    cfg = base_cfg(workspace, epochs=4, scheduler="plateau", lr_patience=0,
                   lr_factor=0.1, lr=0.01)
    T.train(cfg, quiet=True)
    history = json.load(open(os.path.join(T.run_dir(cfg), "history.json")))
    rates = [h["lr"] for h in history]
    assert rates[-1] < rates[0], rates


def test_flipping_is_only_applied_to_training_clips(workspace):
    split, _, proc = workspace
    from src.data.dataset import CricShotDataset
    from src.data.augment import ClipFlip
    cfg = dict(T.DEFAULTS)
    cfg.update(base_cfg(workspace, flip=0.5))
    train_loader, val_loader, _ = T._make_loaders(cfg)
    assert isinstance(train_loader.dataset.transform, ClipFlip)
    assert val_loader.dataset.transform is None
