"""G03 (Day 41): each view must really change the clip, and only val is scored.

A tiny fake model reads real clips from a real dataset, so a view that quietly
does nothing shows up as the wrong answer. No GPU.
"""
import json
import os

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.experiments import registry as R
from tools import ensemble as E
from tools import tta as T


def _timeline(frames=15):
    """A clip whose every pixel holds its own frame's number."""
    clip = torch.zeros(frames, 3, 4, 4)
    for i in range(frames):
        clip[i] = float(i)
    return clip


class _LeftOrRight(nn.Module):
    """Answers class 0 for a bright left half, class 1 for a bright right."""

    def __init__(self):
        super().__init__()
        self.unused = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        width = x.shape[-1]
        left = x[..., :width // 2].mean(dim=(1, 2, 3, 4))
        right = x[..., width // 2:].mean(dim=(1, 2, 3, 4))
        rest = torch.zeros_like(left)
        return torch.stack([left, right, rest], dim=1) * 10


def _real_clips(tmp_path, count=4):
    """A split file and .npy clips: left half white, the rest black."""
    processed = tmp_path / "processed"
    folder = processed / "box" / "shot"
    folder.mkdir(parents=True)
    lines = ["clip,class,label,match"]
    for i in range(count):
        frames = np.zeros((15, 8, 8, 3), dtype=np.uint8)
        frames[:, :, :4, :] = 255
        np.save(folder / f"vid{i}_0.npy", frames)
        lines.append(f"shot/vid{i}_0.avi,shot,0,vid{i}")
    split = tmp_path / "val.csv"
    split.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(split), str(processed)


def _fake_restore(split, processed):
    from src.data.dataset import CricShotDataset

    def restore(ckpt, split_csv, device=None):
        data = CricShotDataset(split_csv, processed=processed,
                               classes=["a", "b", "c"])
        return {}, data, _LeftOrRight(), {}

    return restore


# ---- the views --------------------------------------------------------------

def test_plain_changes_nothing():
    clip = _timeline()
    assert T.View("plain")(clip) is clip


def test_flip_mirrors_every_frame():
    clip = torch.arange(16.0).view(1, 1, 4, 4).repeat(15, 3, 1, 1)
    flipped = T.View("flip")(clip)
    assert torch.equal(flipped[:, :, :, 0], clip[:, :, :, 3])
    assert not torch.equal(flipped, clip)


def test_early_and_late_slide_and_repeat_the_edge_frame():
    early = T.View("early")(_timeline())
    late = T.View("late")(_timeline())
    assert early[:, 0, 0, 0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                          12, 12, 12]
    assert late[:, 0, 0, 0].tolist() == [2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                         12, 13, 14]


def test_every_view_keeps_the_frame_count():
    # F02's position table stops at 15
    for name in T.VIEWS:
        assert T.View(name)(_timeline()).shape == (15, 3, 4, 4)


def test_an_unknown_view_fails_loudly():
    with pytest.raises(ValueError, match="unknown view"):
        T.View("sideways")


def test_every_set_keeps_the_plain_view():
    for view_set in T.VIEW_SETS:
        assert "plain" in T.VIEW_SETS[view_set]


# ---- scoring ----------------------------------------------------------------

def test_a_flip_really_reaches_the_model(tmp_path, monkeypatch):
    split, processed = _real_clips(tmp_path)
    monkeypatch.setattr(T, "_restore", _fake_restore(split, processed))
    found = T.view_probs("best.pt", split, batch_size=2,
                         device=torch.device("cpu"))
    assert found["clips"][0] == "shot/vid0_0.avi"
    assert found["views"]["plain"].argmax(axis=1).tolist() == [0, 0, 0, 0]
    assert found["views"]["flip"].argmax(axis=1).tolist() == [1, 1, 1, 1]
    for name in T.VIEWS:
        assert found["views"][name].shape == (4, 3)
        assert np.allclose(found["views"][name].sum(axis=1), 1.0)


def test_stored_features_are_refused(monkeypatch):
    def restore(ckpt, split_csv, device=None):
        return {}, object(), _LeftOrRight(), {}

    monkeypatch.setattr(T, "_restore", restore)
    with pytest.raises(SystemExit, match="needs clips"):
        T.view_probs("best.pt", "val.csv", device=torch.device("cpu"))


def _found(left=0.9):
    views = {}
    for i, name in enumerate(T.VIEWS):
        views[name] = np.array([[left - i * 0.1, 1 - left + i * 0.1]])
    return {"clips": ["x"], "labels": np.array([0]), "classes": ["a", "b"],
            "views": views}


def test_nothing_scored_gives_none_without_touching_the_gpu(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("scored without being asked")

    monkeypatch.setattr(T, "view_probs", refuse)
    os.makedirs(tmp_path / "run")
    assert T.tta_scores("run", str(tmp_path)) is None


def test_scores_are_kept_and_read_back_with_no_temp_file(tmp_path, monkeypatch):
    calls = []

    def fake(ckpt, split_csv, device=None):
        calls.append(split_csv)
        return _found()

    monkeypatch.setattr(T, "view_probs", fake)
    os.makedirs(tmp_path / "run")
    first = T.tta_scores("run", str(tmp_path), score=True)
    second = T.tta_scores("run", str(tmp_path), score=True)
    assert len(calls) == 1
    assert np.array_equal(first["views"]["flip"], second["views"]["flip"])
    assert os.listdir(tmp_path / "run") == [T.CACHE]


def test_a_retrained_run_is_scored_again(tmp_path, monkeypatch):
    calls = []

    def fake(ckpt, split_csv, device=None):
        calls.append(ckpt)
        return _found()

    monkeypatch.setattr(T, "view_probs", fake)
    folder = tmp_path / "run"
    folder.mkdir()
    T.tta_scores("run", str(tmp_path), score=True)
    best = folder / "best.pt"
    best.write_bytes(b"new")
    later = os.path.getmtime(folder / T.CACHE) + 10
    os.utime(best, (later, later))
    assert T.tta_scores("run", str(tmp_path)) is None
    T.tta_scores("run", str(tmp_path), score=True)
    assert len(calls) == 2


def test_only_val_is_scored_on_the_chosen_split(tmp_path, monkeypatch):
    seen = []

    def fake(ckpt, split_csv, device=None):
        seen.append(split_csv)
        return _found()

    monkeypatch.setattr(T, "view_probs", fake)
    os.makedirs(tmp_path / "run")
    T.tta_scores("run", str(tmp_path), score=True, split="author")
    assert seen == [os.path.join(R.SPLITS["author"], "val.csv")]


def test_a_set_averages_only_its_own_views():
    found = _found()
    assert T.averaged(found, ("plain", "flip"))[0, 0] == pytest.approx(0.85)
    assert T.averaged(found, T.VIEWS)[0, 0] == pytest.approx(0.75)


# ---- the commands -----------------------------------------------------------

def _finished_runs(tmp_path, monkeypatch):
    """Two models with their views already scored."""
    rng = np.random.default_rng(5)
    clips = []
    labels = []
    for i in range(100):
        clips.append(f"shot/vid{i % 10}_{i}.avi")
        labels.append(i % 5)
    labels = np.asarray(labels)

    split_dir = tmp_path / "split"
    split_dir.mkdir()
    lines = ["clip,class,label,match"]
    for i, clip in enumerate(clips):
        lines.append(f"{clip},s,{labels[i]},vid{i % 10}")
    (split_dir / "val.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setitem(R.SPLITS, "grouped", str(split_dir))

    out = tmp_path / "experiments"
    results = tmp_path / "results.jsonl"
    names = ["M01_one__grouped", "M01_one__grouped__seed1", "M02_two__grouped"]
    with open(results, "w", encoding="utf-8") as handle:
        for name in names:
            views = {}
            for view in T.VIEWS:
                probs = rng.random((100, 5))
                probs[np.arange(100), labels] += 1.0
                views[view] = probs / probs.sum(axis=1, keepdims=True)
            folder = out / name
            folder.mkdir(parents=True)
            T._save({"clips": clips, "labels": labels,
                     "classes": ["a", "b", "c", "d", "e"], "views": views},
                    str(folder / T.CACHE))
            handle.write(json.dumps({"name": name, "best_val_top1": 0.8}) + "\n")
    return str(out), str(results), names


def test_the_command_prints_every_set_and_the_ledger(tmp_path, monkeypatch, capsys):
    out, results, names = _finished_runs(tmp_path, monkeypatch)
    code = T.main(["--exp", "M01_one", "M02_two", "--out", out,
                   "--results", results])
    printed = capsys.readouterr().out
    assert code == 0
    for word in ("ledger", "plain", "flip", "time", "all", names[1],
                 "M01_one mean"):
        assert word in printed
    assert "80.00" in printed


def test_a_missing_score_is_named(tmp_path, monkeypatch, capsys):
    out, results, names = _finished_runs(tmp_path, monkeypatch)
    os.remove(os.path.join(out, names[2], T.CACHE))
    code = T.main(["--exp", "M01_one", "M02_two", "--out", out,
                   "--results", results])
    printed = capsys.readouterr().out
    assert code == 1
    assert names[2] in printed


def test_an_unfinished_model_fails_loudly(tmp_path, monkeypatch):
    out, results, _ = _finished_runs(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="M09_none"):
        T.main(["--exp", "M09_none", "--out", out, "--results", results])


def test_ensembles_can_read_the_views(tmp_path, monkeypatch, capsys):
    out, results, _ = _finished_runs(tmp_path, monkeypatch)
    code = E.main(["--stream", "M01_one", "--stream", "M02_two", "--tta", "all",
                   "--out", out, "--results", results])
    printed = capsys.readouterr().out
    assert code == 0
    assert "plain flip early late" in printed
    assert "G01" in printed


def test_an_ensemble_names_its_tta_in_the_hint(tmp_path, monkeypatch, capsys):
    out, results, names = _finished_runs(tmp_path, monkeypatch)
    os.remove(os.path.join(out, names[0], T.CACHE))
    code = E.main(["--stream", "M01_one", "--stream", "M02_two", "--tta", "flip",
                   "--out", out, "--results", results])
    assert code == 1
    assert "--score --tta flip" in capsys.readouterr().out
