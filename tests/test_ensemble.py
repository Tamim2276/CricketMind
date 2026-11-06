"""Family G's maths (Day 41), on made-up answers. No model, no GPU.

The checks that matter most: a fitted score held out must not flatter, a match
never splits across folds, and only val is ever scored.
"""
import json
import os

import numpy as np
import pytest

from src.experiments import registry as R
from tools import ensemble as E
from tools import tuning as U


def _answers(labels, classes, rng, right):
    """Probabilities where the true class leads on about `right` of the clips."""
    probs = rng.random((len(labels), classes))
    for i, label in enumerate(labels):
        if rng.random() < right:
            probs[i, label] = probs[i].max() + 1.0
    return probs / probs.sum(axis=1, keepdims=True)


def _stream(clips, labels, probs, classes=("a", "b", "c")):
    return {"clips": list(clips), "labels": np.asarray(labels),
            "probs": np.asarray(probs), "classes": list(classes)}


# ---- weights ----------------------------------------------------------------

def test_weight_grid_is_complete_and_adds_to_one():
    grid = E.weight_grid(3, step=0.25)
    assert len(grid) == 15
    for weights in grid:
        assert len(weights) == 3
        assert sum(weights) == pytest.approx(1.0)


def test_two_streams_get_21_weights_at_the_default_step():
    assert len(E.weight_grid(2)) == 21


def test_vote_is_equal_unless_told_otherwise():
    first = np.array([[1.0, 0.0]])
    second = np.array([[0.0, 1.0]])
    assert E.vote([first, second]).tolist() == [[0.5, 0.5]]
    assert E.vote([first, second], [0.75, 0.25]).tolist() == [[0.75, 0.25]]


def test_fit_weights_leans_on_the_stream_that_knows():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 400)
    good = _answers(labels, 5, rng, 0.9)
    noise = _answers(labels, 5, rng, 0.0)
    weights = E.fit_weights([good, noise], labels)
    assert weights[0] > weights[1]


def test_a_top1_tie_goes_to_the_lower_loss():
    # every weight gets both clips right, so only the loss can choose
    labels = np.array([0, 1])
    sure = np.array([[0.99, 0.01], [0.01, 0.99]])
    unsure = np.array([[0.6, 0.4], [0.4, 0.6]])
    assert E.fit_weights([sure, unsure], labels) == [1.0, 0.0]


# ---- folds ------------------------------------------------------------------

def test_a_match_stays_whole_in_one_fold():
    matches = ["m1", "m1", "m2", "m3", "m3", "m3", "m4", "m5", "m6", "m6"]
    folds = E.match_folds(matches, folds=3)
    seen = {}
    for match, fold in zip(matches, folds):
        assert seen.setdefault(match, fold) == fold


def test_folds_are_balanced_and_repeatable():
    matches = []
    for i in range(40):
        matches += [f"m{i}"] * (1 + i % 5)
    folds = E.match_folds(matches)
    sizes = np.bincount(folds)
    assert len(sizes) == 5
    assert sizes.max() - sizes.min() <= 5
    assert np.array_equal(folds, E.match_folds(matches))


def test_too_few_matches_fail_loudly():
    with pytest.raises(ValueError, match="cannot fill"):
        E.match_folds(["m1", "m2"], folds=5)


# ---- held out ---------------------------------------------------------------

def test_held_out_does_not_flatter_on_pure_noise():
    # nothing can be learned here, so an honest score sits at chance
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 15, 300)
    stacks = []
    for _ in range(5):
        stacks.append(_answers(labels, 15, rng, 0.0))
    matches = []
    for i in range(300):
        matches.append(f"m{i % 30}")
    folds = E.match_folds(matches)

    flattered = E.top1(E.stack_probs(E.fit_stack(stacks, labels), stacks), labels)
    honest = E.top1(E.held_out_stack(stacks, labels, folds), labels)
    assert honest < 0.14
    assert flattered > honest + 0.1


def test_held_out_still_finds_a_real_signal():
    rng = np.random.default_rng(2)
    labels = rng.integers(0, 5, 400)
    stacks = [_answers(labels, 5, rng, 0.85), _answers(labels, 5, rng, 0.85)]
    matches = []
    for i in range(400):
        matches.append(f"m{i % 40}")
    folds = E.match_folds(matches)
    assert E.top1(E.held_out_stack(stacks, labels, folds), labels) > 0.8
    assert E.top1(E.held_out_weights(stacks, labels, folds), labels) > 0.8


def test_a_fold_short_of_a_class_keeps_every_column():
    # class 2 lives in one match, so that fold's head never trains on it
    rng = np.random.default_rng(3)
    labels = np.array([0, 1] * 30 + [2] * 3)
    matches = []
    for i in range(60):
        matches.append(f"m{i % 6}")
    matches += ["only"] * 3
    stacks = [_answers(labels, 3, rng, 0.8), _answers(labels, 3, rng, 0.8)]
    probs = E.held_out_stack(stacks, labels, E.match_folds(matches, folds=3))
    assert probs.shape == (63, 3)
    assert probs[:, :2].sum() > 0


# ---- lining clips up --------------------------------------------------------

def test_align_puts_every_stream_in_one_clip_order():
    first = _stream(["x", "y"], [0, 1], [[1, 0, 0], [0, 1, 0]])
    second = _stream(["y", "x"], [1, 0], [[0, 0.5, 0.5], [0.5, 0.5, 0]])
    clips, labels, stacks = E.align([first, second])
    assert clips == ["x", "y"]
    assert labels.tolist() == [0, 1]
    assert stacks[1].tolist() == [[0.5, 0.5, 0], [0, 0.5, 0.5]]


def test_align_keeps_only_clips_every_stream_has():
    first = _stream(["x", "y"], [0, 1], [[1, 0, 0], [0, 1, 0]])
    second = _stream(["y"], [1], [[0, 1, 0]])
    clips, _, stacks = E.align([first, second])
    assert clips == ["y"]
    assert stacks[0].shape == (1, 3)


def test_streams_that_disagree_on_a_label_fail_loudly():
    first = _stream(["x"], [0], [[1, 0, 0]])
    second = _stream(["x"], [1], [[1, 0, 0]])
    with pytest.raises(ValueError, match="disagree"):
        E.align([first, second])


# ---- the command ------------------------------------------------------------

def _save_run(out, name, clips, labels, probs):
    folder = os.path.join(out, name)
    os.makedirs(folder, exist_ok=True)
    np.savez(os.path.join(folder, U.CACHE), clips=np.asarray(clips),
             labels=np.asarray(labels), probs=probs,
             classes=np.asarray(["a", "b", "c", "d", "e"]))


def _finished_runs(tmp_path, monkeypatch):
    """Two models, one with a repeat, and the split file that names matches."""
    rng = np.random.default_rng(4)
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

    out = str(tmp_path / "experiments")
    names = ["M01_one__grouped", "M01_one__grouped__seed1", "M02_two__grouped"]
    results = tmp_path / "results.jsonl"
    with open(results, "w", encoding="utf-8") as handle:
        for name in names:
            _save_run(out, name, clips, labels, _answers(labels, 5, rng, 0.8))
            handle.write(json.dumps({"name": name, "best_val_top1": 0.8}) + "\n")
    return out, str(results), names


def test_the_command_prints_every_way_of_combining(tmp_path, monkeypatch, capsys):
    out, results, _ = _finished_runs(tmp_path, monkeypatch)
    code = E.main(["--stream", "M01_one", "--stream", "M02_two",
                   "--out", out, "--results", results])
    printed = capsys.readouterr().out
    assert code == 0
    assert "100 clips" in printed
    assert "2 runs" in printed
    for label in ("G01", "G02 weights, held out", "G04 stacking, held out"):
        assert label in printed


def test_a_missing_score_is_named_and_needs_no_gpu(tmp_path, monkeypatch, capsys):
    out, results, names = _finished_runs(tmp_path, monkeypatch)
    os.remove(os.path.join(out, names[1], U.CACHE))
    code = E.main(["--stream", "M01_one", "--stream", "M02_two",
                   "--out", out, "--results", results])
    printed = capsys.readouterr().out
    assert code == 1
    assert names[1] in printed
    assert "--score" in printed


def test_an_unfinished_model_fails_loudly(tmp_path, monkeypatch):
    out, results, _ = _finished_runs(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="M09_none"):
        E.main(["--stream", "M01_one", "--stream", "M09_none",
                "--out", out, "--results", results])


def test_one_stream_is_not_an_ensemble(tmp_path, monkeypatch):
    out, results, _ = _finished_runs(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="two streams"):
        E.main(["--stream", "M01_one", "--out", out, "--results", results])


def test_only_val_is_scored_on_the_chosen_split(tmp_path, monkeypatch):
    seen = []

    def fake_probabilities(ckpt, split_csv, batch_size, device=None,
                           on_batch=None):
        seen.append(split_csv)
        return {"clips": ["x"], "labels": [0], "probs": np.ones((1, 2)) / 2,
                "classes": ["a", "b"]}

    monkeypatch.setattr(U, "probabilities", fake_probabilities)
    out = str(tmp_path)
    os.makedirs(os.path.join(out, "M01_one__author"))
    U.val_probs("M01_one__author", out, score=True, split="author")
    assert len(seen) == 1
    assert seen[0] == os.path.join(R.SPLITS["author"], "val.csv")


def test_a_chosen_device_reaches_the_scoring(tmp_path, monkeypatch, capsys):
    out, results, names = _finished_runs(tmp_path, monkeypatch)
    os.remove(os.path.join(out, names[2], U.CACHE))
    seen = []

    def fake_probabilities(ckpt, split_csv, batch_size, device=None,
                           on_batch=None):
        seen.append(device)
        saved = np.load(os.path.join(out, names[0], U.CACHE))
        return {"clips": list(saved["clips"]), "labels": saved["labels"],
                "probs": saved["probs"], "classes": list(saved["classes"])}

    monkeypatch.setattr(U, "probabilities", fake_probabilities)
    code = E.main(["--stream", "M01_one", "--stream", "M02_two", "--score",
                   "--device", "cpu", "--out", out, "--results", results])
    assert code == 0
    assert len(seen) == 1
    assert str(seen[0]) == "cpu"
