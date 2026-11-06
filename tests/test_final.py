"""Day 41's last step: what is frozen on val stays frozen, and test is read
once. Made-up answers and checkpoints; no model, no GPU.
"""
import json
import os

import numpy as np
import pytest

from src.experiments import registry as R
from tools import ensemble as E
from tools import final as F
from tools import tta as T
from tools import tuning as U

CLASSES = ["a", "b", "c", "d", "e"]


def _answers(labels, rng, right):
    probs = rng.random((len(labels), len(CLASSES)))
    for i, label in enumerate(labels):
        if rng.random() < right:
            probs[i, label] = probs[i].max() + 1.0
    return probs / probs.sum(axis=1, keepdims=True)


class _World:
    """Two finished models (one with a repeat), val answers kept, and a test
    the fake scorer answers. `noise` makes the second model know nothing."""

    def __init__(self, tmp_path, monkeypatch, noise=False):
        self.rng = np.random.default_rng(7)
        self.clips = []
        labels = []
        for i in range(300):
            self.clips.append(f"shot/vid{i % 30}_{i}.avi")
            labels.append(i % 5)
        self.labels = np.asarray(labels)

        split_dir = tmp_path / "split"
        split_dir.mkdir()
        lines = ["clip,class,label,match"]
        for i, clip in enumerate(self.clips):
            lines.append(f"{clip},s,{self.labels[i]},vid{i % 30}")
        for part in ("val", "test"):
            (split_dir / f"{part}.csv").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
        monkeypatch.setitem(R.SPLITS, "grouped", str(split_dir))

        self.out = str(tmp_path / "experiments")
        self.results = str(tmp_path / "results.jsonl")
        self.names = ["M01_one__grouped", "M01_one__grouped__seed1",
                      "M02_two__grouped"]
        with open(self.results, "w", encoding="utf-8") as handle:
            for name in self.names:
                right = 0.7
                if noise and name.startswith("M02"):
                    right = 0.0
                folder = os.path.join(self.out, name)
                os.makedirs(folder)
                with open(os.path.join(folder, "best.pt"), "wb") as ckpt:
                    ckpt.write(name.encode())
                np.savez(os.path.join(folder, U.CACHE),
                         clips=np.asarray(self.clips), labels=self.labels,
                         probs=_answers(self.labels, self.rng, right),
                         classes=np.asarray(CLASSES))
                handle.write(json.dumps({"name": name,
                                         "best_val_top1": right}) + "\n")

        self.scored = []
        monkeypatch.setattr(F, "probabilities", self.fake_probabilities)
        monkeypatch.setattr(F, "view_probs", self.fake_views)

    def fake_probabilities(self, ckpt, split_csv, batch_size=4, device=None,
                           on_batch=None):
        self.scored.append((ckpt, split_csv))
        return {"clips": list(self.clips), "labels": self.labels,
                "probs": _answers(self.labels, self.rng, 0.7),
                "classes": list(CLASSES)}

    def fake_views(self, ckpt, split_csv, device=None):
        self.scored.append((ckpt, split_csv))
        views = {}
        for view in ("plain", "flip", "early", "late"):
            views[view] = _answers(self.labels, self.rng, 0.7)
        return {"clips": list(self.clips), "labels": self.labels,
                "classes": list(CLASSES), "views": views}

    def save(self, method="G01", extra=()):
        argv = ["save", method, "--stream", "M01_one", "--stream", "M02_two",
                "--out", self.out, "--results", self.results]
        return F.main(argv + list(extra))

    def folder(self, method="G01", name="final"):
        return os.path.join(self.out, f"{method}_{name}__grouped")

    def choice(self, method="G01"):
        with open(os.path.join(self.folder(method), F.CHOICE),
                  encoding="utf-8") as handle:
            return json.load(handle)


# ---- save -------------------------------------------------------------------

def test_save_freezes_the_recipe_and_every_checkpoint(tmp_path, monkeypatch):
    world = _World(tmp_path, monkeypatch)
    assert world.save("G02") == 0
    choice = world.choice("G02")
    assert sum(choice["recipe"]["weights"]) == pytest.approx(1.0)
    assert choice["val"]["clips"] == 300
    runs = []
    for stream in choice["streams"]:
        for run in stream["runs"]:
            runs.append(run["name"])
            assert len(run["checkpoint"]["sha256"]) == 64
    assert runs == world.names
    assert world.scored == []        # save never looks at test


def test_g01_saves_equal_weights(tmp_path, monkeypatch):
    world = _World(tmp_path, monkeypatch)
    assert world.save("G01") == 0
    assert world.choice("G01")["recipe"]["weights"] == [0.5, 0.5]


def test_a_saved_stack_answers_as_sklearn_does():
    rng = np.random.default_rng(8)
    labels = rng.integers(0, 5, 200)
    stacks = [_answers(labels, rng, 0.7), _answers(labels, rng, 0.7)]
    recipe = F.recipe_of("G04", stacks, labels)
    model = E.fit_stack(stacks, labels)
    assert np.allclose(F.apply(recipe, stacks), E.stack_probs(model, stacks))
    json.dumps(recipe)               # it must fit in the report


def test_an_ensemble_that_loses_to_its_best_stream_is_refused(tmp_path,
                                                             monkeypatch):
    world = _World(tmp_path, monkeypatch, noise=True)
    with pytest.raises(SystemExit, match="Day 41's check fails"):
        world.save("G01")
    assert not os.path.exists(world.folder("G01"))


def test_nothing_changes_once_test_is_read(tmp_path, monkeypatch):
    world = _World(tmp_path, monkeypatch)
    os.makedirs(world.folder())
    with open(os.path.join(world.folder(), F.REPORT), "w") as handle:
        handle.write("{}")
    with pytest.raises(SystemExit, match="already read"):
        world.save("G01")


def test_val_answers_must_exist_first(tmp_path, monkeypatch, capsys):
    world = _World(tmp_path, monkeypatch)
    os.remove(os.path.join(world.out, world.names[1], U.CACHE))
    assert world.save("G01", ["--tta", "flip"]) == 1
    printed = capsys.readouterr().out
    assert world.names[1] in printed
    assert "--score --tta flip" in printed


def test_a_name_that_breaks_the_folder_is_refused(tmp_path, monkeypatch):
    world = _World(tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="lower case"):
        world.save("G01", ["--name", "c06__f02"])


# ---- test -------------------------------------------------------------------

def test_test_is_read_once(tmp_path, monkeypatch, capsys):
    world = _World(tmp_path, monkeypatch)
    world.save("G02")
    folder = os.path.basename(world.folder("G02"))
    assert F.main(["test", folder, "--out", world.out]) == 0
    with open(os.path.join(world.folder("G02"), F.REPORT),
              encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["clips"] == 300
    assert sorted(report["members_top1"]) == ["M01_one", "M02_two"]
    assert "top-1" in capsys.readouterr().out
    scored = len(world.scored)

    assert F.main(["test", folder, "--out", world.out]) == 1
    assert "read once already" in capsys.readouterr().out
    assert len(world.scored) == scored


def test_only_the_saved_split_test_file_is_scored(tmp_path, monkeypatch):
    world = _World(tmp_path, monkeypatch)
    world.save("G01")
    F.main(["test", os.path.basename(world.folder()), "--out", world.out])
    assert len(world.scored) == 3
    for _, split_csv in world.scored:
        assert split_csv == os.path.join(R.SPLITS["grouped"], "test.csv")


def test_a_changed_checkpoint_stops_the_test(tmp_path, monkeypatch):
    world = _World(tmp_path, monkeypatch)
    world.save("G01")
    with open(os.path.join(world.out, world.names[2], "best.pt"), "wb") as ckpt:
        ckpt.write(b"retrained")
    with pytest.raises(SystemExit, match="changed since"):
        F.main(["test", os.path.basename(world.folder()), "--out", world.out])
    assert world.scored == []


def test_a_cut_off_test_resumes_from_the_runs_already_scored(tmp_path,
                                                            monkeypatch):
    world = _World(tmp_path, monkeypatch)
    world.save("G01")
    real = world.fake_probabilities
    calls = []

    def dies_on_the_third(ckpt, split_csv, batch_size=4, device=None,
                          on_batch=None):
        calls.append(ckpt)
        if len(calls) == 3:
            raise RuntimeError("power cut")
        return real(ckpt, split_csv, batch_size, device)

    monkeypatch.setattr(F, "probabilities", dies_on_the_third)
    folder = os.path.basename(world.folder())
    with pytest.raises(RuntimeError):
        F.main(["test", folder, "--out", world.out])
    monkeypatch.setattr(F, "probabilities", real)
    world.scored.clear()
    assert F.main(["test", folder, "--out", world.out]) == 0
    assert len(world.scored) == 1    # only the run the cut stopped


def test_saving_again_drops_answers_kept_for_the_old_recipe(tmp_path,
                                                           monkeypatch):
    world = _World(tmp_path, monkeypatch)
    world.save("G01")
    kept = os.path.join(world.folder(), "test_probs")
    os.makedirs(kept)
    world.save("G01")
    assert not os.path.exists(kept)


def test_saved_views_are_used_on_test(tmp_path, monkeypatch):
    world = _World(tmp_path, monkeypatch)
    # the views' val answers, as tools.tta keeps them
    for name in world.names:
        cache = os.path.join(world.out, name, U.CACHE)
        saved = np.load(cache)
        views = {}
        for view in ("plain", "flip", "early", "late"):
            views[view] = saved["probs"]
        T._save({"clips": list(saved["clips"]), "labels": saved["labels"],
                 "classes": list(saved["classes"]), "views": views},
                os.path.join(world.out, name, T.CACHE))
    assert world.save("G01", ["--tta", "flip"]) == 0
    assert world.choice()["tta"] == "flip"
    F.main(["test", os.path.basename(world.folder()), "--out", world.out])
    assert len(world.scored) == 3


def test_no_saved_ensemble_fails_loudly(tmp_path):
    with pytest.raises(SystemExit, match="save one first"):
        F.main(["test", "G01_final__grouped", "--out", str(tmp_path)])
