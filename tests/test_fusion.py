"""29.1: late fusion averages finished runs' probabilities; nothing trains."""
import io
import json
import os

import numpy as np
import pytest
import torch

from src.experiments import registry as R
from src.experiments import run as runner
from src.methods import fusion as F
from src.utils.checkpoint import read_results

CLASSES = ["Cover Drive", "Sweep", "Pull"]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A fake avg cache, two splits, two cached runs and three fusions."""
    feats = tmp_path / "features"
    gen = torch.Generator().manual_seed(0)
    rows = []
    for label, cls in enumerate(CLASSES):
        for k in range(6):
            clip = f"{cls}/vid{label}{k}_0.avi"
            rows.append((clip, cls, label))
            for leaf in ("avg", "avg_mirror"):
                path = feats / leaf / (clip[:-4] + ".pt")
                path.parent.mkdir(parents=True, exist_ok=True)
                x = torch.randn(4, 8, generator=gen) + 1.5 * label
                torch.save(x.half(), str(path))
    splits = {}
    for name in ("author", "grouped"):
        folder = tmp_path / f"splits_{name}"
        folder.mkdir()
        for part in ("train", "val", "test"):
            text = "clip,class,label,match\n"
            for clip, cls, label in rows:
                text += f"{clip},{cls},{label},vid{label}\n"
            (folder / f"{part}.csv").write_text(text, encoding="utf-8")
        (folder / "classes.json").write_text(json.dumps(CLASSES),
                                             encoding="utf-8")
        splits[name] = str(folder)
    monkeypatch.setattr(R, "SPLITS", splits)
    monkeypatch.setitem(R.BASES, "fake", {
        "config": os.path.join("configs", "cached_gru.yaml"),
        "set": {"method": "cached_head", "head": "linear",
                "features": str(feats), "view": "avg"}})
    entries = [
        {"id": "X01_linear", "base": "fake", "cfg": {}, "why": "a"},
        {"id": "X02_mlp", "base": "fake", "cfg": {"head": "mlp"}, "why": "b"},
        {"id": "X03_self", "base": "fusion",
         "cfg": {"fuse": [["X01_linear"], ["X01_linear"]]}, "why": "c"},
        {"id": "X04_best", "base": "fusion",
         "cfg": {"fuse": [["X01_linear", "X02_mlp"]]}, "why": "d"},
        {"id": "X05_both", "base": "fusion",
         "cfg": {"fuse": [["X01_linear"], ["X02_mlp"]]}, "why": "e"},
    ]
    monkeypatch.setattr(R, "ENTRIES", entries)
    monkeypatch.setattr(R, "ADOPT", {})
    out = str(tmp_path / "exp")
    over = {"out": out, "epochs": 3, "patience": 99, "batch_size": 3,
            "lr": 1e-2, "workers": 0}
    return {"out": out, "over": over}


def _run(h, exp_id, split="author"):
    buf = io.StringIO()
    runner.run_one(R.get(exp_id), split, overrides=h["over"], quiet=True,
                   out=buf)
    return buf.getvalue()


def _report(h, name, part):
    path = os.path.join(h["out"], name, f"report_{part}.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_a_fusion_waits_for_its_runs_and_names_them(harness):
    status = R.status(R.get("X03_self"), harness["over"])
    assert status == "needs X01_linear on author first"


def test_a_run_fused_with_itself_scores_exactly_what_it_scored_alone(harness):
    _run(harness, "X01_linear")
    _run(harness, "X03_self")
    alone = _report(harness, "X01_linear__author", "val")
    fused = _report(harness, "X03_self__author", "val")
    assert fused["top1"] == alone["top1"]
    assert fused["confusion"] == alone["confusion"]
    # evaluate.py's loss is taken under the GPU's bf16 autocast, fusion's
    # from the same logits in fp32: the predictions are identical, the
    # loss agrees to bf16's three digits
    assert fused["loss"] == pytest.approx(alone["loss"], abs=5e-3)


def test_the_better_run_of_a_stream_is_chosen_on_val(harness):
    _run(harness, "X01_linear")
    _run(harness, "X02_mlp")
    _run(harness, "X04_best")
    rows = {}
    for row in read_results(os.path.join(harness["out"], "results.jsonl")):
        rows[row["name"]] = row
    better = "X01_linear"
    if rows["X02_mlp__author"]["best_val_top1"] > \
            rows["X01_linear__author"]["best_val_top1"]:
        better = "X02_mlp"
    chosen = rows["X04_best__author"]["sources"]
    assert len(chosen) == 1
    assert chosen[0]["id"] == better


def test_a_fusion_trains_nothing_and_hides_test(harness):
    _run(harness, "X01_linear")
    _run(harness, "X02_mlp")
    printed = _run(harness, "X05_both")
    row = None
    for found in read_results(os.path.join(harness["out"], "results.jsonl")):
        if found["name"] == "X05_both__author":
            row = found
    assert row["epochs_run"] == 0
    assert row["method"] == "fusion"
    folder = os.path.join(harness["out"], "X05_both__author")
    assert os.path.exists(os.path.join(folder, "fusion.json"))
    assert not os.path.exists(os.path.join(folder, "best.pt"))
    assert os.path.exists(os.path.join(folder, "report_test.json"))
    assert "test scored and stored, not shown" in printed


def test_only_clips_every_stream_has_are_averaged():
    a = {"clips": ["x", "y", "z"], "labels": [0, 1, 2], "classes": CLASSES,
         "probs": np.eye(3, dtype=np.float32)}
    b = {"clips": ["z", "y"], "labels": [2, 1], "classes": CLASSES,
         "probs": np.array([[0, 0, 1], [0, 1, 0]], dtype=np.float32)}
    clips, labels, probs = F._average([a, b])
    assert clips == ["y", "z"]
    assert list(labels) == [1, 2]
    assert np.allclose(probs, [[0, 1, 0], [0, 0, 1]])


def test_streams_that_disagree_about_a_label_are_refused():
    a = {"clips": ["x"], "labels": [0], "classes": CLASSES,
         "probs": np.ones((1, 3), dtype=np.float32) / 3}
    b = {"clips": ["x"], "labels": [1], "classes": CLASSES,
         "probs": np.ones((1, 3), dtype=np.float32) / 3}
    with pytest.raises(ValueError, match="label"):
        F._average([a, b])


def test_a_badly_written_fuse_is_refused():
    assert F._streams_problem(None).startswith("fusion needs fuse")
    assert F._streams_problem([["A"], []]).startswith("every stream")
    assert F._streams_problem([["A"], ["B", "C"]]) is None


def test_e03_fuses_a09_with_the_better_pose_head():
    cfg = R.resolve(R.get("E03_fuse_rgb_pose"), "author")
    assert cfg["method"] == "fusion"
    assert cfg["fuse"] == [["A09_tf4l8h"], ["E01_pose_gru", "E02_stgcn"]]
