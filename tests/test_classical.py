"""Family D's classical models, through the real runner, on a fake cache:
fitted once, saved, scored in evaluate.py's format, repeatable."""
import io
import json
import os

import pytest
import torch

from src.experiments import registry as R
from src.experiments import run as runner
from src.methods import classical as C
from src.train import DEFAULTS
from src.utils.checkpoint import read_results

CLASSES = ["Cover Drive", "Sweep", "Pull"]


@pytest.fixture
def world(tmp_path, monkeypatch):
    feats = tmp_path / "features"
    gen = torch.Generator().manual_seed(0)
    rows = []
    for label, cls in enumerate(CLASSES):
        for k in range(12):
            clip = f"{cls}/vid{label}{k}_0.avi"
            rows.append((clip, cls, label))
            path = feats / "avg" / (clip[:-4] + ".pt")
            path.parent.mkdir(parents=True, exist_ok=True)
            x = torch.randn(4, 8, generator=gen) + 2.0 * label
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
    monkeypatch.setitem(R.BASES, "fake_classical", {
        "config": os.path.join("configs", "cached_gru.yaml"),
        "set": {"method": "classical", "head": "linear", "view": "avg",
                "flip": 0.0, "features": str(feats)}})
    entries = []
    for i, head in enumerate(C.HEADS):
        entries.append({"id": f"X{i + 1:02d}_{head}", "base": "fake_classical",
                        "cfg": {"head": head}, "why": "x"})
    monkeypatch.setattr(R, "ENTRIES", entries)
    monkeypatch.setattr(R, "ADOPT", {})
    out = str(tmp_path / "exp")
    return {"out": out, "over": {"out": out}}


def _run(w, exp_id, **kw):
    buf = io.StringIO()
    what = runner.run_one(R.get(exp_id), "author", overrides=w["over"],
                          quiet=True, out=buf, **kw)
    return what, buf.getvalue()


def _report(w, exp_id, part="val"):
    path = os.path.join(w["out"], f"{exp_id}__author", f"report_{part}.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.mark.parametrize("head", C.HEADS)
def test_every_classical_head_fits_and_scores(world, head):
    if head == "xgb":
        pytest.importorskip("xgboost")
    exp_id = "X%02d_%s" % (C.HEADS.index(head) + 1, head)
    what, text = _run(world, exp_id)
    assert what == "trained"
    assert "test scored and stored, not shown" in text
    row = read_results(os.path.join(world["out"], "results.jsonl"))[-1]
    assert row["stopped"] == "fitted once"
    assert row["epochs_run"] == 1
    assert row["deterministic"] is True
    folder = os.path.join(world["out"], f"{exp_id}__author")
    assert os.path.exists(os.path.join(folder, "model.joblib"))
    assert os.path.exists(os.path.join(folder, "report_card.png"))
    report = _report(world, exp_id)
    total = 0
    for line in report["confusion"]:
        total += sum(line)
    assert total == report["clips"] == 36
    assert report["top1"] <= report["top2"] <= report["top3"] <= 1.0


def test_the_report_has_evaluate_pys_fields(world):
    _run(world, "X01_linear")
    report = _report(world, "X01_linear")
    for key in ("checkpoint", "split_csv", "classes", "top1", "top3",
                "macro_f1", "per_class", "confusion", "trained_val_top1"):
        assert key in report, key
    assert report["checkpoint"]["path"].endswith("model.joblib")
    assert report["loss"] is not None          # logistic regression has probs


def test_svm_scores_without_probabilities(world):
    _run(world, "X02_svm")
    assert _report(world, "X02_svm")["loss"] is None


def test_a_fitted_model_is_skipped_next_time(world, monkeypatch):
    _run(world, "X03_rf")

    def no_fit(*a, **k):
        raise AssertionError("must not fit again")
    monkeypatch.setattr(C, "fit", no_fit)
    assert _run(world, "X03_rf")[0] == "skipped"


@pytest.mark.parametrize("exp_id", ["X03_rf", "X04_xgb", "X06_mlp"])
def test_the_same_seed_gives_the_same_model(world, exp_id):
    if exp_id == "X04_xgb":
        pytest.importorskip("xgboost")
    _run(world, exp_id)
    first = _report(world, exp_id)
    _run(world, exp_id, force=True)
    second = _report(world, exp_id)
    assert second["confusion"] == first["confusion"]
    assert second["top3"] == first["top3"]


def test_settings_a_single_fit_cannot_use_are_refused():
    cfg = dict(DEFAULTS)
    cfg.update({"method": "classical", "head": "svm", "view": "avg",
                "flip": 0.0})
    assert C.check(cfg) is None
    for change, word in (({"flip": 0.5}, "flip"), ({"view": "flat"}, "vector"),
                         ({"head": "lda"}, "unknown")):
        bad = dict(cfg)
        bad.update(change)
        assert word in C.check(bad), change


def test_all_six_family_d_ids_are_valid():
    for entry in R.entries("D"):
        assert C.check(R.resolve(entry, "author")) is None, entry["id"]


# ---- progress: a loading bar and a fitting clock --------------------------

def test_loading_shows_a_progress_line(world, monkeypatch, capsys):
    monkeypatch.setattr(C.progress, "is_tty", lambda: True)
    runner.run_one(R.get("X01_linear"), "author", overrides=world["over"],
                   quiet=False)
    out = capsys.readouterr().out
    assert "loading train" in out
    assert "scoring val" in out
    assert "fitted in" in out


def test_a_slow_fit_shows_a_running_clock(monkeypatch, capsys):
    import time

    class Slow:
        def fit(self, x, y):
            time.sleep(1.3)

    monkeypatch.setattr(C.progress, "is_tty", lambda: True)
    C._fit_with_clock(Slow(), None, None, "svm", quiet=False)
    assert "fitting svm ... 0:01 elapsed" in capsys.readouterr().out


def test_quiet_fits_print_nothing(monkeypatch, capsys):
    class Quick:
        def fit(self, x, y):
            pass

    monkeypatch.setattr(C.progress, "is_tty", lambda: True)
    C._fit_with_clock(Quick(), None, None, "rf", quiet=True)
    assert capsys.readouterr().out == ""
