"""The harness, on a tiny fake cache: both splits, skips, resumes, scores
test without showing it, adopts an old run, and never swallows a crash."""
import io
import json
import os

import pytest
import torch

from src.experiments import registry as R
from src.experiments import run as runner
from src.utils.checkpoint import read_results

CLASSES = ["Cover Drive", "Sweep", "Pull"]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    """A fake avg cache with a mirror half, two splits, and two entries."""
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
                # each class sits somewhere else, so it is learnable
                x = torch.randn(4, 8, generator=gen) + 3.0 * label
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
        {"id": "X03_tcn", "base": "fake", "cfg": {"head": "tcn"}, "why": "c"},
    ]
    monkeypatch.setattr(R, "ENTRIES", entries)
    monkeypatch.setattr(R, "ADOPT", {})

    out = str(tmp_path / "exp")
    over = {"out": out, "epochs": 2, "patience": 99, "batch_size": 3,
            "lr": 1e-3, "workers": 0}
    return {"out": out, "over": over, "tmp": tmp_path}


def _ledger(h):
    return read_results(os.path.join(h["out"], "results.jsonl"))


def _run(h, exp_id, split="author", **kw):
    buf = io.StringIO()
    what = runner.run_one(R.get(exp_id), split, overrides=h["over"],
                          quiet=True, out=buf, **kw)
    return what, buf.getvalue()


def test_a_dry_run_writes_nothing(harness):
    buf = io.StringIO()
    runner.run_many(["X01_linear"], ["author", "grouped"], dry_run=True,
                    overrides=harness["over"], quiet=True, out=buf)
    text = buf.getvalue()
    assert "X01_linear__author: train, then score val + test" in text
    assert "X01_linear__grouped" in text
    assert "splits_grouped" in text
    assert not os.path.exists(harness["out"])


def test_one_run_gives_a_row_and_two_reports(harness):
    what, text = _run(harness, "X01_linear")
    assert what == "trained"
    rows = _ledger(harness)
    assert len(rows) == 1
    assert rows[0]["name"] == "X01_linear__author"
    assert rows[0]["split_name"] == "author"
    folder = os.path.join(harness["out"], "X01_linear__author")
    for part in ("val", "test"):
        assert os.path.exists(os.path.join(folder, f"report_{part}.json"))
    assert not os.path.exists(os.path.join(folder, "resume.pt"))


def test_test_is_scored_but_never_printed(harness):
    _, text = _run(harness, "X01_linear")
    assert "val top-1" in text
    assert "test scored and stored, not shown" in text
    assert "test top-1" not in text


def test_a_finished_run_is_skipped(harness):
    _run(harness, "X01_linear")
    what, text = _run(harness, "X01_linear")
    assert what == "skipped"
    assert "already done" in text
    assert len(_ledger(harness)) == 1


def test_a_run_trained_but_not_scored_is_only_scored(harness, monkeypatch):
    _run(harness, "X01_linear")
    folder = os.path.join(harness["out"], "X01_linear__author")
    os.remove(os.path.join(folder, "report_test.json"))

    def no_training(*a, **k):
        raise AssertionError("must not train again")
    monkeypatch.setattr(runner, "train", no_training)

    what, _ = _run(harness, "X01_linear")
    assert what == "scored"
    assert os.path.exists(os.path.join(folder, "report_test.json"))
    assert len(_ledger(harness)) == 1


def test_resume_pt_means_resume_even_with_a_row(harness, monkeypatch):
    _run(harness, "X01_linear")
    folder = os.path.join(harness["out"], "X01_linear__author")
    open(os.path.join(folder, "resume.pt"), "wb").close()

    seen = []
    real = runner.train

    def spy(cfg, fresh=False, quiet=False):
        seen.append(fresh)
        os.remove(os.path.join(folder, "resume.pt"))   # stand-in file
        return real(cfg, fresh=True, quiet=quiet)
    monkeypatch.setattr(runner, "train", spy)

    what, _ = _run(harness, "X01_linear")
    assert what == "resumed"
    assert seen == [False]


def test_force_trains_again_and_rescores(harness):
    _run(harness, "X01_linear")
    what, _ = _run(harness, "X01_linear", force=True)
    assert what == "trained"
    assert len(_ledger(harness)) == 2


def test_run_many_does_both_splits_and_leaves_out_what_cannot_start(harness):
    buf = io.StringIO()
    done = runner.run_many(["X01_linear", "X03_tcn"], ["author", "grouped"],
                           overrides=harness["over"], quiet=True, out=buf)
    assert done == {"trained": 2}
    assert "X03_tcn" in buf.getvalue()
    assert "no tcn head yet" in buf.getvalue()
    names = set()
    for row in _ledger(harness):
        names.add(row["name"])
    assert names == {"X01_linear__author", "X01_linear__grouped"}


def test_a_crash_stops_everything(harness, monkeypatch):
    calls = []

    def boom(cfg, fresh=False, quiet=False):
        calls.append(cfg["name"])
        raise RuntimeError("diverged")
    monkeypatch.setattr(runner, "train", boom)
    with pytest.raises(RuntimeError):
        runner.run_many(["X01_linear", "X02_mlp"], ["author", "grouped"],
                        overrides=harness["over"], quiet=True,
                        out=io.StringIO())
    assert calls == ["X01_linear__author"]


def test_an_old_run_with_the_same_config_is_adopted(harness, monkeypatch):
    # the old run: same config, older name
    cfg = R.resolve(R.get("X01_linear"), "author", harness["over"])
    cfg["name"] = "old_run"
    runner.train(cfg, quiet=True)
    monkeypatch.setitem(R.ADOPT, "X01_linear__author", "old_run")

    def no_training(*a, **k):
        raise AssertionError("adoption must not train")
    monkeypatch.setattr(runner, "train", no_training)

    what, text = _run(harness, "X01_linear")
    assert what == "adopted"
    row = _ledger(harness)[-1]
    assert row["name"] == "X01_linear__author"
    assert row["adopted_from"]["name"] == "old_run"
    assert row["checkpoint"].endswith(os.path.join("old_run", "best.pt"))
    folder = os.path.join(harness["out"], "X01_linear__author")
    assert os.path.exists(os.path.join(folder, "report_val.json"))
    # and it now counts as done
    assert _run(harness, "X01_linear")[0] == "skipped"


def test_an_old_run_with_a_different_config_is_refused(harness, monkeypatch):
    cfg = R.resolve(R.get("X01_linear"), "author", harness["over"])
    cfg["name"] = "old_run"
    cfg["lr"] = 5e-4
    runner.train(cfg, quiet=True)
    monkeypatch.setitem(R.ADOPT, "X01_linear__author", "old_run")
    with pytest.raises(SystemExit, match="lr differ"):
        _run(harness, "X01_linear")


# ---- 9.3: the three properties that make the harness safe ----------------

def _folder(h, name):
    return os.path.join(h["out"], name)


def _history(h, name):
    with open(os.path.join(_folder(h, name), "history.json"),
              encoding="utf-8") as fh:
        rows = json.load(fh)
    kept = []
    for row in rows:
        clean = dict(row)
        clean.pop("secs", None)          # wall-clock time, never the same
        kept.append(clean)
    return kept


def _report(h, name, part):
    path = os.path.join(_folder(h, name), f"report_{part}.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def test_idempotent_a_second_call_touches_nothing(harness, monkeypatch):
    _run(harness, "X01_linear")
    folder = _folder(harness, "X01_linear__author")
    before = {}
    for leaf in os.listdir(folder):
        before[leaf] = os.path.getmtime(os.path.join(folder, leaf))
    rows = _ledger(harness)

    def forbidden(*a, **k):
        raise AssertionError("a finished run must not train or score")
    monkeypatch.setattr(runner, "train", forbidden)
    monkeypatch.setattr(runner, "evaluate_checkpoint", forbidden)

    assert _run(harness, "X01_linear")[0] == "skipped"
    assert _ledger(harness) == rows
    after = {}
    for leaf in os.listdir(folder):
        after[leaf] = os.path.getmtime(os.path.join(folder, leaf))
    assert after == before


@pytest.mark.parametrize("exp_id", ["X01_linear", "X02_mlp"])
def test_deterministic_same_seed_same_numbers(harness, exp_id):
    # flip is 0.5 here, so the mirror coin is part of what must repeat
    name = exp_id + "__author"
    _run(harness, exp_id)
    first = _ledger(harness)[-1]
    first_history = _history(harness, name)
    first_val = _report(harness, name, "val")

    _run(harness, exp_id, force=True)
    second = _ledger(harness)[-1]
    assert second["recorded_at"] >= first["recorded_at"]
    for key in ("best_val_top1", "epochs_run", "stopped"):
        assert second[key] == first[key], key
    assert _history(harness, name) == first_history
    second_val = _report(harness, name, "val")
    assert second_val["confusion"] == first_val["confusion"]
    assert second_val["loss"] == first_val["loss"]


def test_the_row_records_the_config_that_ran(harness):
    _run(harness, "X02_mlp", "grouped")
    row = _ledger(harness)[-1]
    want = R.resolve(R.get("X02_mlp"), "grouped", harness["over"])
    assert row["cfg"] == want
    saved = torch.load(os.path.join(_folder(harness, "X02_mlp__grouped"),
                                    "best.pt"), weights_only=False)
    assert saved["cfg"] == want


def test_every_real_ledger_row_still_means_what_it_meant():
    # if a registry edit changes what an ID means after it ran, its old
    # rows would describe a different experiment; this catches that
    path = os.path.join("experiments", "results.jsonl")
    checked = 0
    for row in read_results(path):
        name = row.get("name", "")
        if "__" not in name:
            continue
        exp_id, split = name.split("__", 1)
        entry = R.get(exp_id)
        differ = R.config_differences(row["cfg"], R.resolve(entry, split))
        assert differ == [], (name, differ)
        checked += 1
    assert checked > 0


def test_a_different_seed_changes_the_numbers(harness):
    # so the determinism test above could fail, and its pass means something
    name = "X02_mlp__author"
    _run(harness, "X02_mlp")
    first_history = _history(harness, name)
    harness["over"]["seed"] = 7
    _run(harness, "X02_mlp", force=True)
    assert _history(harness, name) != first_history


# ---- 9.5: the report card comes with the scores ---------------------------

def test_a_scored_run_gets_its_report_card(harness, figures_go_to_tmp):
    _run(harness, "X01_linear")
    folder = _folder(harness, "X01_linear__author")
    assert os.path.exists(os.path.join(folder, "report_card.png"))
    assert os.path.exists(os.path.join(folder, "report_card_dark.png"))
    kept = os.path.join(figures_go_to_tmp, "X01_linear__author_report.png")
    assert os.path.exists(kept)
