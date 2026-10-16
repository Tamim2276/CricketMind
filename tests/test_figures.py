"""tools/figures.py on a tiny fake ledger: what gets drawn, what is reported
as empty, that reruns are byte-identical, and that test is never read."""
import io
import json
import os

import pytest

from tools import figures as F

CLASSES = ["Cover Drive", "Late Cut", "Pull"]


def _report(top1):
    return {"classes": CLASSES, "top1": top1, "macro_f1": top1 - 0.02,
            "clips": 90, "confusion": [[25, 3, 2], [6, 20, 4], [1, 2, 27]],
            "per_class": {"f1": [0.8, 0.6, 0.85]}}


@pytest.fixture
def world(tmp_path):
    rows = []
    exp = tmp_path / "experiments"
    for name, top1, minutes in (("P01_effnetv2s__author", 0.40, 7.5),
                                ("P01_effnetv2s__grouped", 0.44, 6.6),
                                ("P04_videomae__author", 0.52, 9.0),
                                ("P04_videomae__grouped", 0.57, 6.5),
                                ("B01_effnetv2s__author", 0.79, 199.2)):
        rows.append({"name": name, "best_val_top1": top1,
                     "minutes": minutes, "split": "data/splits_author"})
        folder = exp / name
        folder.mkdir(parents=True)
        (folder / "report_val.json").write_text(json.dumps(_report(top1)),
                                                encoding="utf-8")
        # never read: a figure that touched it would crash here
        (folder / "report_test.json").write_text("{broken", encoding="utf-8")
    rows.append({"name": "smoke_cached", "best_val_top1": 0.03,
                 "minutes": 0.1, "split": "data/splits_author"})
    results = tmp_path / "results.jsonl"
    text = ""
    for row in rows:
        text += json.dumps(row) + "\n"
    results.write_text(text, encoding="utf-8")
    notes = tmp_path / "notes"
    return {"results": str(results), "exp": str(exp), "notes": str(notes),
            "mirror": str(notes / "runs")}


def _build(w, **kw):
    buf = io.StringIO()
    made = F.build(results=w["results"], exp_dir=w["exp"], notes=w["notes"],
                   mirror=w["mirror"], out=buf, **kw)
    return made, buf.getvalue()


def test_the_figures_that_have_data_are_drawn_in_both_themes(world):
    made, _ = _build(world)
    stems = []
    for stem, caption in made:
        stems.append(stem)
    assert stems == ["probe", "family_P", "family_B", "split_gap",
                     "cost_accuracy", "per_class_f1"]
    for stem in stems:
        for leaf in (f"{stem}.png", f"{stem}_dark.png"):
            assert os.path.exists(os.path.join(world["notes"], "compare",
                                               leaf)), leaf


def test_empty_families_are_reported_not_drawn(world):
    _, text = _build(world)
    for fam in ("A", "D", "V", "C"):
        assert f"family {fam}: no rows yet" in text
        assert not os.path.exists(os.path.join(world["notes"], "compare",
                                               f"family_{fam}.png"))


def test_every_run_gets_a_report_card_once(world):
    _, text = _build(world)
    assert "report cards: 5 drawn, 0 already there" in text
    assert os.path.exists(os.path.join(world["mirror"],
                                       "P04_videomae__author_report.png"))
    _, text = _build(world)
    assert "report cards: 0 drawn, 5 already there" in text
    _, text = _build(world, redraw=True)
    assert "report cards: 5 drawn" in text


def test_a_rerun_is_byte_identical(world):
    _build(world)
    compare = os.path.join(world["notes"], "compare")
    first = {}
    for leaf in sorted(os.listdir(compare)):
        with open(os.path.join(compare, leaf), "rb") as fh:
            first[leaf] = fh.read()
    with open(os.path.join(world["notes"], "index.md"), "rb") as fh:
        index = fh.read()
    _build(world)
    for leaf in first:
        with open(os.path.join(compare, leaf), "rb") as fh:
            assert fh.read() == first[leaf], leaf
    with open(os.path.join(world["notes"], "index.md"), "rb") as fh:
        assert fh.read() == index


def test_the_index_lists_every_figure_with_its_question(world):
    _build(world)
    with open(os.path.join(world["notes"], "index.md"), encoding="utf-8") as fh:
        text = fh.read()
    assert "[probe](compare/probe.png)" in text
    assert "which cache carries the most shot information?" in text
    assert "| P04_videomae__author | 52.00% |" in text
    assert "[report card](runs/P04_videomae__author_report.png)" in text
    assert "smoke_cached" not in text


def test_an_empty_ledger_draws_nothing_and_says_so(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text("", encoding="utf-8")
    buf = io.StringIO()
    made = F.build(results=str(results), exp_dir=str(tmp_path / "none"),
                   notes=str(tmp_path / "notes"),
                   mirror=str(tmp_path / "notes" / "runs"), out=buf)
    assert made == []
    assert "family P: no rows yet" in buf.getvalue()
