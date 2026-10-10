"""The metrics are checked against hand-worked numbers, and the staleness
guard is checked by actually making a report stale."""
import json
import os

import numpy as np
import pytest

from src.evaluate import (StaleReport, confusion, file_fingerprint,
                          format_report, load_report, per_class)


# ---- the metrics, against arithmetic done by hand -----------------------

def test_confusion_rows_are_truth_and_columns_are_prediction():
    m = confusion([0, 0, 1, 2], [0, 1, 1, 0], 3)
    assert m.tolist() == [[1, 1, 0],
                          [0, 1, 0],
                          [1, 0, 0]]


def test_every_clip_lands_somewhere_in_the_matrix():
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, 60)
    preds = rng.integers(0, 5, 60)
    assert confusion(labels, preds, 5).sum() == 60


def test_precision_recall_and_f1_by_hand():
    # class 0: predicted 2, of which 1 right; actually 2, of which 1 right
    #   precision 1/2, recall 1/2, F1 1/2
    # class 1: predicted 2, of which 1 right; actually 1, of which 1 right
    #   precision 1/2, recall 1/1, F1 2*.5*1/1.5 = 0.666..
    # class 2: predicted 0; actually 1, none right -> all zero
    m = confusion([0, 0, 1, 2], [0, 1, 1, 0], 3)
    got = per_class(m)
    assert got["precision"] == pytest.approx([0.5, 0.5, 0.0])
    assert got["recall"] == pytest.approx([0.5, 1.0, 0.0])
    assert got["f1"] == pytest.approx([0.5, 2 / 3, 0.0])
    assert got["support"] == [2, 1, 1]


def test_a_class_never_predicted_scores_zero_not_nan():
    """Division by zero here would poison the macro average silently."""
    got = per_class(confusion([0, 0, 1], [0, 0, 0], 2))
    assert got["precision"][1] == 0.0 and got["f1"][1] == 0.0
    assert not np.isnan(got["f1"]).any()


def test_a_perfect_prediction_scores_one_everywhere():
    got = per_class(confusion([0, 1, 2, 0], [0, 1, 2, 0], 3))
    assert got["f1"] == pytest.approx([1.0, 1.0, 1.0])


# ---- the fingerprint ----------------------------------------------------

def test_the_fingerprint_changes_when_the_bytes_change(tmp_path):
    p = tmp_path / "ckpt.pt"
    p.write_bytes(b"one")
    first = file_fingerprint(str(p))
    p.write_bytes(b"two")
    second = file_fingerprint(str(p))
    assert first["sha256"] != second["sha256"]


def test_the_same_bytes_give_the_same_hash(tmp_path):
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    assert file_fingerprint(str(a))["sha256"] == file_fingerprint(str(b))["sha256"]


def test_a_big_file_is_hashed_in_blocks(tmp_path):
    p = tmp_path / "big.pt"
    p.write_bytes(b"x" * (3 << 20))            # larger than one read block
    assert len(file_fingerprint(str(p))["sha256"]) == 64


# ---- the staleness guard, which is the point of the file ----------------

@pytest.fixture
def report_and_ckpt(tmp_path):
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"weights v1")
    report = {"checkpoint": file_fingerprint(str(ckpt)), "top1": 0.89}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return str(path), str(ckpt)


def test_a_matching_report_loads(report_and_ckpt):
    path, ckpt = report_and_ckpt
    assert load_report(path, ckpt)["top1"] == 0.89


def test_a_report_for_an_older_checkpoint_is_refused(report_and_ckpt):
    """The exact failure that cost this project a day."""
    path, ckpt = report_and_ckpt
    open(ckpt, "wb").write(b"weights v2 -- retrained since")
    with pytest.raises(StaleReport, match="different file"):
        load_report(path, ckpt)


def test_the_refusal_says_what_to_do_about_it(report_and_ckpt):
    """It raises rather than warns, and the message is actionable."""
    path, ckpt = report_and_ckpt
    open(ckpt, "wb").write(b"changed")
    with pytest.raises(StaleReport) as caught:
        load_report(path, ckpt)
    assert "Re-run the evaluation" in str(caught.value)


def test_a_missing_checkpoint_is_refused(report_and_ckpt):
    path, ckpt = report_and_ckpt
    os.remove(ckpt)
    with pytest.raises(StaleReport, match="not there now"):
        load_report(path, ckpt)


def test_the_guard_finds_the_checkpoint_from_the_report(report_and_ckpt):
    path, _ = report_and_ckpt
    assert load_report(path)["top1"] == 0.89     # no path passed in


def test_touching_the_file_without_changing_it_is_fine(report_and_ckpt):
    """mtime moves, bytes do not. A timestamp check would false-alarm here."""
    path, ckpt = report_and_ckpt
    os.utime(ckpt, (0, 0))
    assert load_report(path, ckpt)["top1"] == 0.89


# ---- the printed form ---------------------------------------------------

def test_the_report_prints_worst_class_first():
    report = {
        "checkpoint": {"path": "x", "sha256": "a" * 64, "mtime": "now"},
        "split_csv": "s", "clips": 4, "dropped": 0,
        "classes": ["Good", "Bad"], "loss": 1.0,
        "top1": 0.5, "top2": 0.75, "top3": 1.0,
        "macro_f1": 0.5, "macro_precision": 0.5, "macro_recall": 0.5,
        "per_class": {"precision": [0.9, 0.1], "recall": [0.9, 0.1],
                      "f1": [0.9, 0.1], "support": [2, 2]},
        "confusion": [[2, 0], [2, 0]],
    }
    text = format_report(report)
    assert text.index("Bad") < text.index("Good")
    assert "Bad" in text.split("most common confusions")[1]
