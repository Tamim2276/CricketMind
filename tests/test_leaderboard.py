"""The leaderboard: latest row wins, splits never averaged, gaps and
missing runs shown, smoke rows skipped and counted."""
from src.experiments import registry as R
from tools import leaderboard as L
from src.utils.checkpoint import read_results


def _row(name, top1, **extra):
    row = {"name": name, "best_val_top1": top1,
           "split": "data/splits_author"}
    row.update(extra)
    return row


def test_both_splits_side_by_side_with_the_gap():
    rows = [_row("P04_videomae__author", 0.52),
            _row("P04_videomae__grouped", 0.57)]
    text = L.render(L.collect(rows, "P"))
    line = ""
    for candidate in text.splitlines():
        if "P04_videomae" in candidate:
            line = candidate
    assert "52.00%" in line and "57.00%" in line
    assert "-5.00" in line


def test_the_latest_row_of_a_run_wins():
    rows = [_row("P01_effnetv2s__author", 0.30),
            _row("P01_effnetv2s__author", 0.40)]
    data = L.collect(rows)
    assert data["runs"][("P01_effnetv2s", "author")]["best_val_top1"] == 0.40


def test_smoke_rows_are_skipped_and_counted():
    rows = [_row("smoke_cached", 0.03), _row("smoke_frozen", 0.13),
            _row("P01_effnetv2s__author", 0.40)]
    data = L.collect(rows)
    assert data["smoke"] == 2
    assert "2 smoke-test rows skipped" in L.render(data)
    assert "smoke_cached" not in L.render(data)


def test_missing_runs_are_listed():
    rows = [_row("B01_effnetv2s__author", 0.79)]
    data = L.collect(rows, "B")
    assert data["total"] == 16
    assert len(data["pending"]) == 15
    text = L.render(data)
    assert "done 1 of 16 runs; 15 still to run" in text
    assert "B01_effnetv2s (grouped)" in text
    assert "B02_effnetv2b3" in text


def test_a_family_with_no_rows_says_so_not_an_empty_table():
    text = L.render(L.collect([], "C"))
    assert "no rows yet" in text


def test_adopted_rows_are_marked():
    rows = [_row("B01_effnetv2s__author", 0.7923,
                 adopted_from={"name": "mimic_author"})]
    assert "79.23%*" in L.render(L.collect(rows, "B"))


def test_old_names_and_strangers_are_shown_not_dropped():
    rows = [_row("mimic_author", 0.32), _row("mimic_author", 0.79),
            _row("Z01_gone__author", 0.5)]
    data = L.collect(rows)
    assert data["earlier"]["mimic_author"][1] == 2
    text = L.render(data)
    assert "79.00%  (2 runs)" in text
    assert "not in the registry: Z01_gone__author" in text


def test_sorting_puts_the_best_first():
    rows = [_row("P01_effnetv2s__author", 0.40),
            _row("P04_videomae__author", 0.52)]
    text = L.render(L.collect(rows, "P"), sort_by="author")
    assert text.index("P04_videomae") < text.index("P01_effnetv2s")


def test_the_real_ledger_renders():
    data = L.collect(read_results("experiments/results.jsonl"))
    text = L.render(data)
    assert data["total"] == 2 * len(R.ENTRIES)
    assert "mimic_author" in text
    assert "79.23%" in text
