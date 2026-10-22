"""Day 48: the receipt checker, on receipts from the tiny DuckDB."""

import dataclasses

import pytest

from cricketmind_v3.application.receipt_checker import check_receipt
from cricketmind_v3.domain.receipts import from_line, query_text, to_line
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_engines import engines_on
from cricketmind_v3.pipeline.check_receipts import check_file, main, write_receipts
from tests.integration.test_duckdb_repo import build
from tests.integration.test_ingest import quiet
from tests.integration.test_receipts_db import build_from

FARHAN = CricketSubject(player="Sahibzada Farhan")
QUESTIONS = [
    ("strike_rate", FARHAN),
    ("sr_vs_average", CricketSubject(player="Fakhar Zaman")),  # a bootstrap range
    ("economy", CricketSubject(bowler="E Malinga")),
    (
        "venue_first_innings_avg",
        CricketSubject(venue="Lord's", match_format="T20", gender="female"),
    ),
    ("workload_ratio_flag", CricketSubject(bowler="E Malinga", date="2026-10-11")),
    ("review_overturn_rate", CricketSubject(team="Australia", gender="female")),
    (
        "match_won_by_runs",
        CricketSubject(
            team="Pakistan", opposition="Sri Lanka", date="2026-10-11", gender="male"
        ),
    ),
]


@pytest.fixture
def db(tmp_path):
    connection = build(tmp_path)
    yield connection
    connection.close()


def receipts_of(db) -> list:
    engines = engines_on(db)
    made = []
    for metric, subject in QUESTIONS:
        made.append(engines.fact(metric, subject).receipt)
    return made


# The plan's tests


def test_a_correct_receipt_passes(db):
    engines = engines_on(db)
    for receipt in receipts_of(db):
        assert check_receipt(receipt, engines) == [], receipt.query


def test_a_tampered_number_fails(db):
    receipt = receipts_of(db)[0]
    tampered = dataclasses.replace(receipt, value=170.0)
    assert check_receipt(tampered, engines_on(db)) == [
        f"the receipt says 170.0, the engines say {receipt.value}"
    ]


# And every other way a receipt can be wrong


def test_a_tampered_sample_or_range_fails(db):
    engines = engines_on(db)
    strike, versus = receipts_of(db)[0], receipts_of(db)[1]
    assert check_receipt(dataclasses.replace(strike, n=60), engines) == [
        "the receipt counts 60, the engines count 44"
    ]
    wider = dataclasses.replace(versus, interval=(-100.0, 100.0))
    assert len(check_receipt(wider, engines)) == 1


def test_a_number_moved_to_another_player_fails(db):
    receipt = receipts_of(db)[0]  # Farhan's 165.9
    moved = dataclasses.replace(
        receipt,
        query=query_text("strike_rate", CricketSubject(player="Fakhar Zaman"), None),
    )
    problems = check_receipt(moved, engines_on(db))
    assert problems[0].startswith("the receipt says 165.9")


def test_a_receipt_from_other_data_is_not_confirmed(db, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    two_matches = build_from(other, ("1535463", "1550709"))
    problems = check_receipt(receipts_of(db)[0], engines_on(two_matches))
    assert problems[0].startswith("other data: the receipt names ")
    two_matches.close()


def test_a_changed_engine_says_so(db):
    old = dataclasses.replace(receipts_of(db)[0], version="0")
    assert check_receipt(old, engines_on(db)) == [
        "N8.M1 has changed: the receipt names version 0, it is now 1"
    ]


def test_a_receipt_naming_the_wrong_engine_fails(db):
    wrong = dataclasses.replace(receipts_of(db)[0], engine="N8.M2")
    problems = check_receipt(wrong, engines_on(db))
    assert "the question is N8.M1's, not N8.M2's" in problems


# The script


def test_the_script_confirms_a_whole_file(db, tmp_path):
    path = tmp_path / "receipts.jsonl"
    write_receipts(path, receipts_of(db))
    assert check_file(db, path, say=quiet) == (7, 0)


def test_the_script_finds_the_tampered_line(db, tmp_path):
    path = tmp_path / "receipts.jsonl"
    receipts = receipts_of(db)
    receipts[2] = dataclasses.replace(receipts[2], value=1.0)
    write_receipts(path, receipts)
    said = []
    assert check_file(db, path, say=said.append) == (6, 1)
    assert said[2].startswith("line 3: NOT CONFIRMED: the receipt says 1.0")
    assert said[-1] == "6 confirmed, 1 not"


def test_an_unreadable_line_is_not_confirmed(db, tmp_path):
    path = tmp_path / "receipts.jsonl"
    path.write_text(to_line(receipts_of(db)[0]) + "\nnot a receipt\n", encoding="utf-8")
    assert check_file(db, path, say=quiet) == (1, 1)


def test_the_command_line(tmp_path):
    db = build(tmp_path)
    good = tmp_path / "good.jsonl"
    bad = tmp_path / "bad.jsonl"
    receipts = receipts_of(db)
    write_receipts(good, receipts)
    write_receipts(bad, [dataclasses.replace(receipts[0], value=1.0)])
    db.close()  # the script opens the file itself, read-only
    database = str(tmp_path / "cricket.duckdb")
    assert main([database, str(good)]) == 0
    assert main([database, str(bad)]) == 1
    assert main([database]) == 2
    assert from_line(good.read_text(encoding="utf-8").splitlines()[0]) == receipts[0]
