"""Day 47: receipts from the real engines on the tiny DuckDB."""

import zipfile

import duckdb
import pytest

from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_engines import engines_on
from cricketmind_v3.infrastructure.snapshot import data_fingerprint
from cricketmind_v3.pipeline.attributes import build_attributes, read_player_meta
from cricketmind_v3.pipeline.derived import build_derived
from cricketmind_v3.pipeline.ingest import ingest
from cricketmind_v3.pipeline.players import build_players
from tests.integration.test_attributes_join import SAMPLE, thushara_by_hand
from tests.integration.test_duckdb_repo import build
from tests.integration.test_ingest import FIXTURES, quiet

FARHAN = CricketSubject(player="Sahibzada Farhan")


def build_from(folder, match_ids, hand=()):
    """A tiny database from some of the fixtures, with hand labels if given."""
    zip_path = folder / "all_json.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for match_id in match_ids:
            archive.write(FIXTURES / f"{match_id}.json", f"{match_id}.json")
    db_path = folder / "cricket.duckdb"
    ingest(zip_path, db_path, say=quiet)
    db = duckdb.connect(str(db_path))
    build_players(db)
    build_attributes(db, read_player_meta(SAMPLE), list(hand))
    build_derived(db)
    return db


@pytest.fixture
def db(tmp_path):
    connection = build(tmp_path)
    yield connection
    connection.close()


# The plan's tests


def test_the_same_question_twice_gives_the_same_receipt(db):
    engines = engines_on(db)
    first = engines.fact("strike_rate", FARHAN).receipt
    again = engines.fact("strike_rate", FARHAN).receipt
    assert first == again
    assert first.data == data_fingerprint(db)
    assert (first.engine, first.version, first.n) == ("N8.M1", "1", 44)


def test_a_changed_data_file_gives_a_different_receipt(tmp_path):
    whole = tmp_path / "whole"
    part = tmp_path / "part"
    whole.mkdir()
    part.mkdir()
    everything = build_from(whole, ("1490709", "1535463", "1550709"))
    two = build_from(part, ("1535463", "1550709"))  # one match fewer: another zip
    same_answer = engines_on(everything).fact("strike_rate", FARHAN)
    other_data = engines_on(two).fact("strike_rate", FARHAN)
    assert same_answer.value == other_data.value  # Farhan's balls are in both
    assert same_answer.receipt != other_data.receipt  # but the data is not the same
    assert same_answer.receipt.data != other_data.receipt.data
    everything.close()
    two.close()


def test_a_changed_hand_label_changes_the_fingerprint(tmp_path):
    plain = tmp_path / "plain"
    labelled = tmp_path / "labelled"
    plain.mkdir()
    labelled.mkdir()
    without = build_from(plain, ("1550709",))
    with_label = build_from(labelled, ("1550709",), hand=[thushara_by_hand()])
    assert data_fingerprint(without) != data_fingerprint(with_label)
    without.close()
    with_label.close()


def test_every_engine_gives_receipts(db):
    engines = engines_on(db)
    ground = CricketSubject(
        venue="Rawalpindi Cricket Stadium", match_format="T20", gender="male"
    )
    answers = [
        engines.fact("economy", CricketSubject(bowler="E Malinga")),
        engines.fact("venue_first_innings_avg", ground),
        engines.fact(
            "workload_ratio_flag", CricketSubject(bowler="E Malinga", date="2026-10-11")
        ),
        engines.fact(
            "review_overturn_rate", CricketSubject(team="Australia", gender="female")
        ),
        engines.fact("team_win_share", CricketSubject(team="Pakistan", gender="male")),
    ]
    engine_names = []
    for answer in answers:
        assert answer.receipt.value == answer.value
        engine_names.append(answer.receipt.engine)
    assert engine_names == ["N8.M1", "N8.M2", "N8.M3", "N8.M4", "N8.M5"]
