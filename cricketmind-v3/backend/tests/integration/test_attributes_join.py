"""Days 34-35: attributes stored, joined to every ball, and the coverage counted."""

from pathlib import Path

import duckdb
import pytest

from cricketmind_v3.domain.names import find_players
from cricketmind_v3.pipeline.attributes import (
    Attributes,
    build_attributes,
    coverage,
    coverage_report,
    read_player_meta,
)
from cricketmind_v3.pipeline.ingest import ingest
from cricketmind_v3.pipeline.parse_match import read_match
from cricketmind_v3.pipeline.players import build_players, load_players
from tests.integration.test_ingest import BALLS, FIXTURES, IDS, make_zip, quiet

SAMPLE = FIXTURES.parent / "player_meta_sample.csv"

# the sample's bowler types, by the name used in the matches (all three fixtures)
TYPED = {
    "E Malinga": "pace",
    "M Theekshana": "spin",
    "PHKD Mendis": "spin",
    "Abrar Ahmed": "spin",
    "Fakhar Zaman": "spin",
    "B Fernando": "pace",
    "Shadab Khan": "spin",
}


def thushara_by_hand() -> Attributes:
    return Attributes(
        person_id="ee1b6c27",  # N Thushara: not in the sample
        known_as=None,
        full_name=None,
        country=None,
        batting_hand="right",
        bowler_type="pace",
        batting_style=None,
        bowling_style=None,
        source="hand-labelled",
    )


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "cricket.duckdb"
    ingest(make_zip(tmp_path), path, say=quiet)
    connection = duckdb.connect(str(path))
    build_players(connection)
    build_attributes(connection, read_player_meta(SAMPLE), [])
    yield connection
    connection.close()


# The plan's test (Day 35)


def test_the_join_keeps_every_delivery(db):
    joined = db.execute("SELECT COUNT(*) FROM deliveries_full").fetchone()[0]
    assert joined == BALLS  # no ball lost, none doubled


def test_one_bowler_two_teams_keeps_his_type(db):
    types = db.execute(
        "SELECT DISTINCT match_id, bowler_type FROM deliveries_full "
        "WHERE bowler = 'E Malinga' ORDER BY match_id"
    ).fetchall()
    assert types == [("1535463", "pace"), ("1550709", "pace")]


def test_the_coverage_is_counted_ball_by_ball(db):
    # counted again here, from the files, not from the database
    expected = 0
    for match_id in IDS:
        for row in read_match(Path(FIXTURES / f"{match_id}.json")).deliveries:
            if row.bowler in TYPED:
                expected += 1
    found = coverage(db)
    assert found.balls == BALLS
    assert found.with_bowler_type == expected


def test_the_unknown_bowlers_come_most_balls_first(db):
    unknown = coverage(db).unknown_bowlers
    balls = []
    for _, _, count in unknown:
        balls.append(count)
    assert balls == sorted(balls, reverse=True)
    names = []
    for _, name, _ in unknown:
        names.append(name)
    assert "N Thushara" in names  # no type yet


def test_a_hand_label_fills_a_gap(db):
    build_attributes(db, read_player_meta(SAMPLE), [thushara_by_hand()])
    types = db.execute(
        "SELECT DISTINCT bowler_type FROM deliveries_full WHERE bowler = 'N Thushara'"
    ).fetchall()
    assert types == [("pace",)]


def test_a_hand_label_wins_over_the_download(db):
    wrong_in_source = Attributes(
        person_id="5750bcb4",  # E Malinga
        known_as=None,
        full_name=None,
        country=None,
        batting_hand="right",
        bowler_type="spin",
        batting_style=None,
        bowling_style=None,
        source="hand-labelled",
    )
    build_attributes(db, read_player_meta(SAMPLE), [wrong_in_source])
    row = db.execute(
        "SELECT bowler_type, known_as, source FROM attributes WHERE person_id = '5750bcb4'"
    ).fetchone()
    assert row == ("spin", "Eshan Malinga", "hand-labelled")  # the name is kept


def test_full_names_become_names_to_search_by(db):
    players = load_players(db)
    assert find_players("Eshan Malinga", players)[0].name == "E Malinga"
    assert find_players("Kamindu Mendis", players)[0].name == "PHKD Mendis"


def test_the_report_reads_well(db):
    text = coverage_report(coverage(db))
    assert text.startswith(f"balls: {BALLS}")
    assert "| T20 | male |" in text
