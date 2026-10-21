"""Day 36: the derived tables, checked against counts made from the files directly."""

import duckdb
import pytest

from cricketmind_v3.pipeline.derived import build_derived
from cricketmind_v3.pipeline.ingest import ingest
from cricketmind_v3.pipeline.parse_match import read_match
from tests.integration.test_ingest import FIXTURES, IDS, make_zip, quiet


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "cricket.duckdb"
    ingest(make_zip(tmp_path), path, say=quiet)
    connection = duckdb.connect(str(path))
    build_derived(connection)
    yield connection
    connection.close()


def every_ball():
    balls = []
    for match_id in IDS:
        balls.extend(read_match(FIXTURES / f"{match_id}.json").deliveries)
    return balls


# Innings scores (M2)


def test_one_row_per_innings(db):
    assert db.execute("SELECT COUNT(*) FROM innings_scores").fetchone()[0] == 6


def test_scores_wickets_and_results(db):
    rows = db.execute(
        "SELECT innings, team, runs, wickets, legal_balls, won, chasing "
        "FROM innings_scores WHERE match_id = '1550709' ORDER BY innings"
    ).fetchall()
    assert rows == [
        (1, "Pakistan", 215, 3, 120, True, False),
        (2, "Sri Lanka", 133, 7, 120, False, True),
    ]


def test_a_successful_chase(db):
    row = db.execute(
        "SELECT team, wickets, won, chasing FROM innings_scores "
        "WHERE match_id = '1490709' AND innings = 2"
    ).fetchone()
    assert row == ("Australia", 3, True, True)  # won by 7 wickets


def test_retired_hurt_is_not_a_wicket(db):
    db.execute(
        "UPDATE deliveries SET wickets = 1, wicket_kind = 'retired hurt' "
        "WHERE match_id = '1550709' AND innings = 1 AND over = 0 AND ball_index = 1"
    )
    build_derived(db)
    wickets = db.execute(
        "SELECT wickets FROM innings_scores WHERE match_id = '1550709' AND innings = 1"
    ).fetchone()[0]
    assert wickets == 3  # unchanged: the batter could come back


# Venues


def test_one_name_per_ground(db):
    keys = db.execute("SELECT venue_key FROM venues ORDER BY venue_key").fetchall()
    assert keys == [
        ("Lord's",),  # was "Lord's, London"
        ("Maharaja Yadavindra Singh International Cricket Stadium",),
        ("Rawalpindi Cricket Stadium",),
    ]


def test_two_spellings_of_one_ground_are_one_venue(db):
    db.execute("UPDATE matches SET venue = 'Lord''s' WHERE match_id = '1550709'")
    build_derived(db)
    row = db.execute(
        "SELECT matches, names_used FROM venues WHERE venue_key = 'Lord''s'"
    ).fetchone()
    assert row == (2, 2)


# Bowler days (M3)


def test_balls_bowled_per_bowler_per_day(db):
    # E Malinga bowled in two of the matches, on two different days
    expected = {}
    for ball in every_ball():
        if ball.bowler == "E Malinga" and ball.wides == 0 and ball.noballs == 0:
            expected[ball.date] = expected.get(ball.date, 0) + 1
    rows = db.execute(
        "SELECT CAST(date AS VARCHAR), legal_balls FROM bowler_days "
        "WHERE bowler = 'E Malinga' ORDER BY date"
    ).fetchall()
    assert dict(rows) == expected
    assert len(rows) == 2


def test_every_ball_is_counted_once(db):
    total = db.execute("SELECT SUM(deliveries) FROM bowler_days").fetchone()[0]
    assert total == len(every_ball())


# Reviews (M4)


def test_one_row_per_review(db):
    reviewed = 0
    for ball in every_ball():
        if ball.review_by is not None:
            reviewed += 1
    assert db.execute("SELECT COUNT(*) FROM reviews").fetchone()[0] == reviewed


def test_an_upheld_review_overturned_the_decision(db):
    # Australia, fielding, reviewed Wyatt's not-out; upheld, so she was out caught
    row = db.execute(
        "SELECT by_batting_side, overturned, out_on_this_ball, wicket_kind "
        "FROM reviews WHERE match_id = '1490709' AND review_batter = 'DN Wyatt'"
    ).fetchone()
    assert row == (False, True, True, "caught")


def test_grounds_that_share_a_name_stay_apart(db):
    # found on Day 36: England has a "County Ground" in Bristol, Taunton and more
    db.execute(
        "UPDATE matches SET venue = 'County Ground, Bristol' WHERE match_id = '1550709'"
    )
    db.execute(
        "UPDATE matches SET venue = 'County Ground, Taunton' WHERE match_id = '1535463'"
    )
    build_derived(db)
    keys = db.execute(
        "SELECT venue_key FROM venues WHERE venue_key LIKE 'County Ground%' ORDER BY 1"
    ).fetchall()
    assert keys == [("County Ground, Bristol",), ("County Ground, Taunton",)]
