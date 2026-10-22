"""Day 44: the batting and bowling measures on the tiny DuckDB of the three fixtures.

Counted by hand from the files:
- Sahibzada Farhan: 73 off 44, 4 fours, 5 sixes, 12 dots, out once.
- Fakhar Zaman: 22 off 17, out once; on the first ball of the match he was the
  non-striker.
- E Malinga: 51 balls, 48 legal, 78 runs off the bat, 3 in wides, no no-balls, 2
  wickets (Maaz Sadaqat at Rawalpindi, MD Shanaka in the IPL). At Rawalpindi: 26 balls,
  24 legal, 39 off the bat, 2 in wides, 1 wicket; 7 balls to Farhan for 11.
"""

import pytest

from cricketmind_v3.application.engines.m1_rates import (
    batting_average,
    boundary_share,
    bowling_strike_rate,
    dismissal_rate,
    dot_ball_share,
    economy,
    runs,
    wickets,
)
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.render import shown
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from tests.integration.test_duckdb_repo import build

FARHAN = CricketSubject(player="Sahibzada Farhan")
FAKHAR = CricketSubject(player="Fakhar Zaman")
MALINGA = CricketSubject(bowler="E Malinga")
FIRST_BALL = "match_id = '1550709' AND innings = 1 AND over = 0 AND ball_index = 1"


@pytest.fixture
def db(tmp_path):
    connection = build(tmp_path)
    yield connection
    connection.close()


@pytest.fixture
def repo(db):
    return DuckDBDeliveryRepository(db)


# Batting


def test_a_batters_numbers(repo):
    assert runs(repo, FARHAN) == Fact(value=73.0, n=44)
    assert batting_average(repo, FARHAN) == Fact(value=73.0, n=44)
    assert boundary_share(repo, FARHAN) == Fact(value=9 / 44, n=44)
    assert dot_ball_share(repo, FARHAN) == Fact(value=12 / 44, n=44)


def test_a_run_out_at_the_other_end_halves_his_average(db, repo):
    assert batting_average(repo, FAKHAR) == Fact(value=22.0, n=17)
    db.execute(
        "UPDATE deliveries SET player_out = 'Fakhar Zaman', wicket_kind = 'run out', "
        f"wickets = 1 WHERE {FIRST_BALL}"
    )
    assert repo.outs_at_other_end(FAKHAR) == 1
    assert batting_average(repo, FAKHAR) == Fact(value=11.0, n=17)  # 22 / 2
    # the striker on that ball, Farhan, is untouched
    assert batting_average(repo, FARHAN) == Fact(value=73.0, n=44)
    assert dismissal_rate(repo, FARHAN) == Fact(value=1 / 44, n=44)


def test_retiring_at_the_other_end_is_not_out(db, repo):
    db.execute(
        "UPDATE deliveries SET player_out = 'Fakhar Zaman', "
        f"wicket_kind = 'retired hurt', wickets = 1 WHERE {FIRST_BALL}"
    )
    assert batting_average(repo, FAKHAR) == Fact(value=22.0, n=17)


# Bowling


def test_a_bowlers_numbers(repo):
    assert wickets(repo, MALINGA) == Fact(value=2.0, n=48)
    assert economy(repo, MALINGA) == Fact(value=6 * 81 / 48, n=48)  # 78 + 3
    assert bowling_strike_rate(repo, MALINGA) == Fact(value=24.0, n=48)
    assert shown("economy", economy(repo, MALINGA).value) == "10.13"  # 10.125 up


def test_a_bowlers_numbers_under_filters(repo):
    rawalpindi = CricketSubject(bowler="E Malinga", venue="Rawalpindi Cricket Stadium")
    assert wickets(repo, rawalpindi) == Fact(value=1.0, n=24)
    assert economy(repo, rawalpindi) == Fact(value=6 * 41 / 24, n=24)
    to_pakistan = CricketSubject(bowler="E Malinga", opposition="Pakistan")
    assert wickets(repo, to_pakistan) == Fact(value=1.0, n=24)
    right = CricketSubject(
        bowler="E Malinga", venue="Rawalpindi Cricket Stadium", batting_hand="right"
    )
    assert economy(repo, right) == Fact(value=6 * 11 / 7, n=7)


def test_a_measure_asks_for_its_own_person(repo):
    # wickets are a bowler's; strike rate filters are a batter's
    assert wickets(repo, FARHAN) is None
    assert runs(repo, MALINGA) is None
