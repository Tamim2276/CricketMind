"""Day 45: batter against bowler from both sides, and match results, on the fixtures.

Counted by hand from the files:
- Sahibzada Farhan against M Theekshana: 25 off 10, no extras; against MD Shanaka: 14
  off 8, and Shanaka had him caught.
- E Malinga to MD Shanaka (IPL): one ball, and the wicket.
- Results: Australia's women beat England's by 7 wickets (2026-07-05); Rajasthan Royals
  beat Sunrisers Hyderabad by 47 runs (2026-05-27); Pakistan beat Sri Lanka by 82 runs
  (2026-10-11).
"""

import pytest

from cricketmind_v3.application.engines.m1_rates import (
    bowling_strike_rate,
    dismissal_rate,
    economy,
    strike_rate,
    wickets,
)
from cricketmind_v3.application.engines.m5_results import (
    match_won_by_runs,
    match_won_by_wickets,
    team_win_share,
)
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from cricketmind_v3.infrastructure.duckdb_results import DuckDBResultRepository
from tests.integration.test_duckdb_repo import build


@pytest.fixture
def db(tmp_path):
    connection = build(tmp_path)
    yield connection
    connection.close()


# Head to head: one pair of players, seen from either side


def test_the_batter_against_the_bowler(db):
    balls = DuckDBDeliveryRepository(db)
    farhan_v_theekshana = CricketSubject(
        player="Sahibzada Farhan", bowler="M Theekshana"
    )
    assert strike_rate(balls, farhan_v_theekshana) == Fact(value=250.0, n=10)
    # the same 10 balls from the bowler's side
    assert economy(balls, farhan_v_theekshana) == Fact(value=15.0, n=10)
    assert wickets(balls, farhan_v_theekshana) == Fact(value=0.0, n=10)


def test_who_got_whom(db):
    balls = DuckDBDeliveryRepository(db)
    farhan_v_shanaka = CricketSubject(player="Sahibzada Farhan", bowler="MD Shanaka")
    assert dismissal_rate(balls, farhan_v_shanaka) == Fact(value=1 / 8, n=8)
    assert wickets(balls, farhan_v_shanaka) == Fact(value=1.0, n=8)
    malinga_v_shanaka = CricketSubject(bowler="E Malinga", player="MD Shanaka")
    assert wickets(balls, malinga_v_shanaka) == Fact(value=1.0, n=1)
    assert bowling_strike_rate(balls, malinga_v_shanaka) == Fact(value=1.0, n=1)


# Match results


def match(team, opposition, date, gender) -> CricketSubject:
    return CricketSubject(team=team, opposition=opposition, date=date, gender=gender)


def test_the_margins(db):
    results = DuckDBResultRepository(db)
    pakistan = match("Pakistan", "Sri Lanka", "2026-10-11", "male")
    assert match_won_by_runs(results, pakistan) == Fact(value=82.0, n=1)
    assert match_won_by_wickets(results, pakistan) is None
    australia = match("Australia", "England", "2026-07-05", "female")
    assert match_won_by_wickets(results, australia) == Fact(value=7.0, n=1)
    rajasthan = match("Rajasthan Royals", "Sunrisers Hyderabad", "2026-05-27", "male")
    assert match_won_by_runs(results, rajasthan) == Fact(value=47.0, n=1)


def test_a_margin_for_the_wrong_winner_has_no_answer(db):
    results = DuckDBResultRepository(db)
    sri_lanka = match("Sri Lanka", "Pakistan", "2026-10-11", "male")
    assert match_won_by_runs(results, sri_lanka) is None  # Sri Lanka lost
    wrong_day = match("Pakistan", "Sri Lanka", "2026-10-12", "male")
    assert match_won_by_runs(results, wrong_day) is None
    womens = match("Pakistan", "Sri Lanka", "2026-10-11", "female")
    assert match_won_by_runs(results, womens) is None


def test_win_shares(db):
    results = DuckDBResultRepository(db)
    pakistan = CricketSubject(team="Pakistan", gender="male")
    assert team_win_share(results, pakistan) == Fact(value=1.0, n=1)
    sri_lanka = CricketSubject(team="Sri Lanka", gender="male", match_format="T20I")
    assert team_win_share(results, sri_lanka) == Fact(value=0.0, n=1)
    australia = CricketSubject(team="Australia", gender="male")
    assert team_win_share(results, australia) is None  # only the women played here


def test_a_tie_is_left_out(db):
    db.execute(
        "UPDATE matches SET winner = NULL, result = 'tie', won_by_runs = NULL "
        "WHERE match_id = '1550709'"
    )
    results = DuckDBResultRepository(db)
    pakistan = CricketSubject(team="Pakistan", gender="male")
    assert team_win_share(results, pakistan) is None
