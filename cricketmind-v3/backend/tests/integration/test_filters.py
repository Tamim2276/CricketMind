"""Day 43: one measure, many filters, on the tiny DuckDB made from the three fixtures.

Counted by hand from the files:
- Sahibzada Farhan: 73 off 44, all at Rawalpindi, a men's T20I of season 2026/27 on
  2026-10-11, batting first for Pakistan against Sri Lanka.
- MD Shanaka: 5 off 3 for Rajasthan Royals against Sunrisers Hyderabad (IPL, New
  Chandigarh, 2026-05-27, season 2026), and 9 off 8 for Sri Lanka at Rawalpindi.
- E Malinga bowled 26 balls at Rawalpindi (innings 1, to Pakistan) and 25 in the IPL
  match (innings 1, to Rajasthan Royals). At Rawalpindi he bowled 7 to Farhan (right-
  handed in the sample file) and 1 to Fakhar Zaman (left-handed).
"""

import dataclasses

import pytest

from cricketmind_v3.application.engines.m1_rates import strike_rate
from cricketmind_v3.application.engines.m2_venues import venue_first_innings_avg
from cricketmind_v3.application.engines.m4_reviews import review_overturn_rate
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from cricketmind_v3.infrastructure.duckdb_reviews import DuckDBReviewRepository
from cricketmind_v3.infrastructure.duckdb_venues import DuckDBVenueRepository
from tests.integration.test_duckdb_repo import build

FARHAN = CricketSubject(player="Sahibzada Farhan")
SHANAKA = CricketSubject(player="MD Shanaka")
MALINGA = CricketSubject(bowler="E Malinga")


@pytest.fixture
def db(tmp_path):
    connection = build(tmp_path)
    yield connection
    connection.close()


@pytest.fixture
def balls(db):
    return DuckDBDeliveryRepository(db)


def farhan(**filters) -> CricketSubject:
    return dataclasses.replace(FARHAN, **filters)


def shanaka(**filters) -> CricketSubject:
    return dataclasses.replace(SHANAKA, **filters)


# The plan's test: the same measure under different filters


@pytest.mark.parametrize(
    "filters",
    [
        {},
        {"opposition": "Sri Lanka"},
        {"venue": "Rawalpindi Cricket Stadium"},
        {"season": "2026/27"},
        {"since": "2026"},
        {"since": "2026-10-11"},  # the day itself counts
        {"before": "2026-10-12"},  # Day 53: up to the day before
        {"before": "2027"},
        {"match_format": "T20I"},
        {"match_format": "T20"},
        {"innings": 1},
        {"gender": "male"},
        {"opposition": "Sri Lanka", "match_format": "T20I", "innings": 1},
    ],
)
def test_every_filter_that_fits_keeps_all_his_balls(balls, filters):
    assert strike_rate(balls, farhan(**filters)) == Fact(value=100.0 * 73 / 44, n=44)


@pytest.mark.parametrize(
    "filters",
    [
        {"opposition": "Pakistan"},  # his own side
        {"opposition": "India"},
        {"venue": "Lord's"},
        {"season": "2026"},
        {"since": "2026-10-12"},
        {"before": "2026-10-11"},  # the day itself is not before it
        {"before": "2026"},
        {"match_format": "ODI"},
        {"innings": 2},
        {"gender": "female"},
    ],
)
def test_every_filter_that_does_not_fit_leaves_nothing(balls, filters):
    assert strike_rate(balls, farhan(**filters)) is None


def test_filters_split_one_players_matches(balls):
    # Shanaka: 3 balls in the IPL, 8 in the T20I
    assert strike_rate(balls, shanaka()) == Fact(value=100.0 * 14 / 11, n=11)
    assert strike_rate(balls, shanaka(match_format="T20I")) == Fact(value=112.5, n=8)
    assert strike_rate(balls, shanaka(season="2026")) == Fact(value=100.0 * 5 / 3, n=3)
    ipl = shanaka(opposition="Sunrisers Hyderabad")
    assert strike_rate(balls, ipl) == Fact(value=100.0 * 5 / 3, n=3)
    ground = shanaka(venue="Maharaja Yadavindra Singh International Cricket Stadium")
    assert strike_rate(balls, ground).n == 3


def test_a_field_the_measure_cannot_filter_by_gets_no_answer(balls):
    assert strike_rate(balls, farhan(date="2026-10-11")) is None
    assert strike_rate(balls, farhan(batting_hand="right")) is None  # a bowler's filter
    assert strike_rate(balls, farhan(match_format="Rugby")) is None
    assert strike_rate(balls, farhan(since="last year")) is None
    assert strike_rate(balls, farhan(before="next year")) is None


# Seen from the bowler's side: opposition is who batted


def test_a_bowlers_opposition_is_the_side_batting(balls):
    to_pakistan = dataclasses.replace(MALINGA, opposition="Pakistan")
    assert len(balls.deliveries(to_pakistan)) == 26
    to_rajasthan = dataclasses.replace(MALINGA, opposition="Rajasthan Royals")
    assert len(balls.deliveries(to_rajasthan)) == 25
    assert len(balls.deliveries(dataclasses.replace(MALINGA, innings=1))) == 51
    assert balls.deliveries(dataclasses.replace(MALINGA, innings=2)) == []


def test_a_bowler_against_left_and_right_handers(balls):
    at_rawalpindi = dataclasses.replace(MALINGA, venue="Rawalpindi Cricket Stadium")
    right = dataclasses.replace(at_rawalpindi, batting_hand="right")
    left = dataclasses.replace(at_rawalpindi, batting_hand="left")
    assert len(balls.deliveries(right)) == 7  # Farhan
    assert len(balls.deliveries(left)) == 1  # Fakhar Zaman


# The same filters on M2 and M4


def test_a_grounds_profile_by_season(db):
    venues = DuckDBVenueRepository(db)
    lords = CricketSubject(venue="Lord's", match_format="T20I", gender="female")
    assert venue_first_innings_avg(venues, lords) == Fact(value=150.0, n=1)
    in_2026 = dataclasses.replace(lords, season="2026")
    assert venue_first_innings_avg(venues, in_2026) == Fact(value=150.0, n=1)
    assert (
        venue_first_innings_avg(venues, dataclasses.replace(lords, since="2027"))
        is None
    )
    assert (
        venue_first_innings_avg(venues, dataclasses.replace(lords, match_format="ODI"))
        is None
    )


def test_reviews_by_side(db):
    # Australia's women: as batters, Voll's review (upheld); as fielders, Wyatt's
    # (upheld) and Capsey's (struck down)
    reviews = DuckDBReviewRepository(db)
    australia = CricketSubject(team="Australia", gender="female")
    batting = dataclasses.replace(australia, side="batting")
    fielding = dataclasses.replace(australia, side="fielding")
    assert review_overturn_rate(reviews, batting) == Fact(value=1.0, n=1)
    assert review_overturn_rate(reviews, fielding) == Fact(value=0.5, n=2)
    keeping = dataclasses.replace(australia, side="wicketkeeping")
    assert review_overturn_rate(reviews, keeping) is None
    t20i = dataclasses.replace(australia, match_format="T20I")
    assert review_overturn_rate(reviews, t20i) == Fact(value=2 / 3, n=3)


def test_a_filter_the_reviews_row_does_not_list_gets_no_answer(db):
    # the adapter could pick reviews by ground; M4's menu row does not offer it
    reviews = DuckDBReviewRepository(db)
    at_lords = CricketSubject(team="Australia", gender="female", venue="Lord's")
    assert review_overturn_rate(reviews, at_lords) is None
