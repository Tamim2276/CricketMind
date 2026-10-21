"""Day 40: the M2 engine, on the fake repository (no database)."""

import dataclasses

from cricketmind_v3.application.engines.m2_venues import (
    venue_chasing_win_share,
    venue_first_innings_avg,
    venue_wicket_share,
)
from cricketmind_v3.application.ports import FakeVenueRepository
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.domain.venues import InningsScore, Wicket

MIRPUR = "Shere Bangla National Stadium"
MIRPUR_T20 = CricketSubject(venue=MIRPUR, match_format="T20", gender="male")


def score(runs, innings, won) -> InningsScore:
    return InningsScore(
        match_type="T20",
        scheduled_balls=120,
        innings=innings,
        runs=runs,
        wickets=6,
        legal_balls=120,
        declared=False,
        won=won,
    )


def mirpur() -> FakeVenueRepository:
    repo = FakeVenueRepository()
    repo.add_innings(
        MIRPUR,
        "T20",
        "male",
        [score(140, 1, False), score(141, 2, True), score(160, 1, True)],
    )
    spin = Wicket(match_type="T20", scheduled_balls=120, bowler_type="spin", kind="lbw")
    pace = dataclasses.replace(spin, bowler_type="pace")
    repo.add_wickets(MIRPUR, "T20", "male", {spin: 3, pace: 1})
    return repo


def test_the_ground_in_one_format():
    repo = mirpur()
    assert venue_first_innings_avg(repo, MIRPUR_T20) == Fact(value=150.0, n=2)
    assert venue_chasing_win_share(repo, MIRPUR_T20) == Fact(value=1.0, n=1)
    spin = dataclasses.replace(MIRPUR_T20, bowler_type="spin")
    assert venue_wicket_share(repo, spin) == Fact(value=0.75, n=4)


def test_another_format_is_another_ground():
    odi = CricketSubject(venue=MIRPUR, match_format="ODI", gender="male")
    assert venue_first_innings_avg(mirpur(), odi) is None


def test_the_menu_row_must_be_filled():
    repo = mirpur()
    assert venue_first_innings_avg(repo, CricketSubject(venue=MIRPUR)) is None
    no_gender = CricketSubject(venue=MIRPUR, match_format="T20")
    assert venue_first_innings_avg(repo, no_gender) is None  # men's or women's?
    assert venue_wicket_share(repo, MIRPUR_T20) is None  # pace or spin?


def test_a_different_question_gets_no_answer():
    # "Bangladesh's first innings at Mirpur" is not the ground's average
    bangladesh = dataclasses.replace(MIRPUR_T20, team="Bangladesh")
    assert venue_first_innings_avg(mirpur(), bangladesh) is None
    shakib = dataclasses.replace(MIRPUR_T20, player="Shakib Al Hasan")
    assert venue_chasing_win_share(mirpur(), shakib) is None
