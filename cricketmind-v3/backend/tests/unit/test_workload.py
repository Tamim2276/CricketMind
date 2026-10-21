"""Day 41: a bowler's workload ratio, on a made-up calendar."""

from datetime import date

from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.workload import (
    Spell,
    balls_by_day,
    balls_in,
    flag,
    workload_ratio,
)


def one_day(day: date, balls: int) -> Spell:
    # a T20 or an ODI: every ball on the one day
    return Spell(first_day=day, days=1, match_balls=250, positions=tuple(range(balls)))


def calendar() -> list[Spell]:
    """Made up: a bowler who bowled 120 balls in the week up to 1 March, against 240 in
    the four weeks (a usual week of 60), and once long before."""
    return [
        one_day(date(2025, 1, 1), 24),  # long before: the data knows him
        one_day(date(2025, 2, 5), 60),
        one_day(date(2025, 2, 12), 60),
        one_day(date(2025, 2, 25), 60),  # this week
        one_day(date(2025, 2, 28), 60),  # this week
    ]


# The plan's test


def test_the_made_up_calendar():
    found = workload_ratio(calendar(), date(2025, 3, 1))
    assert found == Fact(value=2.0, n=240)  # 120 / (240 / 4)
    assert flag(found.value) == "a spike above his usual week"


def test_the_windows_include_the_day_asked_about():
    by_day = balls_by_day(calendar())
    assert balls_in(by_day, date(2025, 3, 1), 7) == 120  # 23 Feb to 1 Mar
    assert balls_in(by_day, date(2025, 2, 28), 1) == 60
    assert balls_in(by_day, date(2025, 3, 1), 28) == 240  # 2 Feb to 1 Mar


# Multi-day matches


def test_a_tests_balls_are_spread_over_its_days():
    test = Spell(
        first_day=date(2024, 12, 26),
        days=5,
        match_balls=2000,  # 400 a day
        positions=(0, 399, 400, 1999),
    )
    assert balls_by_day([test]) == {
        date(2024, 12, 26): 2,
        date(2024, 12, 27): 1,
        date(2024, 12, 30): 1,
    }


def test_a_boxing_day_test_counts_in_the_week_after():
    # 200 of his balls through a five-day Test from 26 December: 40 a day. Put on the
    # first day, none would fall in the week up to 3 January; spread, 28-30 December do
    test = Spell(
        first_day=date(2024, 12, 26),
        days=5,
        match_balls=2000,
        positions=tuple(range(0, 2000, 10)),  # every tenth ball of the match
    )
    by_day = balls_by_day([test])
    assert balls_in(by_day, date(2025, 1, 3), 7) == 120


# The flag


def test_the_flag_follows_the_ratio():
    assert flag(1.51) == "a spike above his usual week"
    assert flag(1.5) == "close to his usual week"
    assert flag(1.0) == "close to his usual week"
    assert flag(0.8) == "close to his usual week"
    assert flag(0.79) == "well below his usual week"


# No answer


def test_no_balls_in_four_weeks_gives_no_answer():
    assert workload_ratio(calendar(), date(2025, 6, 1)) is None


def test_a_newcomer_has_no_usual_week():
    # all his balls inside the four weeks: his usual week is unknown, not zero
    newcomer = [one_day(date(2025, 2, 20), 24), one_day(date(2025, 2, 27), 24)]
    assert workload_ratio(newcomer, date(2025, 3, 1)) is None
