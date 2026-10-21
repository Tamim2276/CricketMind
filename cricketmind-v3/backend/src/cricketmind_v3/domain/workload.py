"""M3: a bowler's balls this week against his usual week, as a flag (Day 41).

Never an injury chance: the data has no injuries, and no nets or training (L-003).
"""

from dataclasses import dataclass
from datetime import date, timedelta

from cricketmind_v3.domain.facts import Fact

WEEK = 7  # days in "this week"
FOUR_WEEKS = 28  # the usual week is a quarter of these

# design parameters (D-044), after the cut-offs common in the workload studies
SPIKE = 1.5  # above: far more than his usual week
UNDERLOAD = 0.8  # below: far less


@dataclass(frozen=True, kw_only=True)
class Spell:
    """A bowler's balls in one match, and where each came in the match."""

    first_day: date
    days: int  # dates the match was played on: 1 for a T20, up to 5 for a Test
    match_balls: int  # every ball of the match, by both sides
    positions: tuple[int, ...]  # where each of his balls came, from 0


def balls_by_day(spells: list[Spell]) -> dict[date, int]:
    """Every ball he bowled, wides and no-balls too, on the day it was most likely bowled.

    Cricsheet gives a match's dates, not each ball's. A multi-day match's balls are
    spread over its days in the order they were bowled: a fifth to each day of a
    five-day Test. Putting them all on the first day hid Bumrah's 327 balls at
    Melbourne from the week before the Sydney Test (found on Day 41).
    """
    by_day = {}
    for spell in spells:
        for position in spell.positions:
            later = position * spell.days // spell.match_balls
            day = spell.first_day + timedelta(days=later)
            by_day[day] = by_day.get(day, 0) + 1
    return by_day


def balls_in(by_day: dict[date, int], on: date, days: int) -> int:
    """Balls bowled in the `days` days up to and including `on`."""
    first = on - timedelta(days=days - 1)
    total = 0
    for day, balls in by_day.items():
        if first <= day <= on:
            total += balls
    return total


def workload_ratio(spells: list[Spell], on: date) -> Fact | None:
    """This week's balls over a usual week of the last four; n is the four weeks' balls.

    No answer when he bowled nothing in the four weeks, or when the data knows nothing
    of him before them: a newcomer's usual week is unknown, not zero.
    """
    by_day = balls_by_day(spells)
    week = balls_in(by_day, on, WEEK)
    four_weeks = balls_in(by_day, on, FOUR_WEEKS)
    if four_weeks == 0:
        return None
    window_starts = on - timedelta(days=FOUR_WEEKS - 1)
    if not _bowled_before(by_day, window_starts):
        return None
    usual_week = four_weeks / (FOUR_WEEKS / WEEK)
    return Fact(value=week / usual_week, n=four_weeks)


def flag(ratio: float) -> str:
    """The words for a ratio: him against his own usual week. Never a chance of injury.

    Not "a normal load": after weeks of heavy bowling, a heavy week is his usual one.
    Bumrah's ratio was 1.32 the day he broke down at Sydney, after 626 balls in four
    weeks (found on Day 41); the four weeks' balls show beside the ratio.
    """
    if ratio > SPIKE:
        return "a spike above his usual week"
    if ratio < UNDERLOAD:
        return "well below his usual week"
    return "close to his usual week"


def _bowled_before(by_day: dict[date, int], day: date) -> bool:
    for bowled in by_day:
        if bowled < day:
            return True
    return False
