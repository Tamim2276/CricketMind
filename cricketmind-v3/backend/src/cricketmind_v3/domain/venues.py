"""What M2 measures about a ground (Day 40): the rules, over its innings and wickets."""

from dataclasses import dataclass

from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.faith.gazetteer import BOWLER_TYPES

# balls in a full innings of each limited-overs format. A match of any other length is
# not counted: The Hundred's 100 balls, or a match cut to 10 overs before the start
STANDARD_BALLS = {"T20": 120, "IT20": 120, "ODI": 300, "ODM": 300}
LONG_FORMATS = ("Test", "MDM")  # no fixed length

ALL_OUT = 10

# the dismissals a bowler is credited with: not run outs, retirements or obstruction
BOWLERS_WICKETS = (
    "bowled",
    "caught",
    "caught and bowled",
    "lbw",
    "stumped",
    "hit wicket",
)

# a pace-or-spin share needs the bowler's type for this much of the wickets
# (a design parameter, D-043)
KNOWN_ENOUGH = 0.8


@dataclass(frozen=True, kw_only=True)
class InningsScore:
    """One innings at the ground, as M2 counts it."""

    match_type: str  # "T20"
    scheduled_balls: int | None  # 120 in a T20; None in a Test
    innings: int  # 1: batted first
    runs: int  # extras and penalty runs included
    wickets: int  # dismissals, not retirements
    legal_balls: int
    declared: bool
    won: bool | None  # None when the match had no winner: a tie, no result, a draw


@dataclass(frozen=True, kw_only=True)
class Wicket:
    """A kind of wicket that fell at the ground, to be counted."""

    match_type: str
    scheduled_balls: int | None
    bowler_type: str | None  # "pace" or "spin"; None when not known
    kind: str  # "caught", "run out", ...


def standard(match_type: str, scheduled_balls: int | None) -> bool:
    """A match of the format's usual length: 120 balls an innings in a T20."""
    if match_type in LONG_FORMATS:
        return True
    return STANDARD_BALLS.get(match_type) == scheduled_balls


def complete(score: InningsScore) -> bool:
    """The innings ran its full course: all out, every ball batted, or declared."""
    if score.wickets >= ALL_OUT:
        return True
    if score.match_type in LONG_FORMATS:
        return score.declared
    return score.legal_balls >= score.scheduled_balls


def first_innings_average(scores: list[InningsScore]) -> Fact | None:
    """Runs per first innings; n is the innings counted.

    Only full innings of standard matches: one cut short by rain would pull the
    average down, though the ground did nothing.
    """
    runs = 0
    counted = 0
    for score in scores:
        if score.innings != 1 or not standard(score.match_type, score.scheduled_balls):
            continue
        if not complete(score):
            continue
        runs += score.runs
        counted += 1
    if counted == 0:
        return None
    return Fact(value=runs / counted, n=counted)


def chasing_win_share(scores: list[InningsScore]) -> Fact | None:
    """Of the limited-overs matches with a winner, the share won by the side batting
    second; n is those chases. A rain-rule result counts: it has a winner."""
    chases = 0
    won = 0
    for score in scores:
        if score.innings != 2 or score.match_type in LONG_FORMATS:
            continue  # in a Test, batting second is not chasing
        if not standard(score.match_type, score.scheduled_balls):
            continue
        if score.won is None:
            continue  # a tie or no result
        chases += 1
        if score.won:
            won += 1
    if chases == 0:
        return None
    return Fact(value=won / chases, n=chases)


def wicket_share(wickets: dict[Wicket, int], bowler_type: str) -> Fact | None:
    """The share of the bowlers' wickets taken by pace, or by spin; n is the wickets
    whose bowler's type is known."""
    if bowler_type not in BOWLER_TYPES:
        return None  # "medium" is not a type the data knows: no answer, not 0%
    known = 0
    unknown = 0
    this_type = 0
    for wicket, times in wickets.items():
        if wicket.kind not in BOWLERS_WICKETS:
            continue
        if not standard(wicket.match_type, wicket.scheduled_balls):
            continue
        if wicket.bowler_type is None:
            unknown += times
            continue
        known += times
        if wicket.bowler_type == bowler_type:
            this_type += times
    if known == 0 or known < KNOWN_ENOUGH * (known + unknown):
        return None  # too many bowlers of unknown type to trust a share
    return Fact(value=this_type / known, n=known)
