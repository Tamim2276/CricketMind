"""What M1 measures from a tally of balls (Days 38-44). Each gives a Fact, or None."""

from cricketmind_v3.domain.balls import Ball, batter_out, bowlers_wicket, faced, legal
from cricketmind_v3.domain.facts import Fact


def strike_rate(counts: dict[Ball, int]) -> Fact | None:
    """Runs off the bat per 100 balls faced; n is the balls faced."""
    balls_faced = 0
    runs = 0
    for ball, times in counts.items():
        runs += ball.runs_batter * times
        if faced(ball):
            balls_faced += times
    if balls_faced == 0:
        return None  # no ball faced: there is no rate to give
    return Fact(value=100.0 * runs / balls_faced, n=balls_faced)


def dismissal_rate(counts: dict[Ball, int]) -> Fact | None:
    """Times out per ball faced: 4 outs in 50 balls is 0.08. n is the balls faced.

    An out on a wide (stumped, run out) still counts, as it does in a batting average.
    """
    outs, balls_faced = outs_and_faced(counts)
    if balls_faced == 0:
        return None
    return Fact(value=outs / balls_faced, n=balls_faced)


def outs_and_faced(counts: dict[Ball, int]) -> tuple[int, int]:
    """His times out, and his balls faced: what the dismissal rate and the chance of
    being out (N9, Day 49) are both made from."""
    outs = 0
    balls_faced = 0
    for ball, times in counts.items():
        if batter_out(ball):
            outs += times
        if faced(ball):
            balls_faced += times
    return outs, balls_faced


# More batting measures (Day 44): n is always the balls faced


def runs(counts: dict[Ball, int]) -> Fact | None:
    """Runs off the bat."""
    faced_balls = _faced(counts)
    if faced_balls == 0:
        return None
    total = 0
    for ball, times in counts.items():
        total += ball.runs_batter * times
    return Fact(value=float(total), n=faced_balls)


def batting_average(counts: dict[Ball, int], out_at_other_end: int) -> Fact | None:
    """Runs per time out. A run out at the non-striker's end is a dismissal too, so it
    comes in from outside the balls he faced. Never out: no average."""
    faced_balls = _faced(counts)
    total = 0
    outs = out_at_other_end
    for ball, times in counts.items():
        total += ball.runs_batter * times
        if batter_out(ball):
            outs += times
    if faced_balls == 0 or outs == 0:
        return None
    return Fact(value=total / outs, n=faced_balls)


def boundary_share(counts: dict[Ball, int]) -> Fact | None:
    """The share of balls faced hit for four or six."""
    return _share_of_faced(counts, (4, 6))


def dot_ball_share(counts: dict[Ball, int]) -> Fact | None:
    """The share of balls faced he scored nothing from (byes and leg byes are not his)."""
    return _share_of_faced(counts, (0,))


def _share_of_faced(counts: dict[Ball, int], scores: tuple[int, ...]) -> Fact | None:
    faced_balls = _faced(counts)
    if faced_balls == 0:
        return None
    hits = 0
    for ball, times in counts.items():
        if faced(ball) and ball.runs_batter in scores:
            hits += times
    return Fact(value=hits / faced_balls, n=faced_balls)


def _faced(counts: dict[Ball, int]) -> int:
    total = 0
    for ball, times in counts.items():
        if faced(ball):
            total += times
    return total


# Bowling measures (Day 44): n is always the legal balls bowled


def wickets(counts: dict[Ball, int]) -> Fact | None:
    """Wickets credited to the bowler."""
    legal_balls = _legal(counts)
    if legal_balls == 0:
        return None
    taken = 0
    for ball, times in counts.items():
        if bowlers_wicket(ball):
            taken += times
    return Fact(value=float(taken), n=legal_balls)


def economy(counts: dict[Ball, int]) -> Fact | None:
    """Runs conceded per six legal balls: off the bat, wides and no-balls. Byes and leg
    byes are the keeper's and fielders', not the bowler's."""
    legal_balls = _legal(counts)
    if legal_balls == 0:
        return None
    conceded = 0
    for ball, times in counts.items():
        conceded += (ball.runs_batter + ball.wides + ball.noballs) * times
    return Fact(value=6 * conceded / legal_balls, n=legal_balls)


def bowling_strike_rate(counts: dict[Ball, int]) -> Fact | None:
    """Legal balls per wicket. No wicket: no strike rate."""
    taken = wickets(counts)
    if taken is None or taken.value == 0:
        return None
    return Fact(value=taken.n / taken.value, n=taken.n)


def _legal(counts: dict[Ball, int]) -> int:
    total = 0
    for ball, times in counts.items():
        if legal(ball):
            total += times
    return total
