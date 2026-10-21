"""What M1 measures from a tally of balls (Days 38-39). Each gives a Fact, or None."""

from cricketmind_v3.domain.balls import Ball, batter_out, faced
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
    balls_faced = 0
    outs = 0
    for ball, times in counts.items():
        if batter_out(ball):
            outs += times
        if faced(ball):
            balls_faced += times
    if balls_faced == 0:
        return None
    return Fact(value=outs / balls_faced, n=balls_faced)
