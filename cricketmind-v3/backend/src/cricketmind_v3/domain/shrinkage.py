"""Shrinkage (Day 50): a small record is pulled toward what is normal for batters like him.

Other batters' records under the same filters say two things: the normal out-rate, and
how much batters really differ. The less they differ, the harder he is pulled; the more
balls he has faced, the less he moves. 1 out in 2 balls is not a 50% risk.

Both are found by maximum likelihood (Day 51, D-060): the Beta curve under which the
other batters' records were the most likely. A batter bats until he is out, so his balls
depend on his outs; the likelihood does not mind that, but the method of moments, used on
Day 50, did.
"""

import math
from dataclasses import dataclass

FEW_BATTERS = 30  # fewer others than this: too few to tell what normal is
MOST = 1_000_000  # the most a prior can be worth: beyond it, more balls change nothing
ROUNDS = 5  # turns of fitting the normal rate, then the worth
STEPS = 40  # narrowing steps in each search
EDGE = 1e-6  # a rate is searched for between EDGE and 1 - EDGE


@dataclass(frozen=True, kw_only=True)
class Prior:
    """What we believe before his own balls: a Beta curve of believable out-rates,
    worth so many pretend outs and pretend balls survived."""

    outs: float
    survived: float

    def normal(self) -> float:
        """The out-rate it leans to: 7.6 outs in 138.6 balls is 5.5%."""
        return self.outs / (self.outs + self.survived)

    def worth(self) -> float:
        """How many of his own balls it counts as."""
        return self.outs + self.survived


JEFFREYS = Prior(outs=0.5, survived=0.5)  # no view at all, worth one ball (Day 49)


def population_prior(records: list[tuple[int, int]]) -> Prior | None:
    """The prior from other batters' (times out, balls faced); None if it cannot be told.

    Tries normal rates and worths, keeping the pair under which their records were the
    most likely: first the best worth for the normal rate so far, then the best normal
    rate for that worth, a few times over.
    """
    counts = {}  # (outs, faced) -> how many batters had that record
    batters = 0
    balls = 0
    outs = 0
    longer = False
    for out, faced in records:
        if faced < 1:
            continue
        batters += 1
        balls += faced
        outs += out
        if faced > 1:
            longer = True
        counts[(out, faced)] = counts.get((out, faced), 0) + 1
    if batters < FEW_BATTERS or outs == 0 or outs == balls:
        return None
    if not longer:
        return None  # one ball each: real and luck cannot be told apart

    normal = outs / balls  # a start; the balls-weighted rate leans to the good batters
    worth = 100.0

    def at_worth(log_worth: float) -> float:
        return likelihood(counts, normal, math.exp(log_worth))

    def at_normal(rate: float) -> float:
        return likelihood(counts, rate, worth)

    for _ in range(ROUNDS):
        worth = math.exp(best(at_worth, 0.0, math.log(MOST)))
        normal = best(at_normal, EDGE, 1 - EDGE)
    if worth > MOST / 2:
        # no sign that batters differ at all: pulling him all the way to normal would
        # give a range that is falsely sure (11 batters in the fixtures did this)
        return None
    return Prior(outs=normal * worth, survived=(1 - normal) * worth)


def likelihood(
    counts: dict[tuple[int, int], int], normal: float, worth: float
) -> float:
    """How likely these records are if each batter's rate came from this Beta curve:
    its logarithm, so adding replaces multiplying."""
    outs = normal * worth
    survived = (1 - normal) * worth
    base = log_beta(outs, survived)
    total = 0.0
    for (out, faced), times in counts.items():
        total += times * (log_beta(outs + out, survived + faced - out) - base)
    return total


def log_beta(a: float, b: float) -> float:
    """The logarithm of the Beta function, which sizes a Beta curve."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def best(score, low: float, high: float) -> float:
    """Where score is highest between low and high. Golden-section search: try two
    points, keep the part of the range around the better one, and repeat."""
    shrink = (math.sqrt(5) - 1) / 2  # 0.618: each step keeps this much of the range
    left = high - shrink * (high - low)
    right = low + shrink * (high - low)
    left_score = score(left)
    right_score = score(right)
    for _ in range(STEPS):
        if left_score > right_score:
            high = right
            right = left
            right_score = left_score
            left = high - shrink * (high - low)
            left_score = score(left)
        else:
            low = left
            left = right
            left_score = right_score
            right = low + shrink * (high - low)
            right_score = score(right)
    return (low + high) / 2


def shrunk_rate(prior: Prior, outs: int, faced: int) -> float:
    """His out-rate pulled toward normal: his outs and the prior's, over both their balls."""
    return (prior.outs + outs) / (prior.worth() + faced)
