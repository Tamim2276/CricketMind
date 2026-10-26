"""Scoring chances against what happened (Day 53).

The weather forecast: on all the days it said 70% rain, it should have rained on about 70%
of them. That is calibration. The Brier score asks how close each chance was to what
happened: 0 is perfect, and always saying 50% scores 0.25.
"""

import math
import random
from dataclasses import dataclass

BINS = 10  # bins of equal size, by chance
RESAMPLES = 1_000  # bootstrap resamples of whole matches
SEED = 42
EDGE = 1e-12  # a chance of exactly 0 or 1 would make the log loss infinite


@dataclass(frozen=True, kw_only=True)
class Bin:
    """One bin of the reliability plot."""

    count: int
    chance: float  # the mean chance given
    observed: float  # the share that happened


def brier(chances: list[float], outcomes: list[int]) -> float:
    """The mean of (chance - outcome) squared: lower is better."""
    total = 0.0
    for chance, outcome in zip(chances, outcomes, strict=True):
        total += (chance - outcome) ** 2
    return total / len(chances)


def log_loss(chances: list[float], outcomes: list[int]) -> float:
    """How surprised the chances were by what happened: lower is better."""
    total = 0.0
    for chance, outcome in zip(chances, outcomes, strict=True):
        chance = min(max(chance, EDGE), 1 - EDGE)
        if outcome:
            total -= math.log(chance)
        else:
            total -= math.log(1 - chance)
    return total / len(chances)


def bin_of(chances: list[float], bins: int = BINS) -> list[int]:
    """Each chance's bin: sorted by chance and cut into bins of (nearly) equal size."""
    order = []
    for i, chance in enumerate(chances):
        order.append((chance, i))
    order.sort()
    found = [0] * len(chances)
    for rank, (_, i) in enumerate(order):
        found[i] = rank * bins // len(chances)
    return found


def reliability(
    chances: list[float], outcomes: list[int], bins: int = BINS
) -> list[Bin]:
    """The reliability plot's points: in each bin, the mean chance and the share out."""
    counts = [0] * bins
    chance_sums = [0.0] * bins
    outcome_sums = [0] * bins
    for chance, outcome, where in zip(
        chances, outcomes, bin_of(chances, bins), strict=True
    ):
        counts[where] += 1
        chance_sums[where] += chance
        outcome_sums[where] += outcome
    found = []
    for where in range(bins):
        if counts[where]:
            found.append(
                Bin(
                    count=counts[where],
                    chance=chance_sums[where] / counts[where],
                    observed=outcome_sums[where] / counts[where],
                )
            )
    return found


def ece(bins: list[Bin]) -> float:
    """The calibration gap: the size-weighted mean of |share out - mean chance|."""
    total = 0
    gap = 0.0
    for one in bins:
        total += one.count
        gap += one.count * abs(one.observed - one.chance)
    return gap / total


def per_match(
    matches: list[str],
    chances: list[float],
    outcomes: list[int],
    bins: list[int],
    width: int = BINS,
) -> dict[str, list[float]]:
    """What each match adds to the scores: balls, squared error, then for each bin its
    chances' sum and its outcomes' sum. Resampling matches needs only these."""
    found = {}
    for match, chance, outcome, where in zip(
        matches, chances, outcomes, bins, strict=True
    ):
        if match not in found:
            found[match] = [0.0] * (2 + 2 * width)
        row = found[match]
        row[0] += 1
        row[1] += (chance - outcome) ** 2
        row[2 + 2 * where] += chance
        row[3 + 2 * where] += outcome
    return found


def scores_from(totals: list[float]) -> tuple[float, float]:
    """The Brier score and the calibration gap, from summed match rows."""
    balls = totals[0]
    gap = 0.0
    for where in range(2, len(totals), 2):
        gap += abs(totals[where + 1] - totals[where])  # count x |share - chance|
    return totals[1] / balls, gap / balls


def paired_bootstrap(
    tables: dict[str, dict[str, list[float]]],
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> dict[str, list[tuple[float, float]]]:
    """Each model's (Brier, calibration gap) in every resample of whole matches. The same
    matches are drawn for every model, so their differences are fair (paired)."""
    names = sorted(tables)
    matches = sorted(tables[names[0]])
    for name in names:
        if sorted(tables[name]) != matches:
            raise ValueError("every model must be scored on the same matches")
    rng = random.Random(seed)
    found = {}
    for name in names:
        found[name] = []
    for _ in range(resamples):
        times = {}
        for _ in matches:
            match = matches[rng.randrange(len(matches))]
            times[match] = times.get(match, 0) + 1
        for name in names:
            width = len(tables[name][matches[0]])
            totals = [0.0] * width
            for match, count in times.items():
                row = tables[name][match]
                for i in range(width):
                    totals[i] += count * row[i]
            found[name].append(scores_from(totals))
    return found


def keeps(brier_range: tuple[float, float], gap_range: tuple[float, float]) -> bool:
    """The rule of Day 53's pre-registration: the new model is kept only if its Brier
    score is lower by more than luck (the 95% range of new - old wholly below zero) and
    its calibration is not clearly worse (that range for the gap not wholly above zero)."""
    _, brier_high = brier_range
    gap_low, _ = gap_range
    return brier_high < 0 and gap_low <= 0
