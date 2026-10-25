"""N9 T1: the chance of being out within his next k balls, and how sure we can be (Day 49).

Two kinds of uncertainty. Luck: even at a known rate he may or may not be out. And the
rate itself: 4 outs in 50 balls only roughly says how risky he is. The Beta curve says
how believable each rate is; every replay draws a rate from it, then plays the balls.
"""

import random
from dataclasses import dataclass

from cricketmind_v3.domain.bootstrap import middle_95

DRAWS = 20_000  # replays behind every answer (as for N-004)
SEED = 42  # the same balls always give the same answer
PRIOR = 0.5  # Jeffreys prior, Beta(0.5, 0.5): before any ball, no view on the rate
PICTURED = 100  # overs kept for the picture (D-054)


@dataclass(frozen=True, kw_only=True)
class OutWithin:
    """What the replays say."""

    chance: float  # the middle answer: half the replays give more, half less
    low: float  # the middle 95% of the replays
    high: float
    survival: tuple[tuple[float, float, float], ...]  # still batting after each ball
    overs: tuple[int, ...]  # 100 representative overs: the ball he was out on, or 0


def believable_rates(
    outs: int, balls: int, rng: random.Random, draws: int
) -> list[float]:
    """Out-rates per ball, each drawn from Beta(0.5 + outs, 0.5 + balls not out on)."""
    rates = []
    for _ in range(draws):
        rates.append(rng.betavariate(PRIOR + outs, PRIOR + balls - outs))
    return rates


def out_within(
    outs: int, balls: int, k: int, draws: int = DRAWS, seed: int = SEED
) -> OutWithin | None:
    """The chance he is out within his next k balls faced, with its range.

    At one rate p, surviving k balls has the chance (1 - p) to the power k: what
    playing those balls over and over would give. Each replay's rate comes from the
    Beta curve, so the spread of the answers is the honest range.
    """
    if balls < 1 or k < 1 or not 0 <= outs <= balls:
        return None
    rng = random.Random(seed)
    rates = believable_rates(outs, balls, rng, draws)
    chances = []
    for rate in rates:
        chances.append(1 - (1 - rate) ** k)
    low, high = middle_95(chances)

    survival = []
    for ball in range(k + 1):
        alive = []
        for rate in rates:
            alive.append((1 - rate) ** ball)
        alive_low, alive_high = middle_95(alive)
        survival.append((middle(alive), alive_low, alive_high))

    # the picture's overs, played ball by ball: each replay's rate, one over
    played = []
    for rate in rates:
        played.append(_one_over(rate, k, rng))
    return OutWithin(
        chance=middle(chances),
        low=low,
        high=high,
        survival=tuple(survival),
        overs=representative(played, PICTURED),
    )


def middle(values: list[float]) -> float:
    """The middle value: half the values are below it, half above."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def representative(played: list[int], keep: int) -> tuple[int, ...]:
    """Overs that stand for all the replays: sorted by the ball he was out on (the
    overs he survived last), then every so many taken, so 41% out stays 41 of 100."""
    early_first = []
    for fell in played:
        if fell == 0:
            early_first.append((1, 0))  # survived: after every wicket
        else:
            early_first.append((0, fell))
    early_first.sort()
    step = len(early_first) / keep
    kept = []
    for i in range(keep):
        _, fell = early_first[int(i * step + step / 2)]
        kept.append(fell)
    return tuple(kept)


def _one_over(rate: float, k: int, rng: random.Random) -> int:
    # the ball he is out on, or 0 if he survives all k
    for ball in range(1, k + 1):
        if rng.random() < rate:
            return ball
    return 0
