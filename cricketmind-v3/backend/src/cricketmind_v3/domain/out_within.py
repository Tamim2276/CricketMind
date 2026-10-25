"""N9 T1: the chance of being out within his next k balls, and how sure we can be (Day 49).

Two kinds of uncertainty. Luck: even at a known rate he may or may not be out. And the
rate itself: 4 outs in 50 balls only roughly says how risky he is. The Beta curve says
how believable each rate is; every replay draws a rate from it, then plays the balls.
Since Day 50 the curve starts from what is normal (a prior), and the risk may change
with how long he has been in (stages).
"""

import random
from dataclasses import dataclass

from cricketmind_v3.domain.bootstrap import middle_95
from cricketmind_v3.domain.shrinkage import JEFFREYS, Prior

DRAWS = 20_000  # replays behind every answer (as for N-004)
SEED = 42  # the same balls always give the same answer
PICTURED = 100  # overs kept for the picture (D-054)


@dataclass(frozen=True, kw_only=True)
class OutWithin:
    """What the replays say."""

    chance: float  # the middle answer: half the replays give more, half less
    low: float  # the middle 95% of the replays
    high: float
    survival: tuple[tuple[float, float, float], ...]  # still batting after each ball
    overs: tuple[int, ...]  # 100 representative overs: the ball he was out on, or 0


@dataclass(frozen=True, kw_only=True)
class Stage:
    """A stretch of his innings with its own risk (Day 50): from this ball faced on,
    until the next stage starts."""

    first: int  # his ball faced it starts at: 1, 6, 16 or 31
    outs: int  # his record there, under the question's filters
    faced: int
    prior: Prior  # what is normal there, before his own balls


def believable_rates(
    outs: int, balls: int, rng: random.Random, draws: int, prior: Prior = JEFFREYS
) -> list[float]:
    """Out-rates per ball, each drawn from the Beta curve: the prior's pretend outs and
    balls survived, plus his own."""
    rates = []
    for _ in range(draws):
        rates.append(rng.betavariate(prior.outs + outs, prior.survived + balls - outs))
    return rates


def out_within(
    outs: int,
    balls: int,
    k: int,
    prior: Prior = JEFFREYS,
    draws: int = DRAWS,
    seed: int = SEED,
) -> OutWithin | None:
    """The chance he is out within his next k balls faced, with its range: one risk
    for his whole innings.

    At one rate p, surviving k balls has the chance (1 - p) to the power k: what
    playing those balls over and over would give. Each replay's rate comes from the
    Beta curve, so the spread of the answers is the honest range.
    """
    whole = Stage(first=1, outs=outs, faced=balls, prior=prior)
    return out_within_stages([whole], k, 0, draws, seed)


def out_within_stages(
    stages: list[Stage],
    k: int,
    faced_so_far: int,
    draws: int = DRAWS,
    seed: int = SEED,
) -> OutWithin | None:
    """The same, when the risk depends on how long he has been in (Day 50).

    His next ball is his ball number faced_so_far + 1; each of the next k balls has
    the risk of the stage it falls in. Each replay draws one rate for every stage.
    """
    if k < 1 or faced_so_far < 0 or not _fits(stages, faced_so_far):
        return None
    rng = random.Random(seed)
    rates = []  # for each stage, one believable rate per replay
    for stage in stages:
        rates.append(believable_rates(stage.outs, stage.faced, rng, draws, stage.prior))
    ahead = []  # the stage of each of his next k balls
    for ball in range(faced_so_far + 1, faced_so_far + k + 1):
        ahead.append(_stage_at(stages, ball))

    # still batting after each ball: the product of surviving every ball so far
    survival = []
    alive = []
    for upto in range(k + 1):
        alive = _still_batting(rates, ahead[:upto], draws)
        alive_low, alive_high = middle_95(alive)
        survival.append((middle(alive), alive_low, alive_high))
    chances = []
    for still in alive:  # after all k balls
        chances.append(1 - still)
    low, high = middle_95(chances)

    # the picture's overs, played ball by ball: each replay's rates, one over
    played = []
    for replay in range(draws):
        ball_rates = []
        for stage in ahead:
            ball_rates.append(rates[stage][replay])
        played.append(_one_over(ball_rates, rng))
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


def _fits(stages: list[Stage], faced_so_far: int) -> bool:
    # stages in order, the first covering his next ball, each record possible, and
    # at least one ball of his own: no balls, no answer about him
    if not stages or stages[0].first > faced_so_far + 1:
        return False
    balls = 0
    before = 0
    for stage in stages:
        if stage.first <= before or not 0 <= stage.outs <= stage.faced:
            return False
        before = stage.first
        balls += stage.faced
    return balls >= 1


def _stage_at(stages: list[Stage], ball: int) -> int:
    # the last stage that has started by this ball
    found = 0
    for number, stage in enumerate(stages):
        if ball >= stage.first:
            found = number
    return found


def _still_batting(
    rates: list[list[float]], balls: list[int], draws: int
) -> list[float]:
    # for each replay, the chance of surviving these balls: (1 - rate) once per ball
    # in each stage
    counts = []
    for _ in rates:
        counts.append(0)
    for stage in balls:
        counts[stage] += 1
    alive = []
    for replay in range(draws):
        still = 1.0
        for stage, count in enumerate(counts):
            if count:
                still *= (1 - rates[stage][replay]) ** count
        alive.append(still)
    return alive


def _one_over(ball_rates: list[float], rng: random.Random) -> int:
    # the ball he is out on, or 0 if he survives them all
    for ball, rate in enumerate(ball_rates, start=1):
        if rng.random() < rate:
            return ball
    return 0
