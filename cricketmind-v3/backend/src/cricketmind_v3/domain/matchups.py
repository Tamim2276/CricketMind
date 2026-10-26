"""N9 T2: one batter against two bowlers, and the difference (Days 51-52).

How risky is Kohli against Bumrah? Three records say: Kohli against everyone, Bumrah
against batters who bat where Kohli bats, and their meetings. The first two give what to
expect, by the odds rule: Kohli's edge on normal times Bumrah's. The meetings nudge that
only as far as meetings have been shown to matter, which is hardly at all (D-061).

The question is a what-if: had this bowler bowled instead of that one. Each replay draws
one Kohli and one of each bowler, so the difference is the bowler alone, not luck.
"""

import math
import random
from dataclasses import dataclass

from cricketmind_v3.domain.bootstrap import middle_95
from cricketmind_v3.domain.out_within import (
    DRAWS,
    PICTURED,
    SEED,
    OutWithin,
    believable_rates,
    middle,
    survival_and_chances,
)
from cricketmind_v3.domain.shrinkage import (
    EDGE,
    MOST,
    best,
    log_beta,
    population_prior,
    shrunk_rate,
)


@dataclass(frozen=True, kw_only=True)
class Pair:
    """One batter against one bowler, under a question's filters."""

    batter: str  # register ids
    bowler: str
    order: str  # "top", "middle" or "tail": where the batter batted
    outs: int
    faced: int


@dataclass(frozen=True, kw_only=True)
class Matchup:
    """What the replays say about the what-if."""

    difference: float  # the middle of (chance against this bowler - against that one)
    low: float  # the middle 95% of the differences
    high: float
    this: OutWithin  # against the bowler of the what-if
    that: OutWithin  # against the bowler he would replace
    n: int  # the thinnest record it rests on, in balls (C3)
    meetings_worth: (
        float  # how many balls the expected rate counts as, against meetings
    )


def expected(batter_rate: float, bowler_rate: float, normal: float) -> float:
    """What to expect when they meet: the odds rule (Bill James's log5). His odds of
    being out times the bowler's, over the normal odds."""
    odds = _odds(batter_rate) * _odds(bowler_rate) / _odds(normal)
    return odds / (1 + odds)


def meetings_worth(
    pairs: list[Pair],
    batter_rates: dict[str, float],
    bowler_rates: dict[str, float],
    normal: float,
) -> float:
    """How much meetings matter beyond the two records: the worth of the Beta curve
    around each pair's expected rate under which the meetings were the most likely.
    MOST means they add nothing."""
    counts = {}  # (outs, faced, expected rate) -> how many pairs
    for pair in pairs:
        rate = expected(batter_rates[pair.batter], bowler_rates[pair.bowler], normal)
        # to a hundredth of a percent, so alike pairs are counted together
        rate = min(max(round(rate, 4), 0.0001), 0.9999)
        key = (pair.outs, pair.faced, rate)
        counts[key] = counts.get(key, 0) + 1

    def at_worth(log_worth: float) -> float:
        worth = math.exp(log_worth)
        total = 0.0
        for (outs, faced, rate), times in counts.items():
            ahead = rate * worth
            behind = (1 - rate) * worth
            after = log_beta(ahead + outs, behind + faced - outs)
            total += times * (after - log_beta(ahead, behind))
        return total

    return math.exp(best(at_worth, 0.0, math.log(MOST)))


def matchup(
    pairs: list[Pair],
    batter: str,
    this_bowler: str,
    that_bowler: str,
    k: int,
    draws: int = DRAWS,
    seed: int = SEED,
) -> Matchup | None:
    """The chance he is out within his next k balls against this bowler, minus against
    that one, with its range; None when it cannot be told."""
    if k < 1 or this_bowler == that_bowler:
        return None
    batters = _totals(pairs, by_bowler=False)
    bowlers = _totals(pairs, by_bowler=True)
    for who, totals in (
        (batter, batters),
        (this_bowler, bowlers),
        (that_bowler, bowlers),
    ):
        if totals.get(who, (0, 0))[1] < 1:
            return None  # no balls of his: no answer about him
    outs = 0
    balls = 0
    for pair in pairs:
        outs += pair.outs
        balls += pair.faced
    normal = (
        outs / balls
    )  # a ball's rate, over every ball: where the odds rule is level

    # what is normal for a batter like him, and for a bowler, from the others
    batter_prior = population_prior(_others(batters, (batter,)))
    bowler_prior = population_prior(_others(bowlers, (this_bowler, that_bowler)))
    if batter_prior is None or bowler_prior is None:
        return None
    batter_rates = {}
    for who, (out, faced) in batters.items():
        batter_rates[who] = shrunk_rate(batter_prior, out, faced)
    bowler_rates = {}
    for who, (out, faced) in bowlers.items():
        bowler_rates[who] = shrunk_rate(bowler_prior, out, faced)
    worth = meetings_worth(pairs, batter_rates, bowler_rates, normal)

    rng = random.Random(seed)
    his_outs, his_faced = batters[batter]
    his = believable_rates(his_outs, his_faced, rng, draws, batter_prior)
    sides = []
    for bowler in (this_bowler, that_bowler):
        bowler_outs, bowler_faced = bowlers[bowler]
        theirs = believable_rates(bowler_outs, bowler_faced, rng, draws, bowler_prior)
        met_outs, met_faced = _met(pairs, batter, bowler)
        rates = []
        for replay in range(draws):
            rate = expected(his[replay], theirs[replay], normal)
            # their meetings nudge it, as much as meetings are worth
            rates.append(
                rng.betavariate(
                    rate * worth + met_outs, (1 - rate) * worth + met_faced - met_outs
                )
            )
        sides.append(rates)

    ahead = []  # one stage: every ball has the replay's rate
    for _ in range(k):
        ahead.append(0)
    summaries = []
    for rates in sides:
        survival, chances = survival_and_chances([rates], ahead, draws)
        summaries.append((survival, chances))
    differences = []
    for replay in range(draws):
        differences.append(summaries[0][1][replay] - summaries[1][1][replay])
    low, high = middle_95(differences)

    # the pictures: both overs played with the same balls, so only the bowler changes
    this_played = []
    that_played = []
    for replay in range(draws):
        luck = []
        for _ in range(k):
            luck.append(rng.random())
        this_played.append(_fell(sides[0][replay], luck))
        that_played.append(_fell(sides[1][replay], luck))
    this_overs, that_overs = representative_pairs(this_played, that_played, PICTURED)

    found = []
    for (survival, chances), overs in zip(
        summaries, (this_overs, that_overs), strict=True
    ):
        chance_low, chance_high = middle_95(chances)
        found.append(
            OutWithin(
                chance=middle(chances),
                low=chance_low,
                high=chance_high,
                survival=survival,
                overs=overs,
            )
        )
    return Matchup(
        difference=middle(differences),
        low=low,
        high=high,
        this=found[0],
        that=found[1],
        n=min(his_faced, bowlers[this_bowler][1], bowlers[that_bowler][1]),
        meetings_worth=worth,
    )


def representative_pairs(
    this_played: list[int], that_played: list[int], keep: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Paired overs that stand for all the replays: sorted by the ball he was out on
    against this bowler, then against that one (survived last), and every so many taken.
    Each pair stays a pair: the same replay, the same luck."""
    order = []
    for replay in range(len(this_played)):
        order.append((_later(this_played[replay]), _later(that_played[replay]), replay))
    order.sort()
    step = len(order) / keep
    this_kept = []
    that_kept = []
    for i in range(keep):
        _, _, replay = order[int(i * step + step / 2)]
        this_kept.append(this_played[replay])
        that_kept.append(that_played[replay])
    return tuple(this_kept), tuple(that_kept)


def in_order(pairs: list[Pair], order: str) -> list[Pair]:
    """Only the balls faced by batters batting in this part of the order."""
    found = []
    for pair in pairs:
        if pair.order == order:
            found.append(pair)
    return found


def _odds(rate: float) -> float:
    rate = min(max(rate, EDGE), 1 - EDGE)
    return rate / (1 - rate)


def _totals(pairs: list[Pair], by_bowler: bool) -> dict[str, tuple[int, int]]:
    # each batter's (or bowler's) outs and balls, against everyone
    totals = {}
    for pair in pairs:
        if by_bowler:
            who = pair.bowler
        else:
            who = pair.batter
        outs, faced = totals.get(who, (0, 0))
        totals[who] = (outs + pair.outs, faced + pair.faced)
    return totals


def _others(
    totals: dict[str, tuple[int, int]], leave_out: tuple[str, ...]
) -> list[tuple[int, int]]:
    found = []
    for who, record in totals.items():
        if who not in leave_out:
            found.append(record)
    return found


def _met(pairs: list[Pair], batter: str, bowler: str) -> tuple[int, int]:
    # their meetings: his outs and balls against this bowler
    outs = 0
    faced = 0
    for pair in pairs:
        if pair.batter == batter and pair.bowler == bowler:
            outs += pair.outs
            faced += pair.faced
    return outs, faced


def _fell(rate: float, luck: list[float]) -> int:
    # the ball he is out on, or 0: out when the ball's luck falls under the rate
    for ball, chance in enumerate(luck, start=1):
        if chance < rate:
            return ball
    return 0


def _later(fell: int) -> tuple[int, int]:
    # survived sorts after every wicket
    if fell == 0:
        return (1, 0)
    return (0, fell)
