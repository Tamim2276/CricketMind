"""Is a difference real, or luck? Replay the same balls many times (the bootstrap, Day 39).

If the replays sometimes flip the answer (the range holds 0), C4 makes the words say
"no detectable difference".
"""

import random
from collections.abc import Callable

from cricketmind_v3.domain.balls import Ball, by_format
from cricketmind_v3.domain.facts import Fact

RESAMPLES = 20_000  # replays behind every range (as for N-002)
SEED = 42  # the same balls always give the same range
FIXED_ENOUGH = 20  # the average batter is taken as fixed only on 20 times the balls

# a measure turns a tally of balls into a Fact: measures.strike_rate, for one
Measure = Callable[[dict[Ball, int]], Fact | None]


def replay(counts: dict[Ball, int], rng: random.Random) -> dict[Ball, int]:
    """As many balls as before, each picked at random from these; one can come twice.

    Picking 40,000 balls one at a time, 20,000 times, takes minutes. Deciding how many
    of each kind come up gives the same replays in a second: the first kind's number
    is a binomial draw, the next kind shares what is left, and so on.
    """
    to_pick = sum(counts.values())  # balls still to pick
    undecided = to_pick  # balls of the kinds not decided yet
    picked = {}
    for ball, count in counts.items():
        if to_pick == 0:
            break
        if count == undecided:
            times = to_pick  # the last kind takes the rest
        else:
            times = rng.binomialvariate(to_pick, count / undecided)
        if times > 0:
            picked[ball] = times
        to_pick -= times
        undecided -= count
    return picked


def middle_95(values: list[float]) -> tuple[float, float]:
    """The range of the middle 95%: the lowest 2.5% and the highest 2.5% are dropped."""
    ordered = sorted(values)
    cut = int(len(ordered) * 0.025)
    return ordered[cut], ordered[len(ordered) - 1 - cut]


def difference(
    first: dict[Ball, int],
    second: dict[Ball, int],
    measure: Measure,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> Fact | None:
    """The first's measure minus the second's, with its 95% range.

    n is the smaller of the two samples, so C3 protects the thinner side (D-030).
    """
    one = measure(first)
    other = measure(second)
    if one is None or other is None:
        return None
    rng = random.Random(seed)
    differences = []
    for _ in range(resamples):
        one_again = measure(replay(first, rng))
        other_again = measure(replay(second, rng))
        if one_again is None or other_again is None:
            return None  # a replay with no ball faced: too few balls to say anything
        differences.append(one_again.value - other_again.value)
    return _with_range(one.value - other.value, min(one.n, other.n), differences)


def versus_average(
    player: dict[Ball, int],
    everyone_else: dict[Ball, int],
    measure: Measure,
    resamples: int = RESAMPLES,
    seed: int = SEED,
) -> Fact | None:
    """The player's measure minus everyone else's, with its 95% range.

    Like with like: the average is everyone else's measure in each format, mixed in the
    player's own shares. Kohli faced Bumrah only in T20s, so he is set against other
    batters' T20 balls, not their Test balls (found on Day 39).

    Only the player's balls are replayed: the average is taken as a fixed number, which
    is fair only when it rests on many times more balls (FIXED_ENOUGH).
    """
    mine = measure(player)
    if mine is None:
        return None
    average = average_on_his_mix(player, everyone_else, measure)
    if average is None:
        return None
    rng = random.Random(seed)
    differences = []
    for _ in range(resamples):
        again = measure(replay(player, rng))
        if again is None:
            return None
        differences.append(again.value - average)
    # n: the player's side is the thinner one
    return _with_range(mine.value - average, mine.n, differences)


def average_on_his_mix(
    player: dict[Ball, int], everyone_else: dict[Ball, int], measure: Measure
) -> float | None:
    """Everyone else's measure in each of the player's formats, in his shares.

    None when a format of his has too few balls of everyone else's to count as fixed.
    """
    mine = measure(player)
    theirs_by_format = by_format(everyone_else)
    average = 0.0
    for match_type, his in by_format(player).items():
        his_part = measure(his)
        if his_part is None:
            continue  # only wides in this format: it carries no share
        theirs = measure(theirs_by_format.get(match_type, {}))
        if theirs is None or theirs.n < FIXED_ENOUGH * his_part.n:
            return None  # the average is not much surer than the player here
        share = his_part.n / mine.n  # 1.0 when he played one format only
        average += share * theirs.value
    return average


def _with_range(value: float, n: int, differences: list[float]) -> Fact | None:
    low, high = middle_95(differences)
    if not low <= value <= high:
        return None  # a range that misses its own estimate is too odd to report
    return Fact(value=value, n=n, interval=(low, high))
