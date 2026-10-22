"""Day 39: the bootstrap. Replays of the same balls, and the range they give."""

import random

from cricketmind_v3.domain.balls import Ball, tally
from cricketmind_v3.domain.bootstrap import (
    average_on_his_mix,
    difference,
    middle_95,
    replay,
    versus_average,
)
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.measures import strike_rate
from tests.examples import V2_COVER_DRIVE, V2_PULL, v2_runs


def ball(runs=0, wides=0, batter="V Kohli", match_type="T20", gender="male") -> Ball:
    return Ball(
        match_type=match_type,
        gender=gender,
        batter=batter,
        runs_batter=runs,
        wides=wides,
        noballs=0,
        player_out=None,
        wicket_kind=None,
    )


def exact_range(first: list[int], second: list[int]) -> tuple[float, float]:
    """The bootstrap with no luck in it: every possible replay added up, not drawn.

    For strike rates on 50 balls each, no wides: a replay's strike rate is 2 x its runs.
    """
    first_ways = _ways_to_score(first)
    second_ways = _ways_to_score(second)
    ways = {}  # strike-rate difference -> how many pairs of replays give it
    for runs_one, count_one in first_ways.items():
        for runs_two, count_two in second_ways.items():
            gap = 2 * (runs_one - runs_two)
            ways[gap] = ways.get(gap, 0) + count_one * count_two
    everything = sum(ways.values())
    low = None
    high = None
    so_far = 0
    for gap in sorted(ways):
        so_far += ways[gap]
        if low is None and so_far * 1000 >= everything * 25:
            low = float(gap)
        if high is None and so_far * 1000 >= everything * 975:
            high = float(gap)
    return low, high


def _ways_to_score(runs: list[int]) -> dict[int, int]:
    # run total -> how many of the possible replays give it
    ways = {0: 1}
    for _ in range(len(runs)):
        after = {}
        for total, count in ways.items():
            for scored in runs:
                after[total + scored] = after.get(total + scored, 0) + count
        ways = after
    return ways


# Replays


def test_a_replay_has_as_many_balls_as_before():
    counts = {ball(4): 10, ball(1): 15, ball(0): 25}
    rng = random.Random(42)
    for _ in range(100):
        again = replay(counts, rng)
        assert sum(again.values()) == 50
        for picked in again:
            assert picked in counts  # only balls that happened


def test_replays_pick_as_one_ball_at_a_time_would():
    # picking 50 balls one at a time from 5 sixes, 15 fours and 30 dots: the number of
    # fours is binomial, 50 picks at 0.3 each: on average 15, spread (variance)
    # 50 x 0.3 x 0.7 = 10.5
    counts = {ball(6): 5, ball(4): 15, ball(0): 30}
    expected = {ball(6): (5, 4.5), ball(4): (15, 10.5), ball(0): (30, 12.0)}
    rng = random.Random(1)
    seen = {ball(6): [], ball(4): [], ball(0): []}
    for _ in range(20_000):
        again = replay(counts, rng)
        for kind, picks in seen.items():
            picks.append(again.get(kind, 0))
    for kind, (mean, variance) in expected.items():
        average = sum(seen[kind]) / len(seen[kind])
        spread = 0.0
        for times in seen[kind]:
            spread += (times - average) ** 2
        spread = spread / len(seen[kind])
        assert abs(average - mean) < 0.1
        assert abs(spread - variance) < 0.05 * variance


def test_one_kind_of_ball_never_changes():
    dots = {ball(0): 30}
    assert replay(dots, random.Random(42)) == dots
    sixes = {ball(6): 30}
    same = difference(sixes, dots, strike_rate, resamples=200)
    assert same == Fact(value=600.0, n=30, interval=(600.0, 600.0))


def test_the_middle_95():
    values = []
    for value in range(1000):
        values.append(float(value))
    assert middle_95(values) == (25.0, 974.0)  # 25 dropped at each end


# The plan's test: the v2 running example (SYNTHETIC; N-001, N-002)


def test_the_v2_example():
    found = difference(tally(V2_COVER_DRIVE), tally(V2_PULL), strike_rate)
    assert found.value == 30.0  # 142.0 - 112.0
    assert found.n == 50
    low, high = found.interval
    assert low < 0 < high  # the replays flip the answer: no detectable difference
    assert (low, high) == (-16.0, 78.0)  # seed 42, 20,000 replays: pinned


def test_the_v2_example_is_within_one_run_of_the_exact_range():
    exact = exact_range(v2_runs(50, 71), v2_runs(50, 56))
    assert exact == (-16.0, 76.0)  # N-002
    for seed in (42, 1, 2, 3):
        found = difference(
            tally(V2_COVER_DRIVE), tally(V2_PULL), strike_rate, seed=seed
        )
        low, high = found.interval
        # one run in 50 balls is 2 points of strike rate
        assert abs(low - exact[0]) <= 2.0
        assert abs(high - exact[1]) <= 2.0


def test_the_same_seed_gives_the_same_range():
    first = difference(tally(V2_COVER_DRIVE), tally(V2_PULL), strike_rate, seed=7)
    again = difference(tally(V2_COVER_DRIVE), tally(V2_PULL), strike_rate, seed=7)
    assert first == again


# Differences


def test_n_is_the_thinner_side():
    many = {ball(1): 25, ball(0): 25}
    few = {ball(4): 2, ball(0): 6}
    assert difference(many, few, strike_rate, resamples=500).n == 8


def test_a_replay_with_no_ball_faced_gives_no_answer():
    # a wide and a dot: a quarter of replays are two wides, no ball faced
    thin = {ball(wides=1): 1, ball(0): 1}
    assert difference(thin, {ball(1): 10}, strike_rate, resamples=200) is None


def test_a_range_that_misses_its_estimate_gives_no_answer():
    balls = {}
    for name in "ABCDEFGHIJ":
        balls[ball(batter=name)] = 5

    def odd(counts):
        # 1 for the real balls, 0 for almost any replay
        if counts == balls:
            return Fact(value=1.0, n=50)
        return Fact(value=0.0, n=50)

    assert difference(balls, {ball(0): 50}, odd, resamples=200) is None


# A player against the average batter


def test_the_average_is_a_fixed_number():
    player = {ball(6): 2, ball(4): 3, ball(1): 5, ball(0): 10}  # 29 off 20: 145.0
    slow = {ball(1, batter="x"): 200, ball(0, batter="x"): 200}  # 50.0 on 400 balls
    quick = {ball(1, batter="x"): 400}  # 100.0 on 400 balls
    against_slow = versus_average(player, slow, strike_rate)
    against_quick = versus_average(player, quick, strike_rate)
    assert against_slow.value == 95.0
    assert against_quick.value == 45.0
    assert against_slow.n == 20
    # the same replays of the player: a quicker average moves the range, not its width
    assert against_slow.interval[0] - 50.0 == against_quick.interval[0]
    assert against_slow.interval[1] - 50.0 == against_quick.interval[1]


def test_the_average_must_rest_on_twenty_times_the_balls():
    player = {ball(4): 5, ball(0): 15}  # 20 balls
    assert versus_average(player, {ball(1): 400}, strike_rate) is not None
    assert versus_average(player, {ball(1): 399}, strike_rate) is None


# Like with like (found on Day 39: Kohli faced Bumrah only in T20s)


def t20_and_test_balls() -> dict[Ball, int]:
    # everyone else: 100.0 in T20s on 400 balls, 40.0 in Tests on 2,000 balls
    return {
        ball(1, batter="x"): 400,
        ball(1, batter="x", match_type="Test"): 800,
        ball(0, batter="x", match_type="Test"): 1200,
    }


def test_a_t20_player_is_set_against_t20_balls_only():
    player = {ball(6): 2, ball(1): 8, ball(0): 10}  # 20 off 20 in T20s: 100.0
    found = versus_average(player, t20_and_test_balls(), strike_rate, resamples=500)
    assert found.value == 0.0  # level with the T20 average, not 50 points above all


def test_the_average_is_mixed_in_his_shares():
    # half his balls in T20s, half in ODIs: half of each average
    player = {ball(1): 20, ball(1, match_type="ODI"): 20}  # 100.0
    everyone_else = {
        ball(6, batter="x"): 160,
        ball(0, batter="x"): 640,  # T20s: 120.0 on 800 balls
        ball(4, batter="x", match_type="ODI"): 160,
        ball(0, batter="x", match_type="ODI"): 640,  # ODIs: 80.0 on 800 balls
    }
    assert average_on_his_mix(player, everyone_else, strike_rate) == 100.0


def test_a_format_everyone_else_lacks_gives_no_answer():
    player = {ball(1): 20, ball(1, match_type="ODI"): 20}
    everyone_else = {ball(1, batter="x"): 800}  # no ODI balls at all
    assert average_on_his_mix(player, everyone_else, strike_rate) is None
    assert versus_average(player, everyone_else, strike_rate) is None


def test_twenty_times_the_balls_in_each_format():
    player = {ball(1): 20, ball(1, match_type="ODI"): 20}
    everyone_else = {
        ball(1, batter="x"): 800,
        ball(1, batter="x", match_type="ODI"): 399,  # under 20 x 20
    }
    assert average_on_his_mix(player, everyone_else, strike_rate) is None


def test_a_mans_balls_are_set_against_mens_balls_only():
    # Day 42: everyone else's T20 balls hold women's too (strike rate 96.9 against
    # 125.8 in the real data); a man is set against men only
    player = {ball(1): 20}  # 100.0
    everyone_else = {
        ball(1, batter="x"): 400,  # men: 100.0
        ball(1, batter="x", gender="female"): 200,
        ball(0, batter="x", gender="female"): 200,  # women: 50.0
    }
    assert average_on_his_mix(player, everyone_else, strike_rate) == 100.0
