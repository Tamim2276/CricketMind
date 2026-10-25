"""Day 50: pulling a small record toward what is normal.

SYNTHETIC: every batter here is made up.
"""

import random

from cricketmind_v3.domain.out_within import out_within
from cricketmind_v3.domain.shrinkage import (
    JEFFREYS,
    Prior,
    population_prior,
    shrunk_rate,
)

# 40 batters, 500 balls each: 20 careful ones out 10 times (2%), 20 risky ones out 40
# times (8%). Batters really differ here, far more than luck would make them
TWO_KINDS = [(10, 500)] * 20 + [(40, 500)] * 20


def test_a_population_worked_out_by_hand():
    # normal: 1,000 outs in 20,000 balls is 5%; one ball varies by 0.05 * 0.95 = 0.0475
    # seen: 40 batters * 500 balls * (3% from normal) squared = 18.0
    # luck alone: 39 * 0.0475 = 1.8525
    # room: 20,000 - 40 * 500 * 500 / 20,000 - 39 = 19,461
    # real spread: (18.0 - 1.8525) / 19,461; worth: 0.0475 / spread - 1 = 56.247
    prior = population_prior(TWO_KINDS)
    assert abs(prior.normal() - 0.05) < 1e-12
    assert abs(prior.worth() - 56.247) < 0.001
    assert abs(prior.outs - 0.05 * 56.247) < 0.001


def test_it_finds_a_prior_it_was_not_told():
    # 2,000 made-up batters whose true rates come from Beta(5, 95): 5% normal, worth
    # 100 balls; each faces from 1 to 300 balls, and luck decides his outs
    rng = random.Random(42)
    records = []
    for _ in range(2000):
        rate = rng.betavariate(5, 95)
        balls = rng.randint(1, 300)
        outs = 0
        for _ in range(balls):
            if rng.random() < rate:
                outs += 1
        records.append((outs, balls))
    prior = population_prior(records)
    assert abs(prior.normal() - 0.05) < 0.003
    assert 75 < prior.worth() < 125


# The plan's test


def test_one_out_in_two_balls_is_not_a_50_percent_risk():
    prior = population_prior(TWO_KINDS)
    # with no view at all (Day 49), his next ball is a coin toss
    alone = out_within(1, 2, 1)
    assert 0.45 < alone.chance < 0.55
    # pulled toward normal: (2.81 + 1) / (56.25 + 2), about 6.5%
    assert abs(shrunk_rate(prior, 1, 2) - 0.0654) < 0.0005
    pulled = out_within(1, 2, 1, prior)
    assert 0.04 < pulled.chance < 0.08
    assert abs(pulled.chance - shrunk_rate(prior, 1, 2)) < 0.01


def test_many_balls_barely_move():
    # Kohli's real record, T20 powerplay against pace: 84 outs in 2,855 balls
    prior = population_prior(TWO_KINDS)
    assert abs(shrunk_rate(prior, 84, 2855) - 84 / 2855) < 0.001
    # and the fewer his balls, the closer to normal
    few = shrunk_rate(prior, 4, 50)
    assert abs(few - 0.05) < abs(4 / 50 - 0.05)


def test_the_pull_narrows_a_small_records_range():
    prior = population_prior(TWO_KINDS)
    alone = out_within(4, 50, 6)
    pulled = out_within(4, 50, 6, prior)
    assert pulled.high - pulled.low < alone.high - alone.low


def test_jeffreys_is_worth_one_ball():
    assert JEFFREYS.worth() == 1.0
    assert JEFFREYS.normal() == 0.5
    assert Prior(outs=7.6, survived=131.0).normal() == 7.6 / 138.6


# When normal cannot be told: no prior, so no answer


def test_too_few_batters():
    assert population_prior(TWO_KINDS[:29]) is None
    assert population_prior(TWO_KINDS[:10] + TWO_KINDS[20:40]) is not None  # 30


def test_batters_who_do_not_differ():
    # all 40 out 25 times in 500: every difference would be luck, and there is none
    assert population_prior([(25, 500)] * 40) is None


def test_nobody_out_or_everybody_out():
    assert population_prior([(0, 50)] * 40) is None
    assert population_prior([(50, 50)] * 40) is None


def test_one_ball_each():
    # out or not out on one ball: a rate of 0 or 1 cannot tell skill from luck
    records = [(1, 1)] * 10 + [(0, 1)] * 30
    assert population_prior(records) is None


def test_batters_with_no_balls_are_left_out():
    with_empty = [*TWO_KINDS, (0, 0), (0, 0)]
    assert population_prior(with_empty) == population_prior(TWO_KINDS)
