"""Days 50-51: pulling a small record toward what is normal.

SYNTHETIC: every batter here is made up.
"""

import random

from cricketmind_v3.domain.out_within import out_within
from cricketmind_v3.domain.shrinkage import (
    JEFFREYS,
    MOST,
    Prior,
    best,
    likelihood,
    population_prior,
    shrunk_rate,
)

# 40 batters, 500 balls each: 20 careful ones out 10 times (2%), 20 risky ones out 40
# times (8%). Batters really differ here, far more than luck would make them
TWO_KINDS = [(10, 500)] * 20 + [(40, 500)] * 20


def made_up_careers(rng, batters, outs, survived, longest):
    """Batters whose true rates come from Beta(outs, survived). Each plays 1 to 40
    innings and, as in cricket, bats until he is out or the innings ends, after 1 to
    `longest` balls."""
    records = []
    for _ in range(batters):
        rate = rng.betavariate(outs, survived)
        out_count = 0
        faced = 0
        for _ in range(rng.randint(1, 40)):
            for _ in range(rng.randint(1, longest)):
                faced += 1
                if rng.random() < rate:
                    out_count += 1
                    break
        records.append((out_count, faced))
    return records


def counted(records):
    counts = {}
    for record in records:
        counts[record] = counts.get(record, 0) + 1
    return counts


def test_the_fit_is_the_most_likely_curve():
    prior = population_prior(TWO_KINDS)
    counts = counted(TWO_KINDS)
    found = likelihood(counts, prior.normal(), prior.worth())
    for normal, worth in (
        (prior.normal() + 0.002, prior.worth()),
        (prior.normal() - 0.002, prior.worth()),
        (prior.normal(), prior.worth() * 1.2),
        (prior.normal(), prior.worth() / 1.2),
    ):
        assert likelihood(counts, normal, worth) < found
    assert abs(prior.normal() - 0.05) < 0.001  # 20 at 2%, 20 at 8%
    assert 40 < prior.worth() < 80


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


def test_batting_until_out_does_not_fool_it():
    # T20-like: 5.5% normal, worth 200 balls. The balls-weighted rate is too low,
    # because the good batters last longest and face the most balls (found on Day 51)
    rng = random.Random(7)
    records = made_up_careers(rng, 2000, 11, 189, 40)
    outs = 0
    balls = 0
    for out, faced in records:
        outs += out
        balls += faced
    assert outs / balls < 0.054
    prior = population_prior(records)
    assert abs(prior.normal() - 0.055) < 0.003
    assert 150 < prior.worth() < 260


# The plan's test


def test_one_out_in_two_balls_is_not_a_50_percent_risk():
    prior = population_prior(TWO_KINDS)
    # with no view at all (Day 49), his next ball is a coin toss
    alone = out_within(1, 2, 1)
    assert 0.45 < alone.chance < 0.55
    # pulled toward normal: about 6.5%
    assert 0.055 < shrunk_rate(prior, 1, 2) < 0.075
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


def test_the_search_finds_the_top():
    def hill(x):
        return -((x - 3) ** 2)

    assert abs(best(hill, 0.0, 10.0) - 3.0) < 1e-6


# When normal cannot be told: no prior, so no answer


def test_batters_who_do_not_differ():
    # all 40 out 25 times in 500: no sign they differ, so no telling how hard to pull;
    # all the way to normal would give a range that is falsely sure
    assert population_prior([(25, 500)] * 40) is None
    counts = counted([(25, 500)] * 40)
    assert likelihood(counts, 0.05, MOST) > likelihood(counts, 0.05, 1000)


def test_too_few_batters():
    assert population_prior(TWO_KINDS[:29]) is None
    assert population_prior(TWO_KINDS[:10] + TWO_KINDS[20:40]) is not None  # 30


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
