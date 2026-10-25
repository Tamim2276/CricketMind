"""Day 49: the chance of being out within k balls, with its honest range.

SYNTHETIC: the v2 running example, the pull's 4 outs in 50 balls (N-003, N-004).
"""

import math

from cricketmind_v3.domain.out_within import (
    Stage,
    middle,
    out_within,
    out_within_stages,
    representative,
)
from cricketmind_v3.domain.shrinkage import JEFFREYS, Prior


def exact_chances(
    outs: int, balls: int, k: int, prior: Prior = JEFFREYS
) -> tuple[float, float, float]:
    """The answer with no luck in it: the Beta curve's own 2.5%, 50% and 97.5% points,
    found by adding up the curve, then turned into chances of being out within k."""
    a = prior.outs + outs
    b = prior.survived + balls - outs
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    steps = 40_000
    area = [0.0]
    before = 0.0
    for i in range(1, steps):
        x = i / steps
        height = math.exp((a - 1) * math.log(x) + (b - 1) * math.log(1 - x) - log_beta)
        area.append(area[-1] + (before + height) / 2 / steps)
        before = height
    found = []
    for share in (0.025, 0.5, 0.975):
        for i, so_far in enumerate(area):
            if so_far >= share * area[-1]:
                rate = i / steps
                found.append(1 - (1 - rate) ** k)
                break
    low, mid, high = found
    return mid, low, high


# The plan's test


def test_the_v2_example():
    found = out_within(4, 50, 6)
    assert abs(found.chance - 0.405) < 0.005  # N-004
    assert abs(found.low - 0.155) < 0.01
    assert abs(found.high - 0.697) < 0.01
    # seed 42, 20,000 replays: pinned
    assert (round(found.chance, 4), round(found.low, 4), round(found.high, 4)) == (
        0.4056,
        0.1568,
        0.6917,
    )


def test_the_replays_land_near_the_exact_answer():
    mid, low, high = exact_chances(4, 50, 6)
    assert abs(mid - 0.4049) < 0.001  # added up, not drawn
    found = out_within(4, 50, 6)
    assert abs(found.chance - mid) < 0.01
    assert abs(found.low - low) < 0.01
    assert abs(found.high - high) < 0.01


def test_the_honest_range_is_far_wider_than_v2s():
    # v2 played 20,000 overs at 8% a ball: 0.394, range 0.387 to 0.400 (N-003). That
    # range is luck only; it forgets that 50 balls say little about the rate
    point = 1 - (1 - 4 / 50) ** 6
    assert round(point, 3) == 0.394
    found = out_within(4, 50, 6)
    assert found.high - found.low > 20 * (0.400 - 0.387)


def test_more_balls_give_a_narrower_range():
    thin = out_within(4, 50, 6)
    thick = out_within(400, 5000, 6)  # the same 8%, a hundred times the evidence
    assert thick.high - thick.low < 0.07
    assert thin.high - thin.low > 0.5


def test_the_same_seed_gives_the_same_answer():
    assert out_within(4, 50, 6) == out_within(4, 50, 6)
    other = out_within(4, 50, 6, seed=7)
    mid, low, high = exact_chances(4, 50, 6)
    # another seed lands near the same answer
    assert abs(other.chance - mid) < 0.01
    assert abs(other.low - low) < 0.01
    assert abs(other.high - high) < 0.01


# The summary kept for the pictures (D-054)


def test_still_batting_ball_by_ball():
    found = out_within(4, 50, 6)
    assert found.survival[0] == (1.0, 1.0, 1.0)  # now, he is batting
    for ball in range(1, 7):
        assert found.survival[ball][0] < found.survival[ball - 1][0]
        low, high = found.survival[ball][1], found.survival[ball][2]
        assert low <= found.survival[ball][0] <= high
    assert abs(found.survival[6][0] - (1 - found.chance)) < 0.001


def test_the_pictured_overs_stand_for_all_the_replays():
    found = out_within(4, 50, 6)
    assert len(found.overs) == 100
    # the expected share out: 1 - E[(1 - p)^6] for p from Beta(4.5, 46.5)
    survive = 1.0
    for i in range(6):
        survive *= (46.5 + i) / (51 + i)
    out_count = 0
    for fell in found.overs:
        assert 0 <= fell <= 6
        if fell:
            out_count += 1
    assert abs(out_count - 100 * (1 - survive)) <= 3
    assert out_count == 41  # seed 42: pinned


def test_representative_overs_keep_the_order_of_the_wickets():
    # sorted, early wickets first: 1, 1, 2, 3, 6, then the three survivors; every
    # second one is kept, so 3 survivors in 8 stay 2 in 4
    played = [0, 3, 1, 0, 6, 2, 0, 1]
    assert representative(played, 4) == (1, 3, 0, 0)
    assert middle([3.0, 1.0, 2.0]) == 2.0


# No answer


def test_no_answer_without_a_question_or_balls():
    assert out_within(0, 0, 6) is None  # no balls faced
    assert out_within(4, 50, 0) is None  # within no balls
    assert out_within(60, 50, 6) is None  # more outs than balls


# How long he has been in (Day 50, on trial)

# SYNTHETIC: so many balls that each stage's rate is all but known: 10% a ball in his
# first five, 2% after
NEW = Stage(first=1, outs=10_000, faced=100_000, prior=JEFFREYS)
SET = Stage(first=6, outs=2_000, faced=100_000, prior=JEFFREYS)


def test_one_stage_is_the_day_49_answer():
    whole = Stage(first=1, outs=4, faced=50, prior=JEFFREYS)
    assert out_within_stages([whole], 6, 0) == out_within(4, 50, 6)


def test_each_ball_has_the_risk_of_its_stage():
    # he has faced 3: balls 4 and 5 are in his first stage, 6 and 7 in the next
    found = out_within_stages([NEW, SET], 4, 3)
    still = 0.9 * 0.9 * 0.98 * 0.98  # 0.7779: by hand
    assert abs(found.chance - (1 - still)) < 0.002
    assert abs(found.survival[2][0] - 0.81) < 0.002  # after his two risky balls
    assert abs(found.survival[4][0] - still) < 0.002


def test_once_he_is_in_only_the_later_risk_counts():
    found = out_within_stages([NEW, SET], 4, 5)  # balls 6 to 9: all in the next stage
    assert abs(found.chance - (1 - 0.98**4)) < 0.002


def test_the_pictured_overs_show_the_early_danger():
    # out on ball 1 or 2 (10% each) far more often than on ball 3 or 4 (2% each)
    found = out_within_stages([NEW, SET], 4, 3)
    early = 0
    late = 0
    for fell in found.overs:
        if fell in (1, 2):
            early += 1
        if fell in (3, 4):
            late += 1
    assert early > 3 * late


def test_stages_that_do_not_fit_give_no_answer():
    assert out_within_stages([SET, NEW], 4, 3) is None  # out of order
    assert out_within_stages([SET], 4, 3) is None  # nothing for balls 4 and 5
    assert out_within_stages([NEW, SET], 4, -1) is None  # faced -1 balls
    assert out_within_stages([NEW, SET], 0, 3) is None  # within no balls
    nothing = Stage(first=1, outs=0, faced=0, prior=JEFFREYS)
    assert out_within_stages([nothing], 6, 0) is None  # no balls of his own
    impossible = Stage(first=1, outs=3, faced=2, prior=JEFFREYS)
    assert out_within_stages([impossible], 6, 0) is None
