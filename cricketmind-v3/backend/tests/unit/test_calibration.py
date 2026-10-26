"""Day 53: the test year's moments and windows, and each model's chance.

SYNTHETIC: every batter and bowler here is made up.
"""

import random

from cricketmind_v3.domain.calibration import (
    Fitted,
    Moment,
    ball_chance,
    displayed_middle,
    fit,
    window_chance,
    windows,
)
from cricketmind_v3.domain.matchups import expected
from cricketmind_v3.domain.shrinkage import Prior, population_prior, shrunk_rate
from cricketmind_v3.domain.stages import Record, by_batter, in_stage
from tests.unit.test_matchups import field


def moment(batter, before, out=False, faced=True, match="m1", bowler="B") -> Moment:
    return Moment(
        match=match,
        innings=1,
        batter=batter,
        bowler=bowler,
        faced_before=before,
        order="top",
        out=out,
        faced=faced,
    )


# Windows


def test_windows_by_hand():
    # X faces 8 balls and is out on the 8th; Y faces 3 and the innings ends; Z faces 2,
    # then is stumped off a wide
    moments = []
    for before in range(8):
        moments.append(moment("X", before, out=before == 7))
    for before in range(3):
        moments.append(moment("Y", before))
    moments.append(moment("Z", 0))
    moments.append(moment("Z", 1))
    moments.append(moment("Z", 2, out=True, faced=False))
    found = {}
    for window in windows(moments):
        found[(window.start.batter, window.start.faced_before)] = (
            window.balls,
            window.out,
        )
    assert found[("X", 0)] == (6, False)  # balls 1-6: not out
    assert found[("X", 2)] == (6, True)  # balls 3-8: out on the 8th
    assert found[("X", 3)] == (6, True)  # his wicket does not shorten it
    assert found[("X", 7)] == (6, True)
    assert found[("Y", 0)] == (3, False)  # cut short by the innings ending
    assert found[("Y", 2)] == (1, False)
    assert found[("Z", 0)] == (6, True)  # the wide he was out on counts
    assert len(found) == 8 + 3 + 3  # a window from every moment


# The models, fitted on made-up records


def made_up_records() -> list[Record]:
    """The batters of tests/unit/test_matchups' field, by stage: bat0 to bat39 careful
    (4%) or risky (6%), and K at exactly 5%, with a quarter of their balls in stage 1."""
    records = []
    for i in range(40):
        risky = i % 2
        records.append(
            Record(
                batter=f"bat{i}", stage=1, order="top", outs=40 + 20 * risky, faced=1000
            )
        )
        records.append(
            Record(
                batter=f"bat{i}",
                stage=2,
                order="top",
                outs=120 + 60 * risky,
                faced=3000,
            )
        )
    records.append(Record(batter="K", stage=1, order="top", outs=1250, faced=25_000))
    records.append(Record(batter="K", stage=2, order="top", outs=3750, faced=75_000))
    return records


RECORDS = made_up_records()
FITTED = fit(RECORDS, field())


def test_d_is_his_record_alone():
    found = ball_chance(FITTED, "D", moment("K", 0))
    assert found == (0.5 + 5000) / (1 + 100_000)


def test_a_and_b_pull_toward_normal():
    prior = population_prior(by_batter(RECORDS))
    assert ball_chance(FITTED, "A", moment("K", 0)) == shrunk_rate(prior, 5000, 100_000)
    # everyone here bats at the top, so B's group is everyone
    assert ball_chance(FITTED, "B", moment("K", 0)) == shrunk_rate(prior, 5000, 100_000)


def test_c_uses_his_stage():
    early = population_prior(in_stage(RECORDS, 1))
    later = population_prior(in_stage(RECORDS, 2))
    # his 1st ball is in stage 1, his 7th in stage 2
    assert ball_chance(FITTED, "C", moment("K", 0)) == shrunk_rate(early, 1250, 25_000)
    assert ball_chance(FITTED, "C", moment("K", 6)) == shrunk_rate(later, 3750, 75_000)


def test_e_is_the_odds_rule_on_his_group():
    found = ball_chance(FITTED, "E", moment("K", 0, bowler="B"))
    rate = expected(
        FITTED.batter_rates["top"]["K"],
        FITTED.bowler_rates["top"]["B"],
        FITTED.normal["top"],
    )
    assert abs(found - rate) < 0.001  # no meetings here, and they are worth much
    assert found > ball_chance(FITTED, "E", moment("K", 0, bowler="S"))  # B is sharper


def test_a_window_is_exact():
    # his record alone, 4 outs in 50 balls (the v2 example): surviving 6 balls, averaged
    # over Beta(4.5, 46.5), is the product of (46.5 + j) / (51 + j)
    small = fit(
        [*RECORDS, Record(batter="v2", stage=1, order="top", outs=4, faced=50)], field()
    )
    survive = 1.0
    for j in range(6):
        survive *= (46.5 + j) / (51 + j)
    window = windows([moment("v2", before) for before in range(6)])[0]
    assert window.balls == 6
    assert abs(window_chance(small, "D", window) - (1 - survive)) < 1e-12
    # and the same, the slow way: 200,000 rates drawn from the curve
    rng = random.Random(42)
    total = 0.0
    for _ in range(200_000):
        rate = rng.betavariate(4.5, 46.5)
        total += 1 - (1 - rate) ** 6
    assert abs(total / 200_000 - (1 - survive)) < 0.002


def test_a_window_across_stages():
    # from his 4th ball: balls 4-5 in stage 1, 6-9 in stage 2
    moments = []
    for before in range(3, 12):
        moments.append(moment("K", before))
    window = windows(moments)[0]
    early = FITTED.stage[("top", 1)]
    later = FITTED.stage[("top", 2)]
    still = 1.0
    for outs, survived, count in (
        (early.outs + 1250, early.survived + 25_000 - 1250, 2),
        (later.outs + 3750, later.survived + 75_000 - 3750, 4),
    ):
        for j in range(count):
            still *= (survived + j) / (outs + survived + j)
    assert abs(window_chance(FITTED, "C", window) - (1 - still)) < 1e-12
    # B has one curve for all six balls, stages or not
    prior = FITTED.order["top"]
    a = prior.outs + 5000
    b = prior.survived + 95_000
    one = 1.0
    for j in range(6):
        one *= (b + j) / (a + b + j)
    assert abs(window_chance(FITTED, "B", window) - (1 - one)) < 1e-12


def test_the_displayed_middle_is_below_the_mean_for_a_small_record():
    small = fit(
        [*RECORDS, Record(batter="new", stage=1, order="top", outs=1, faced=3)], field()
    )
    window = windows([moment("new", before) for before in range(6)])[0]
    assert displayed_middle(small, window) < window_chance(small, "B", window)


def test_no_chance_without_balls_of_his_own():
    assert ball_chance(FITTED, "B", moment("nobody", 0)) is None
    only_a_wide = fit(
        [*RECORDS, Record(batter="w", stage=1, order="top", outs=1, faced=0)], field()
    )
    assert ball_chance(only_a_wide, "D", moment("w", 0)) is None
    assert ball_chance(FITTED, "E", moment("K", 0, bowler="nobody")) is None


def by_hand() -> Fitted:
    # a Fitted written by hand: v2's 4 outs in 50 balls under a Jeffreys-like prior, and
    # meetings worth only 10 balls, with 3 outs in 20 against bowler M
    return Fitted(
        totals={"v2": (4, 50)},
        by_stage={},
        usual={"v2": "top"},
        every=Prior(outs=0.5, survived=0.5),
        order={"top": Prior(outs=0.5, survived=0.5)},
        stage={},
        batter_rates={"top": {"v2": 0.05}},
        bowler_rates={"top": {"M": 0.05}},
        normal={"top": 0.05},
        worth={"top": 10.0},
        met={("top", "v2", "M"): (3, 20)},
    )


def test_e_hears_the_meetings_as_much_as_they_are_worth():
    # the odds rule: a normal batter against a normal bowler is 5%; then
    # (0.05 x 10 + 3) / (10 + 20) = 3.5 / 30
    found = ball_chance(by_hand(), "E", moment("v2", 0, bowler="M"))
    assert abs(found - 3.5 / 30) < 1e-12


def test_the_displayed_middle_by_hand():
    # Beta(4.5, 46.5): Kerman's median (4.5 - 1/3) / (51 - 2/3) = 0.08278; over 6 balls,
    # 1 - (1 - 0.08278)^6 = 0.4045, beside the exact middle 0.4049 (Day 49, N-004)
    window = windows([moment("v2", before) for before in range(6)])[0]
    found = displayed_middle(by_hand(), window)
    median = (4.5 - 1 / 3) / (51 - 2 / 3)
    assert abs(found - (1 - (1 - median) ** 6)) < 1e-12
    assert abs(found - 0.4049) < 0.002
