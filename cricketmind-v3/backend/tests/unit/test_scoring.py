"""Day 53: scoring chances against what happened. Every number worked out by hand."""

import math

from cricketmind_v3.domain.scoring import (
    Bin,
    bin_of,
    brier,
    ece,
    keeps,
    log_loss,
    paired_bootstrap,
    per_match,
    reliability,
    scores_from,
)


def test_the_brier_score_by_hand():
    # (0.2 - 0)^2 = 0.04 and (0.8 - 1)^2 = 0.04: the mean is 0.04
    assert abs(brier([0.2, 0.8], [0, 1]) - 0.04) < 1e-12
    # always 50%: 0.25 whatever happens
    assert brier([0.5, 0.5, 0.5], [0, 1, 1]) == 0.25
    # sure and right is perfect, sure and wrong is the worst
    assert brier([1.0, 0.0], [1, 0]) == 0.0
    assert brier([1.0], [0]) == 1.0


def test_the_log_loss_by_hand():
    found = log_loss([0.8, 0.25], [1, 0])
    assert abs(found - (-math.log(0.8) - math.log(0.75)) / 2) < 1e-12
    assert log_loss([0.0], [1]) < 30  # a sure miss is costly, but not infinite


def test_bins_of_equal_size():
    chances = [0.9, 0.1, 0.5, 0.3, 0.7, 0.2]
    # sorted: 0.1, 0.2, 0.3 in the first bin, 0.5, 0.7, 0.9 in the second
    assert bin_of(chances, 2) == [1, 0, 1, 0, 1, 0]


def test_a_reliability_plot_by_hand():
    chances = [0.1, 0.1, 0.1, 0.1, 0.6, 0.6, 0.6, 0.6, 0.6]
    outcomes = [0, 0, 0, 1, 1, 1, 1, 0, 0]
    found = reliability(chances, outcomes, 2)
    # 9 cut in 2: the first bin takes 5, the four 0.1s and the first 0.6 (out), so
    # 2 out of 5 against a mean chance of 0.2; the second, four 0.6s, 2 out of 4
    assert found[0].count == 5
    assert abs(found[0].chance - 0.2) < 1e-12
    assert found[0].observed == 0.4
    assert found[1] == Bin(count=4, chance=0.6, observed=0.5)
    # the gap: (5 x 0.2 + 4 x 0.1) / 9
    assert abs(ece(found) - (5 * 0.2 + 4 * 0.1) / 9) < 1e-12


def test_a_forecast_that_is_right_on_average_has_no_gap():
    chances = [0.25] * 8
    outcomes = [1, 0, 0, 0, 1, 0, 0, 0]
    assert ece(reliability(chances, outcomes, 1)) == 0.0


def test_scores_come_back_from_match_rows():
    matches = ["m1", "m1", "m2"]
    chances = [0.2, 0.8, 0.5]
    outcomes = [0, 1, 1]
    bins = bin_of(chances, 2)
    rows = per_match(matches, chances, outcomes, bins, 2)
    totals = [0.0] * 6
    for row in rows.values():
        for i in range(6):
            totals[i] += row[i]
    found_brier, found_gap = scores_from(totals)
    assert abs(found_brier - brier(chances, outcomes)) < 1e-12
    assert abs(found_gap - ece(reliability(chances, outcomes, 2))) < 1e-12


def test_the_bootstrap_resamples_whole_matches_for_every_model_alike():
    matches = ["m1", "m1", "m2", "m2", "m3"]
    outcomes = [0, 1, 0, 0, 1]
    good = [0.1, 0.9, 0.1, 0.1, 0.9]
    poor = [0.5, 0.5, 0.5, 0.5, 0.5]
    tables = {
        "good": per_match(matches, good, outcomes, bin_of(good, 2), 2),
        "poor": per_match(matches, poor, outcomes, bin_of(poor, 2), 2),
    }
    found = paired_bootstrap(tables, resamples=200, seed=42)
    assert len(found["good"]) == 200
    for (good_brier, _), (poor_brier, _) in zip(
        found["good"], found["poor"], strict=True
    ):
        assert good_brier < poor_brier  # paired: the same matches, so always better
        assert poor_brier == 0.25
    assert paired_bootstrap(tables, resamples=20) == paired_bootstrap(
        tables, resamples=20
    )


def test_models_must_be_scored_on_the_same_matches():
    one = per_match(["m1"], [0.5], [1], [0], 1)
    other = per_match(["m2"], [0.5], [1], [0], 1)
    try:
        paired_bootstrap({"a": one, "b": other}, resamples=1)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_the_rule_of_the_trials():
    # better by more than luck, and calibration not clearly worse: kept
    assert keeps((-0.002, -0.001), (-0.01, 0.003))
    # better, but the range reaches zero, even by a hair: not kept
    assert not keeps((-0.002, 0.0), (-0.01, 0.003))
    assert not keeps((-0.002, 0.000001), (-0.01, 0.003))
    # better, but clearly worse calibrated: not kept
    assert not keeps((-0.002, -0.001), (0.001, 0.004))
    # worse: not kept
    assert not keeps((0.001, 0.002), (-0.01, 0.0))


def test_two_alike_models_differ_by_nothing_in_every_resample():
    # paired: the same matches for both, so the same model scores the same every time
    matches = ["m1", "m2", "m2", "m3"]
    outcomes = [1, 0, 1, 0]
    chances = [0.7, 0.2, 0.6, 0.1]
    table = per_match(matches, chances, outcomes, bin_of(chances, 2), 2)
    found = paired_bootstrap({"a": table, "b": dict(table)}, resamples=50)
    assert found["a"] == found["b"]
    assert len(set(found["a"])) > 1  # the resamples do differ from each other


def test_a_bin_that_said_too_much_still_counts_as_a_gap():
    # 0.9 twice and never out: the gap is 0.9, not -0.9, in a resample as on the whole
    chances = [0.9, 0.9, 0.1, 0.1]
    outcomes = [0, 0, 1, 1]
    bins = bin_of(chances, 2)
    rows = per_match(["m1", "m1", "m2", "m2"], chances, outcomes, bins, 2)
    totals = [0.0] * 6
    for row in rows.values():
        for i in range(6):
            totals[i] += row[i]
    _, found_gap = scores_from(totals)
    assert abs(found_gap - 0.9) < 1e-12
    assert abs(ece(reliability(chances, outcomes, 2)) - 0.9) < 1e-12
