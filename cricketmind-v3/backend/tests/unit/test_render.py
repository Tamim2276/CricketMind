import dataclasses

import pytest

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.registry import CLAIM_TYPES
from cricketmind_v3.domain.render import (
    RenderError,
    counted,
    fixed,
    percent,
    render,
    render_claim,
    signed,
    signed_points,
)
from cricketmind_v3.domain.subject import CricketSubject
from tests.examples import DOCUMENT_TEMPLATE, c1, c2, c3

# One test per kind of claim


def test_a_counted_claim_shows_its_balls():
    assert render_claim(c1()) == "a strike rate of 142.0 [D, 50 balls]"


def test_a_prediction_shows_its_range():
    assert render_claim(c2()) == "a 41% chance [P, range 16% to 70%]"


def test_a_measurement_shows_its_clip():
    assert render_claim(c3()) == "a weight transfer of 0.31 [M, clip vid1_0]"


def test_a_dismissal_rate_is_a_percent_per_ball():
    claim = dataclasses.replace(c1(), metric="dismissal_rate", value=0.08)
    assert render_claim(claim) == "a dismissal rate of 8.0% per ball [D, 50 balls]"


def test_one_ball_is_singular():
    claim = dataclasses.replace(c1(), n=1)
    assert render_claim(claim) == "a strike rate of 142.0 [D, 1 ball]"


# Rounding: halves always go up (chosen 2026-10-17)


def test_halves_round_up():
    assert percent(0.405, 0) == "41%"  # Python's own rounding gives 40%
    assert percent(0.155, 0) == "16%"
    assert percent(0.145, 0) == "15%"  # 0.145 * 100 in float is 14.4999...
    assert fixed(0.125, 2) == "0.13"  # Python's own rounding gives 0.12


def test_below_a_half_rounds_down():
    assert percent(0.404, 0) == "40%"
    assert fixed(142.04, 1) == "142.0"


# A whole finding


def test_the_documents_finding_becomes_its_sentence():
    finding = Finding(agent="N11b", claims=(c1(), c2()), template=DOCUMENT_TEMPLATE)
    assert render(finding) == (
        "Against pace in the powerplay, Kohli's cover drive returns "
        "a strike rate of 142.0 [D, 50 balls]. "
        "Six short balls carry a 41% chance [P, range 16% to 70%] of a dismissal."
    )


def test_a_slot_used_twice_is_filled_twice():
    finding = Finding(agent="N11b", claims=(c1(),), template="{c1}; again, {c1}.")
    expected = "a strike rate of 142.0 [D, 50 balls]; again, a strike rate of 142.0 [D, 50 balls]."
    assert render(finding) == expected


def test_an_unknown_metric_is_refused():
    claim = dataclasses.replace(c1(), metric="pitch_pace")
    with pytest.raises(RenderError):
        render_claim(claim)


# Differences show their sign (Day 11)


def test_a_difference_shows_its_sign():
    assert signed(30.0, 1) == "+30.0"
    assert signed(-12.54, 1) == "-12.5"


def test_zero_has_no_sign():
    assert signed(0.0, 1) == "0.0"
    assert signed(-0.04, 1) == "0.0"  # rounds to -0.0 without the fix


def test_a_gap_between_chances_is_in_percentage_points():
    assert signed_points(0.12, 0) == "+12"
    assert signed_points(-0.05, 0) == "-5"


# The six phrases added on Day 11


def test_a_strike_rate_difference():
    claim = dataclasses.replace(c1(), metric="sr_difference", value=30.0, n=100)
    assert render_claim(claim) == "a strike-rate difference of +30.0 [D, 100 balls]"


def test_a_difference_from_the_average_batter():
    claim = dataclasses.replace(c1(), metric="sr_vs_average", value=-8.25)
    assert render_claim(claim) == (
        "a strike-rate difference of -8.3 from the average batter [D, 50 balls]"
    )


def test_a_venue_average_counts_innings():
    claim = dataclasses.replace(
        c1(), metric="venue_first_innings_avg", value=167.4, n=38
    )
    assert render_claim(claim) == "a first-innings average of 167 runs [D, 38 innings]"


def test_one_innings_stays_innings():
    claim = dataclasses.replace(c1(), metric="venue_first_innings_avg", n=1)
    assert render_claim(claim).endswith("[D, 1 innings]")


def test_a_workload_ratio():
    claim = dataclasses.replace(c1(), metric="workload_ratio_flag", value=1.6, n=312)
    # the flag's words come from the number, never an injury chance (Day 41)
    assert render_claim(claim) == (
        "a workload ratio of 1.60, a spike above his usual week [D, 312 balls]"
    )
    usual = dataclasses.replace(claim, value=1.5)
    assert render_claim(usual) == (
        "a workload ratio of 1.50, close to his usual week [D, 312 balls]"
    )
    low = dataclasses.replace(claim, value=0.5)
    assert render_claim(low) == (
        "a workload ratio of 0.50, well below his usual week [D, 312 balls]"
    )


def test_a_review_overturn_rate_counts_reviews():
    claim = dataclasses.replace(c1(), metric="review_overturn_rate", value=0.274, n=120)
    assert render_claim(claim) == "a review overturn rate of 27% [D, 120 reviews]"
    one = dataclasses.replace(claim, n=1)
    assert render_claim(one).endswith("[D, 1 review]")


def test_a_matchup_difference_shows_signed_points_and_range():
    claim = dataclasses.replace(
        c2(), metric="matchup_difference", value=0.12, interval=(-0.05, 0.30)
    )
    assert render_claim(claim) == (
        "a difference of +12 percentage points in the chance of a dismissal "
        "[P, range -5 to +30]"
    )


def test_every_claim_type_on_the_menu_renders():
    for claim_type in CLAIM_TYPES:
        claim = Claim(
            id="c1",
            kind=claim_type.provenance,
            metric=claim_type.name,
            subject=CricketSubject(),
            value=0.1,
            n=50,
            interval=(0.0, 0.2),
            clip_id="vid1_0",
        )
        text = render_claim(claim)
        assert "{" not in text, claim_type.name
        assert text.endswith("]"), claim_type.name


# A comparison shows its range, even when counted (invariant I4)


def test_a_counted_comparison_shows_its_range():
    claim = dataclasses.replace(
        c1(), metric="sr_difference", value=30.0, n=100, interval=(-16.0, 76.0)
    )
    assert render_claim(claim) == (
        "a strike-rate difference of +30.0 [D, 100 balls, range -16.0 to +76.0]"
    )


# Sample sizes in words (Day 20)


def test_counted():
    assert counted(50, "balls") == "50 balls"
    assert counted(1, "balls") == "1 ball"
    assert counted(1, "innings") == "1 innings"
    assert counted(1, "reviews") == "1 review"
    assert counted(1, "wickets") == "1 wicket"  # Day 40
