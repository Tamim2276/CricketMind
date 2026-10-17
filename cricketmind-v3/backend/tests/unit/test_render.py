import dataclasses

import pytest

from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.render import (
    RenderError,
    fixed,
    percent,
    render,
    render_claim,
)
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
