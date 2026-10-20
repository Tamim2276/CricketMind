import dataclasses

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.c4_wording import check_c4
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from tests.examples import KOHLI_COVER_DRIVE, KOHLI_PULL, c1, compared

HEDGE = ", though the estimate is very uncertain"


def chance(value: float, interval: tuple[float, float], claim_id: str = "c2") -> Claim:
    return Claim(
        id=claim_id,
        kind=Provenance.PREDICTIVE,
        metric="p_dismissal_within_k",
        subject=KOHLI_PULL,
        value=value,
        interval=interval,
        source="N9.T1",
    )


def reasons(template: str, *claims: Claim) -> list[str]:
    finding = Finding(agent="N11b", claims=tuple(claims), template=template)
    found = []
    for failure in check_c4(finding, Context()):
        assert failure.check == "C4"
        found.append(failure.reason)
    return found


DOCUMENT_CHANCE = chance(0.405, (0.155, 0.697))  # about even; the range spans 3 bands


# Part 1: words about a chance (the plan's example: p = 0.405)


def test_will_probably_get_him_out_fails_at_41_percent():
    template = (
        "Six short balls carry {c2}: they will probably get him out" + HEDGE + "."
    )
    assert reasons(template, DOCUMENT_CHANCE) == [
        "'will' is too strong for any chance a model gives",
        "'probably' does not fit a 41% chance, which is 'about even'",
    ]


def test_the_honest_wording_passes():
    template = (
        "Six short balls carry {c2} of a dismissal, about as likely as not" + HEDGE
    )
    assert reasons(template + ".", DOCUMENT_CHANCE) == []


def test_a_range_across_bands_must_be_admitted():
    template = "Six short balls carry {c2} of a dismissal, about as likely as not."
    assert reasons(template, DOCUMENT_CHANCE) == [
        (
            "the range 16% to 70% spans more than one band; "
            "admit it, e.g. 'the estimate is very uncertain'"
        ),
    ]


def test_a_range_inside_one_band_needs_no_hedge():
    narrow = chance(0.50, (0.45, 0.55))
    assert reasons("{c2}: as likely as not.", narrow) == []


def test_no_words_about_size_is_fine():
    # the number and its range speak for themselves
    narrow = chance(0.50, (0.45, 0.55))
    assert reasons("Six short balls carry {c2} of a dismissal.", narrow) == []


# Each band


def test_a_remote_chance():
    remote = chance(0.05, (0.02, 0.10))
    assert reasons("{c2}: almost certainly not.", remote) == []
    assert reasons("{c2}: unlikely.", remote) == [
        "'unlikely' does not fit a 5% chance, which is 'almost certainly not'",
    ]
    assert reasons("{c2}: possible.", remote) == [
        "'possible' does not fit a 5% chance, which is 'almost certainly not'",
    ]


def test_a_likely_chance():
    likely = chance(0.70, (0.62, 0.80))
    assert reasons("{c2}: likely.", likely) == []
    assert reasons("{c2}: almost certain.", likely) == [
        "'almost certain' does not fit a 70% chance, which is 'probable'",
    ]


def test_almost_certain_is_not_certain():
    sure = chance(0.95, (0.90, 0.98))
    assert reasons("{c2}: almost certain.", sure) == []  # longest phrase first
    assert reasons("{c2}: certain.", sure) == [
        "'certain' is too strong for any chance a model gives",
    ]


def test_possible_says_nothing_about_size_in_the_middle():
    assert reasons("{c2}: possible.", chance(0.50, (0.45, 0.55))) == []


def test_two_chances_in_one_sentence():
    high = chance(0.70, (0.62, 0.80), claim_id="c1")
    low = chance(0.20, (0.15, 0.30), claim_id="c2")
    template = "{c1} is likely against Bumrah, {c2} unlikely against Rashid."
    assert reasons(template, high, low) == []


def test_a_count_is_not_judged_by_bands():
    assert reasons("He will keep scoring at {c1}.", c1()) == []


# Part 2: differences (the plan's example: 142 vs 112, range -16 to +76)


def cover_vs_pull(interval: tuple[float, float] = (-16.0, 76.0)) -> Claim:
    vs_pull = dataclasses.replace(KOHLI_COVER_DRIVE, other_shot="Pull")
    return Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="sr_difference",
        subject=vs_pull,
        value=30.0,
        n=50,
        interval=interval,
        source="N8.M1",
    )


def test_the_stronger_stroke_fails_when_the_range_includes_zero_f6():
    template = "Against the pull, the cover drive is the stronger stroke: {c1}."
    assert reasons(template, cover_vs_pull()) == [
        (
            "the range -16.0 to +76.0 includes zero, so there may be no difference "
            "at all; say 'no detectable difference'"
        ),
        "'stronger' claims a difference the range -16.0 to +76.0 does not show",
    ]


def test_saying_no_detectable_difference_passes():
    template = "Against the pull, the cover drive shows no detectable difference: {c1}."
    assert reasons(template, cover_vs_pull()) == []


def test_a_range_touching_zero_includes_it():
    template = "The cover drive is better than the pull: {c1}."
    touching = cover_vs_pull(interval=(0.0, 60.0))
    assert len(reasons(template, touching)) == 2


def test_a_clear_difference_may_be_called_one():
    clear = compared()  # +22 against the average batter, range +8 to +36
    assert reasons("He scores better than the average batter: {c1}.", clear) == []


def test_a_clear_difference_cannot_be_called_none():
    clear = compared()
    assert reasons(
        "No detectable difference from the average batter: {c1}.", clear
    ) == [
        (
            "the range +8.0 to +36.0 does not include zero: the difference is "
            "detectable, so do not say there is none"
        ),
    ]


def test_a_matchup_difference_is_judged_as_a_difference_not_a_chance():
    # +12 points is a gap between two chances, not a 12% chance
    matchup = Claim(
        id="c1",
        kind=Provenance.PREDICTIVE,
        metric="matchup_difference",
        subject=dataclasses.replace(
            KOHLI_PULL, bowler="JJ Bumrah", other_bowler="Rashid Khan"
        ),
        value=0.12,
        interval=(-0.05, 0.30),
        source="N9.T2",
    )
    assert reasons("Bumrah or Rashid: no detectable difference, {c1}.", matchup) == []
    assert reasons("Bumrah is the better option: {c1}.", matchup) == [
        (
            "the range -5 to +30 includes zero, so there may be no difference "
            "at all; say 'no detectable difference'"
        ),
        "'better' claims a difference the range -5 to +30 does not show",
    ]


# Found on review day (Day 27): a gate reports, it never crashes


def test_a_chance_above_100_percent_is_reported_not_a_crash():
    impossible = chance(1.4, (1.2, 1.6))
    assert reasons("Six short balls carry {c2}: likely.", impossible) == [
        "a chance and its range must lie between 0% and 100%",
    ]
