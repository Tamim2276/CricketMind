import dataclasses

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.faith.c2_values import check_c2
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from tests.examples import KOHLI_COVER_DRIVE, c1, c2, c3

# what the engine says when asked again (synthetic, the document's numbers)
ENGINE = {
    "c1": Fact(value=142.0, n=50),
    "c2": Fact(value=0.405, interval=(0.155, 0.697)),
    "c3": Fact(value=0.31),
}


def finding_with(*claims: Claim) -> Finding:
    slots = []
    for claim in claims:
        slots.append("{" + claim.id + "}")
    return Finding(agent="N11b", claims=tuple(claims), template=" ".join(slots) + ".")


def reasons(finding: Finding, facts=None) -> list[str]:
    if facts is None:
        facts = ENGINE
    found = []
    for failure in check_c2(finding, Context(facts=facts)):
        assert failure.check == "C2"
        found.append(failure.reason)
    return found


def test_the_documents_claims_match_the_engine():
    assert reasons(finding_with(c1(), c2(), c3())) == []


def test_a_wrong_value_is_caught_f1():
    wrong = dataclasses.replace(c1(), value=162.0)
    assert reasons(finding_with(wrong)) == [
        "the claim says 162.0, but the engine gives 142.0 for this subject",
    ]


def test_a_difference_too_small_to_see_passes():
    # 142.04 and 142.0 both show as 142.0: the reader sees the same number
    assert reasons(finding_with(dataclasses.replace(c1(), value=142.04))) == []


def test_a_difference_the_reader_would_see_fails():
    # 0.4049 shows as 40%, the engine's 0.405 as 41%
    nearly = dataclasses.replace(c2(), value=0.4049)
    assert reasons(finding_with(nearly)) == [
        "the claim says 40%, but the engine gives 41% for this subject",
    ]


def test_the_sample_size_must_match_exactly():
    # claiming 500 balls instead of 50 would dodge the evidence floor
    inflated = dataclasses.replace(c1(), n=500)
    assert reasons(finding_with(inflated)) == [
        "the claim rests on 500, but the engine counts 50",
    ]


def test_the_range_must_match():
    narrower = dataclasses.replace(c2(), interval=(0.30, 0.50))
    assert reasons(finding_with(narrower)) == [
        "the claim's range is 30% to 50%, but the engine's is 16% to 70%",
    ]


def test_a_comparison_cannot_leave_out_the_engines_range():
    vs_pull = dataclasses.replace(KOHLI_COVER_DRIVE, other_shot="Pull")
    claim = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="sr_difference",
        subject=vs_pull,
        value=30.0,
        n=50,
        source="N8.M1",
    )
    facts = {"c1": Fact(value=30.0, n=50, interval=(-16.0, 76.0))}
    assert reasons(finding_with(claim), facts) == [
        "the claim leaves out the engine's range",
    ]


def test_a_claim_cannot_invent_a_range():
    with_range = dataclasses.replace(c1(), interval=(130.0, 150.0))
    assert reasons(finding_with(with_range)) == [
        "the claim gives a range the engine does not have",
    ]


def test_no_engine_number_for_this_subject_fails():
    assert reasons(finding_with(c1()), {"c1": None}) == [
        "the engine has no number for this subject",
    ]


def test_a_number_never_rechecked_fails():
    # fail closed: if nobody asked the engine, the number is not shown
    assert reasons(finding_with(c1()), {}) == [
        "the number was never re-checked with its engine",
    ]


def test_a_claim_type_not_on_the_menu_is_left_to_c1():
    hollow = dataclasses.replace(c1(), metric="pitch_holding_pace")
    assert reasons(finding_with(hollow), {}) == []


def test_a_measurement_is_compared_too():
    shaky = dataclasses.replace(c3(), value=0.45)
    assert reasons(finding_with(shaky)) == [
        "the claim says 0.45, but the engine gives 0.31 for this subject",
    ]
