import dataclasses

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.c1_registry import check_c1
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import CricketSubject
from tests.examples import DOCUMENT_TEMPLATE, KOHLI_COVER_DRIVE, c1, c2

# N11b (Tactical Analysis) may say these; passed in by hand until the agent specs (Day 65)
ROSTER = Context(
    allowed={
        "N11b": (
            "strike_rate",
            "sr_difference",
            "sr_vs_average",
            "p_dismissal_within_k",
        ),
        "N11k": ("workload_ratio_flag",),
    }
)


def finding_with(*claims: Claim, agent: str = "N11b") -> Finding:
    slots = []
    for claim in claims:
        slots.append("{" + claim.id + "}")
    return Finding(agent=agent, claims=tuple(claims), template=" ".join(slots) + ".")


def reasons(failures) -> list[str]:
    found = []
    for failure in failures:
        assert failure.check == "C1"
        found.append(failure.reason)
    return found


def test_the_document_finding_passes():
    finding = Finding(agent="N11b", claims=(c1(), c2()), template=DOCUMENT_TEMPLATE)
    assert check_c1(finding, ROSTER) == []


def test_a_claim_type_not_on_the_menu_fails_once():
    hollow = dataclasses.replace(c1(), metric="pitch_holding_pace", source="N8.M2")
    failures = check_c1(finding_with(hollow), ROSTER)
    assert len(failures) == 1  # not also "not allowed", "wrong source", ...
    assert failures[0].claim == "c1"
    assert "'pitch_holding_pace' is not on the menu" in failures[0].reason


def test_an_agent_may_only_assert_its_own_claim_types():
    failures = check_c1(finding_with(c1(), agent="N11k"), ROSTER)
    assert reasons(failures) == ["N11k may not assert strike_rate"]


def test_an_agent_not_on_the_roster_fails():
    failures = check_c1(finding_with(c1(), agent="N11z"), ROSTER)
    assert len(failures) == 1
    assert failures[0].claim == ""  # about the whole finding
    assert "not an agent on the roster" in failures[0].reason


def test_a_claim_must_name_its_source():
    failures = check_c1(finding_with(dataclasses.replace(c1(), source=None)), ROSTER)
    assert reasons(failures) == [
        "the claim names no source; strike_rate comes from N8.M1",
    ]


def test_a_claim_must_name_the_right_source():
    wrong = dataclasses.replace(c1(), source="N9.T1")
    failures = check_c1(finding_with(wrong), ROSTER)
    assert reasons(failures) == [
        "the claim says it comes from N9.T1, but strike_rate comes from N8.M1",
    ]


def test_a_count_cannot_pretend_to_be_a_prediction():
    # a strike rate is counted (D); marked predicted, it would skip the evidence floor
    posing = dataclasses.replace(
        c1(), kind=Provenance.PREDICTIVE, interval=(130.0, 150.0)
    )
    failures = check_c1(finding_with(posing), ROSTER)
    assert reasons(failures) == [
        "strike_rate is descriptive, but the claim says predictive",
    ]


def test_a_claim_must_say_whose_number_it_is():
    nobody = dataclasses.replace(c1(), subject=CricketSubject(shot="Cover Drive"))
    failures = check_c1(finding_with(nobody), ROSTER)
    assert reasons(failures) == ["strike_rate needs to know: player"]


def test_every_missing_field_is_named():
    thin = dataclasses.replace(c2(), subject=CricketSubject(player="V Kohli"))
    failures = check_c1(finding_with(thin), ROSTER)
    assert reasons(failures) == [
        "p_dismissal_within_k needs to know: shot, phase, bowler_type, k",
    ]


def test_a_comparison_must_carry_its_range():
    vs_pull = dataclasses.replace(KOHLI_COVER_DRIVE, other_shot="Pull")
    no_range = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="sr_difference",
        subject=vs_pull,
        value=30.0,
        n=50,
        source="N8.M1",
    )
    failures = check_c1(finding_with(no_range), ROSTER)
    assert reasons(failures) == [
        "sr_difference must carry its range, so its uncertainty shows (C4)",
    ]


def test_every_problem_with_one_claim_is_listed():
    messy = dataclasses.replace(
        c1(), source=None, subject=CricketSubject(shot="Cover Drive")
    )
    failures = check_c1(finding_with(messy, agent="N11k"), ROSTER)
    assert len(failures) == 3  # not allowed, no source, no player
