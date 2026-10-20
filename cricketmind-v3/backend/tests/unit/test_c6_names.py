import dataclasses

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.c6_names import check_c6
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from tests.examples import DOCUMENT_TEMPLATE, KOHLI_COVER_DRIVE, KOHLI_PULL, c1, c2

PLAYERS = {
    "V Kohli": ("Kohli", "Virat Kohli", "Virat"),
    "RG Sharma": ("Rohit", "Rohit Sharma"),
    "JJ Bumrah": ("Bumrah",),
    "Rashid Khan": ("Rashid",),
}
NAMES = Context(names=cricket_gazetteer(PLAYERS))


def reasons(template: str, *claims: Claim, context: Context = NAMES) -> list[str]:
    finding = Finding(agent="N11b", claims=tuple(claims), template=template)
    found = []
    for failure in check_c6(finding, context):
        assert failure.check == "C6"
        found.append(failure.reason)
    return found


def test_the_document_finding_passes():
    assert reasons(DOCUMENT_TEMPLATE, c1(), c2()) == []


# The plan's example: Kohli's number in a sentence about Rohit (F2)


def test_the_rohit_kohli_swap_fails():
    template = "Against pace in the powerplay, Rohit's cover drive returns {c1}."
    assert reasons(template, c1()) == [
        "the sentence names 'rohit', but c1 is about player V Kohli",
    ]


def test_the_wrong_shot_fails():
    assert reasons("Kohli's pull returns {c1}.", c1()) == [
        "the sentence names 'pull', but c1 is about shot Cover Drive",
    ]


def test_the_wrong_phase_and_bowler_type_fail():
    template = "Against spin at the death, Kohli's cover drive returns {c1}."
    assert reasons(template, c1()) == [
        "the sentence names 'spin', but c1 is about bowler_type pace",
        "the sentence names 'at the death', but c1 is about phase powerplay",
    ]


def test_the_sentence_cannot_narrow_the_claim():
    # the number covers every shot; the sentence says cover drive
    every_shot = dataclasses.replace(
        c1(), subject=dataclasses.replace(KOHLI_COVER_DRIVE, shot=None)
    )
    assert reasons("Kohli's cover drive returns {c1}.", every_shot) == [
        "the sentence names 'cover drive', but c1 is about every shot",
    ]


# Comparisons have two sides


def test_the_other_shot_of_a_comparison_may_be_named():
    vs_pull = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="sr_difference",
        subject=dataclasses.replace(KOHLI_COVER_DRIVE, other_shot="Pull"),
        value=30.0,
        n=50,
        interval=(-16.0, 76.0),
        source="N8.M1",
    )
    template = (
        "Against the pull, Kohli's cover drive shows no detectable difference: {c1}."
    )
    assert reasons(template, vs_pull) == []


def test_both_bowlers_of_a_matchup_may_be_named():
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
    assert reasons("Bumrah or Rashid against Kohli: {c1}.", matchup) == []


# What C6 does not judge


def test_a_sentence_without_a_number_is_not_judged():
    template = "Rohit opens the batting. Kohli's cover drive returns {c1}."
    assert reasons(template, c1()) == []


def test_reverse_sweep_is_not_read_as_sweep():
    reverse = dataclasses.replace(
        c1(), subject=dataclasses.replace(KOHLI_COVER_DRIVE, shot="Reverse Sweep")
    )
    assert reasons("Kohli's reverse sweep returns {c1}.", reverse) == []


def test_a_surname_shared_by_two_players_fits_either():
    shared = Context(
        names=cricket_gazetteer({"RG Sharma": ("Sharma",), "I Sharma": ("Sharma",)})
    )
    claim = dataclasses.replace(
        c1(), subject=dataclasses.replace(KOHLI_COVER_DRIVE, player="I Sharma")
    )
    assert reasons("Sharma's cover drive returns {c1}.", claim, context=shared) == []


def test_case_and_possessives_do_not_matter():
    assert reasons("KOHLI'S Cover-Drive returns {c1}.", c1()) == []


def test_a_name_not_in_the_list_is_not_judged_a_known_gap():
    # nicknames come with the register on Day 33 (L-019)
    assert reasons("King's cover drive returns {c1}.", c1()) == []
