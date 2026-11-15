import dataclasses

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.c6_names import check_c6
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer, gate_gazetteer
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import CricketSubject
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


# Everyday words are also people in the real register (found on Day 66)

CROWDED = Context(
    names=cricket_gazetteer(
        {
            "V Kohli": ("Kohli",),
            "JT Ball": ("Ball",),
            "E Chance": ("Chance",),
            "YS To": ("To",),
            "VV More": ("More",),
            "M Short": ("Short",),
            "S Hook": ("Hook",),
        }
    )
)


def crowded(template: str) -> list[str]:
    return reasons(template, c1(), context=CROWDED)


def test_an_everyday_word_is_not_a_players_name():
    assert crowded("Kohli's chance to score on each ball is {c1}.") == []


def test_more_at_the_start_of_a_sentence_is_the_word():
    assert crowded("More often than not Kohli's cover drive returns {c1}.") == []


def test_short_at_the_start_of_a_sentence_is_the_word():
    assert crowded("Short balls to Kohli's cover drive return {c1}.") == []


def test_an_everyday_name_with_a_capital_in_the_middle_is_the_player():
    assert crowded("Against pace, Short's cover drive returns {c1}.") == [
        "the sentence names 'short', but c1 is about player V Kohli"
    ]


def test_a_name_at_the_start_of_a_sentence_is_still_a_name():
    assert reasons("Rohit returns {c1}.", c1()) == [
        "the sentence names 'rohit', but c1 is about player V Kohli"
    ]


def test_a_word_that_is_a_shot_and_a_name_is_the_shot_without_a_capital():
    # hook is a shot, and S Hook a player: lower case, only the shot is meant
    assert crowded("Kohli's hook returns {c1}.") == [
        "the sentence names 'hook', but c1 is about shot Cover Drive"
    ]


def test_a_name_written_in_lower_case_is_not_read_a_known_gap():
    # a lower-case surname cannot be told from a word (L-083)
    assert reasons("rohit returns {c1}.", c1()) == []


def test_a_shot_needs_no_capital():
    assert reasons("Kohli's pull returns {c1}.", c1()) != []


# Teams and grounds on a real run (found on Day 77: "Lanka" read as a player)

ON_A_REAL_RUN = Context(
    names=gate_gazetteer(
        {"V Kohli": ("Kohli",), "D Lanka": ("Lanka",), "BCJ Heat": ("Heat",)},
        ("India", "Sri Lanka", "Pakistan", "Brisbane Heat", "Heat"),
        ("Eden Gardens, Kolkata", "Wankhede Stadium"),
    )
)


def team_claim(**fields) -> Claim:
    return dataclasses.replace(c1(), subject=CricketSubject(**fields))


def real_run(template: str, claim: Claim) -> list[str]:
    return reasons(template, claim, context=ON_A_REAL_RUN)


def test_a_team_with_a_players_surname_in_it_is_the_team():
    claim = team_claim(team="Pakistan", opposition="Sri Lanka")
    assert real_run("Against Sri Lanka, Pakistan won at {c1}.", claim) == []


def test_a_team_the_claim_is_not_about_is_stopped():
    claim = team_claim(team="Pakistan", opposition="Sri Lanka")
    expected = (
        "the sentence names 'india', but c1 is about team Pakistan and "
        "opposition Sri Lanka"
    )
    assert real_run("Against India, Pakistan won at {c1}.", claim) == [expected]


def test_a_team_cannot_narrow_a_players_number():
    claim = team_claim(player="V Kohli")
    assert real_run("For India, Kohli returns {c1}.", claim) == [
        "the sentence names 'india', but c1 is about every team"
    ]


def test_a_ground_the_claim_is_about_passes_by_either_name():
    long_name = team_claim(venue="Eden Gardens, Kolkata")
    assert real_run("Eden Gardens gives {c1}.", long_name) == []
    assert (
        real_run("At Eden Gardens, Kolkata the first innings gives {c1}.", long_name)
        == []
    )


def test_another_ground_is_stopped():
    claim = team_claim(venue="Wankhede Stadium")
    assert real_run("Eden Gardens gives {c1}.", claim) != []


def test_an_everyday_word_is_not_a_team():
    claim = team_claim(player="V Kohli")
    assert real_run("In this heat, Kohli returns {c1}.", claim) == []


def test_the_readers_list_stays_as_it_was():
    # N3's rules read `cricket_gazetteer`: the fair test of the readers ran on it (N-044)
    assert "india" not in cricket_gazetteer({"V Kohli": ("Kohli",)})
