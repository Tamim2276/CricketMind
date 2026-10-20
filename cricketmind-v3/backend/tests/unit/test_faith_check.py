import dataclasses

from cricketmind_v3.application.faith_check import faith_check, recompute
from cricketmind_v3.application.ports import FakeFactSource
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.faith.c2_values import check_c2
from cricketmind_v3.domain.faith.gate import Context
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.subject import CricketSubject
from tests.examples import KOHLI_COVER_DRIVE, KOHLI_PULL, c1, c2

ROHIT_COVER_DRIVE = dataclasses.replace(KOHLI_COVER_DRIVE, player="RG Sharma")


def engines() -> FakeFactSource:
    # synthetic: Kohli's numbers from the document, Rohit's invented for the test
    source = FakeFactSource()
    source.add("strike_rate", KOHLI_COVER_DRIVE, Fact(value=142.0, n=50))
    source.add("strike_rate", ROHIT_COVER_DRIVE, Fact(value=118.0, n=44))
    source.add(
        "p_dismissal_within_k", KOHLI_PULL, Fact(value=0.405, interval=(0.155, 0.697))
    )
    return source


# The fake engine


def test_the_fake_gives_back_what_was_added():
    assert engines().fact("strike_rate", KOHLI_COVER_DRIVE) == Fact(value=142.0, n=50)


def test_the_fake_has_nothing_for_an_unknown_subject():
    nobody = CricketSubject(player="Nobody")
    assert engines().fact("strike_rate", nobody) is None


def test_an_equal_subject_finds_the_same_fact():
    # a new object with the same fields is the same subject
    same = CricketSubject(
        player="V Kohli", shot="Cover Drive", phase="powerplay", bowler_type="pace"
    )
    assert engines().fact("strike_rate", same) == Fact(value=142.0, n=50)


# Asking again


def test_every_claim_on_the_menu_is_asked_again():
    finding = Finding(agent="N11b", claims=(c1(), c2()), template="{c1} {c2}.")
    facts = recompute(finding, engines())
    assert facts == {
        "c1": Fact(value=142.0, n=50),
        "c2": Fact(value=0.405, interval=(0.155, 0.697)),
    }


def test_a_claim_type_not_on_the_menu_is_not_asked():
    hollow = dataclasses.replace(c1(), metric="pitch_holding_pace")
    finding = Finding(agent="N11b", claims=(hollow,), template="{c1}.")
    assert recompute(finding, engines()) == {}


# The whole node: ask again, then check


def test_a_correct_finding_passes():
    finding = Finding(agent="N11b", claims=(c1(), c2()), template="{c1} {c2}.")
    assert faith_check(finding, engines(), Context(), (check_c2,)) == []


def test_kohlis_number_given_to_rohit_is_caught_f2():
    # the right value (142.0), the wrong subject: Rohit's own numbers differ
    stolen = dataclasses.replace(c1(), subject=ROHIT_COVER_DRIVE)
    finding = Finding(agent="N11b", claims=(stolen,), template="{c1}.")
    reasons = []
    for failure in faith_check(finding, engines(), Context(), (check_c2,)):
        reasons.append(failure.reason)
    assert reasons == [
        "the claim says 142.0, but the engine gives 118.0 for this subject",
        "the claim rests on 50, but the engine counts 44",
    ]
