import dataclasses

import pytest

from cricketmind_v3.domain.claims import Claim, ClaimError
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import Subject

KOHLI_COVER_DRIVE = Subject(
    player="V Kohli", shot="Cover Drive", phase="powerplay", bowler_type="pace"
)
KOHLI_PULL = Subject(player="V Kohli", shot="Pull", phase="powerplay", bowler_type="pace")


# Three good claims; each test below breaks one rule on a copy.


def c1() -> Claim:
    # the v3 document's c1: counted from history
    return Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="strike_rate",
        subject=KOHLI_COVER_DRIVE,
        value=142.0,
        n=50,
    )


def c2() -> Claim:
    # the v3 document's c2: a prediction with its range
    return Claim(
        id="c2",
        kind=Provenance.PREDICTIVE,
        metric="p_dismissal_within_k",
        subject=KOHLI_PULL,
        value=0.405,
        interval=(0.155, 0.697),
    )


def c3() -> Claim:
    # measured from a clip
    return Claim(
        id="c3",
        kind=Provenance.MEASURED,
        metric="weight_transfer",
        subject=Subject(player="V Kohli", shot="Cover Drive"),
        value=0.31,
        clip_id="vid1_0",
    )


def test_the_documents_c1_is_accepted():
    claim = c1()
    assert claim.value == 142.0
    assert claim.subject.player == "V Kohli"


def test_the_documents_c2_is_accepted():
    assert c2().interval == (0.155, 0.697)


def test_a_measured_claim_with_its_clip_is_accepted():
    assert c3().clip_id == "vid1_0"


def test_a_claim_cannot_change():
    claim = c1()
    with pytest.raises(dataclasses.FrozenInstanceError):
        claim.value = 162.0


def test_rule_1_id_must_look_like_c1():
    for bad_id in ["claim1", "c", "C1", "c1a", "1"]:
        with pytest.raises(ClaimError):
            dataclasses.replace(c1(), id=bad_id)


def test_rule_2_a_descriptive_claim_needs_n():
    with pytest.raises(ClaimError):
        dataclasses.replace(c1(), n=None)
    with pytest.raises(ClaimError):
        dataclasses.replace(c1(), n=0)


def test_rule_3_a_predictive_claim_needs_a_range():
    with pytest.raises(ClaimError):
        dataclasses.replace(c2(), interval=None)


def test_rule_3_the_value_must_sit_inside_its_range():
    with pytest.raises(ClaimError):
        dataclasses.replace(c2(), value=0.9)


def test_rule_4_a_measured_claim_needs_its_clip():
    with pytest.raises(ClaimError):
        dataclasses.replace(c3(), clip_id=None)


def test_the_error_names_the_claim_and_the_rule():
    with pytest.raises(ClaimError) as error:
        dataclasses.replace(c2(), interval=None)
    assert str(error.value) == "c2: a predictive claim needs a range"
