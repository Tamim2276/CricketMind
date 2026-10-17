"""Ready-made subjects and claims from the v3 document, shared by many tests."""

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import Subject

KOHLI_COVER_DRIVE = Subject(
    player="V Kohli", shot="Cover Drive", phase="powerplay", bowler_type="pace"
)
KOHLI_PULL = Subject(player="V Kohli", shot="Pull", phase="powerplay", bowler_type="pace")

DOCUMENT_TEMPLATE = (
    "Against pace in the powerplay, Kohli's cover drive returns {c1}. "
    "Six short balls carry {c2} of a dismissal."
)


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
