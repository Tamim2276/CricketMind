"""Ready-made subjects and claims from the v3 document, shared by many tests."""

from cricketmind_v3.domain.balls import Ball
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import CricketSubject

KOHLI_COVER_DRIVE = CricketSubject(
    player="V Kohli", shot="Cover Drive", phase="powerplay", bowler_type="pace"
)
KOHLI_PULL = CricketSubject(
    player="V Kohli",
    shot="Pull",
    phase="powerplay",
    bowler_type="pace",
    k=6,
    match_format="T20",  # a prediction names its format (Day 50)
)

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
        source="N8.M1",
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
        source="N9.T1",
    )


def c3() -> Claim:
    # measured from a clip
    return Claim(
        id="c3",
        kind=Provenance.MEASURED,
        metric="weight_transfer",
        subject=CricketSubject(player="V Kohli", shot="Cover Drive"),
        value=0.31,
        clip_id="vid1_0",
        source="N6",
    )


KOHLI_SPIN = CricketSubject(player="V Kohli", phase="powerplay", bowler_type="spin")


def compared(
    claim_id: str = "c1",
    value: float = 22.0,
    interval: tuple[float, float] = (8.0, 36.0),
    n: int = 50,
    subject: CricketSubject = KOHLI_COVER_DRIVE,
) -> Claim:
    # sr_vs_average: the player against the average batter with the same filters
    return Claim(
        id=claim_id,
        kind=Provenance.DESCRIPTIVE,
        metric="sr_vs_average",
        subject=subject,
        value=value,
        n=n,
        interval=interval,
        source="N8.M1",
    )


# SYNTHETIC: the v2 corpus's balls (N-001), rebuilt as v2's corpus.py built them. None of
# this is real match data.


def v2_runs(balls: int, runs: int) -> list[int]:
    """v2's recipe: a repeating run pattern, nudged a run at a time until it adds up."""
    pattern = [1, 0, 4, 1, 2, 0, 1, 1, 0, 2]
    each = []
    for i in range(balls):
        each.append(pattern[i % len(pattern)])
    missing = runs - sum(each)
    i = 0
    while missing != 0 and i < balls * 8:
        j = i % balls
        if missing > 0 and each[j] < 6:
            each[j] += 1
            missing -= 1
        elif missing < 0 and each[j] > 0:
            each[j] -= 1
            missing += 1
        i += 1
    return each


def v2_balls(runs: int, outs: int, balls: int = 50) -> list[Ball]:
    """Kohli's balls of one shot; the first `outs` balls are outs (v2 gave no kind)."""
    made = []
    for i, scored in enumerate(v2_runs(balls, runs)):
        out = None
        kind = None
        if i < outs:
            out = "V Kohli"
            kind = "caught"
        made.append(
            Ball(
                match_type="T20",
                gender="male",
                batter="V Kohli",
                runs_batter=scored,
                wides=0,
                noballs=0,
                player_out=out,
                wicket_kind=kind,
            )
        )
    return made


V2_COVER_DRIVE = v2_balls(runs=71, outs=1)  # strike rate 142.0
V2_PULL = v2_balls(runs=56, outs=4)  # strike rate 112.0
