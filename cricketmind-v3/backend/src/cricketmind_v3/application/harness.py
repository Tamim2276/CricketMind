"""The fabrication harness: break a correct finding in each of the eight ways (F1-F8)
and see which gate notices.

SYNTHETIC. The numbers are the v2 synthetic corpus (N-001 to N-004) or invented for
the harness; they say nothing about real players.
"""

import dataclasses
from dataclasses import dataclass

from cricketmind_v3.application.faith_check import faith_check
from cricketmind_v3.application.ports import FakeFactSource
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.faith.checks import ALL_CHECKS, V2_MODE, V3_DESIGN_CHECKS
from cricketmind_v3.domain.faith.gate import Context, Failure
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import CricketSubject


class HarnessError(ValueError):
    """A broken version that is not what it claims to be."""


# The synthetic world

KOHLI = CricketSubject(
    player="V Kohli", shot="Cover Drive", phase="powerplay", bowler_type="pace"
)
KOHLI_VS_PULL = dataclasses.replace(KOHLI, other_shot="Pull")
KOHLI_PULL = dataclasses.replace(KOHLI, shot="Pull", k=6)
KOHLI_SPIN = CricketSubject(player="V Kohli", phase="powerplay", bowler_type="spin")
ROHIT = dataclasses.replace(KOHLI, player="RG Sharma")
VENUE = CricketSubject(venue="Wankhede Stadium", match_format="T20", gender="male")

ROSTER = {
    "N11b": ("strike_rate", "sr_vs_average", "sr_difference", "p_dismissal_within_k"),
}
PLAYERS = {
    "V Kohli": ("Kohli", "Virat Kohli", "Virat"),
    "RG Sharma": ("Rohit", "Rohit Sharma"),
}


def engines() -> FakeFactSource:
    """What the engines answer in the synthetic world."""
    source = FakeFactSource()
    source.add("strike_rate", KOHLI, Fact(value=142.0, n=50))  # N-001
    source.add("strike_rate", ROHIT, Fact(value=118.0, n=44))  # invented
    source.add("sr_vs_average", KOHLI, Fact(value=22.0, n=50, interval=(8.0, 36.0)))
    source.add(
        "sr_difference", KOHLI_VS_PULL, Fact(value=30.0, n=50, interval=(-16.0, 76.0))
    )  # N-001, N-002
    source.add(
        "p_dismissal_within_k", KOHLI_PULL, Fact(value=0.405, interval=(0.155, 0.697))
    )  # N-004
    source.add(
        "sr_vs_average", KOHLI_SPIN, Fact(value=-40.0, n=6, interval=(-90.0, -5.0))
    )  # invented: a small sample
    source.add("venue_first_innings_avg", VENUE, Fact(value=167.0, n=38))  # invented
    return source


def context(source: FakeFactSource) -> Context:
    return Context(
        allowed=ROSTER, names=cricket_gazetteer(PLAYERS), surface=source.numbers()
    )


# The correct finding: four claims, three sentences, every check passes

STRIKE_RATE = Claim(
    id="c1",
    kind=Provenance.DESCRIPTIVE,
    metric="strike_rate",
    subject=KOHLI,
    value=142.0,
    n=50,
    source="N8.M1",
)
VS_AVERAGE = Claim(
    id="c2",
    kind=Provenance.DESCRIPTIVE,
    metric="sr_vs_average",
    subject=KOHLI,
    value=22.0,
    n=50,
    interval=(8.0, 36.0),
    source="N8.M1",
)
VS_PULL = Claim(
    id="c3",
    kind=Provenance.DESCRIPTIVE,
    metric="sr_difference",
    subject=KOHLI_VS_PULL,
    value=30.0,
    n=50,
    interval=(-16.0, 76.0),
    source="N8.M1",
)
CHANCE = Claim(
    id="c4",
    kind=Provenance.PREDICTIVE,
    metric="p_dismissal_within_k",
    subject=KOHLI_PULL,
    value=0.405,
    interval=(0.155, 0.697),
    source="N9.T1",
)
SMALL_SAMPLE = Claim(
    id="c5",
    kind=Provenance.DESCRIPTIVE,
    metric="sr_vs_average",
    subject=KOHLI_SPIN,
    value=-40.0,
    n=6,
    interval=(-90.0, -5.0),
    source="N8.M1",
)
HOLLOW = Claim(
    id="c5",
    kind=Provenance.DESCRIPTIVE,
    metric="pitch_holding_pace",  # on no menu: no engine computes it
    subject=VENUE,
    value=167.0,  # a real number, borrowed from the venue average
    n=38,
    source="N8.M2",
)

TEMPLATE = (
    "Against pace in the powerplay, Kohli's cover drive returns {c1}, "
    "and he is strong on it: {c2}. "
    "Against his pull, the cover drive shows no detectable difference "
    "on this evidence: {c3}. "
    "Six short balls carry {c4} of a dismissal, about as likely as not, "
    "though the estimate is very uncertain."
)


def correct_finding() -> Finding:
    return Finding(
        agent="N11b",
        claims=(STRIKE_RATE, VS_AVERAGE, VS_PULL, CHANCE),
        template=TEMPLATE,
    )


# Broken versions


@dataclass(frozen=True, kw_only=True)
class Case:
    """One version of the finding, and which checks of the full gate should stop it."""

    name: str  # "F2b"
    lie: str  # "right value, wrong subject: the prose"
    finding: Finding
    expected: tuple[str, ...]  # () for a control, which every gate should pass


def cases() -> list[Case]:
    good = correct_finding()
    found = [
        Case(
            name="control",
            lie="none: the correct finding",
            finding=good,
            expected=(),
        ),
        Case(
            name="control-small",
            lie="none: a small sample, with its warning",
            finding=_add(
                good,
                SMALL_SAMPLE,
                "Against spin in the powerplay, on only a few balls, he struggles: {c5}.",
            ),
            expected=(),
        ),
        Case(
            name="F1",
            lie="wrong value: 162.0 instead of 142.0",
            finding=_swap_claim(good, dataclasses.replace(STRIKE_RATE, value=162.0)),
            expected=("C2",),
        ),
        Case(
            name="F2a",
            lie="right value, wrong subject: the claim says Rohit",
            finding=_swap_claim(good, dataclasses.replace(STRIKE_RATE, subject=ROHIT)),
            expected=("C2",),
        ),
        Case(
            name="F2b",
            lie="right value, wrong subject: the sentence says Rohit",
            finding=_swap_words(good, "Kohli's cover drive", "Rohit's cover drive"),
            expected=("C6",),
        ),
        Case(
            name="F3",
            lie="uncited number: 'close to 150' typed in the sentence",
            finding=_swap_words(good, "returns {c1},", "returns {c1}, close to 150,"),
            expected=("C5",),
        ),
        Case(
            name="F4",
            lie="certainty inflation: 'will likely be dismissed' at 41%",
            finding=_swap_words(
                good, "about as likely as not", "so he will likely be dismissed"
            ),
            expected=("C4",),
        ),
        Case(
            name="F5",
            lie="small-sample generalisation: 'struggles' on six balls",
            finding=_add(
                good,
                SMALL_SAMPLE,
                "Against spin in the powerplay, he struggles: {c5}.",
            ),
            expected=("C3",),
        ),
        Case(
            name="F6",
            lie="spurious difference: 'the stronger stroke', range includes zero",
            finding=_swap_words(
                good,
                "shows no detectable difference on this evidence",
                "is the stronger stroke",
            ),
            expected=("C4", "C7"),
        ),
        Case(
            name="F7",
            lie="unsupported claim type: the surface is holding together",
            finding=_add(good, HOLLOW, "The surface is holding together: {c5}."),
            expected=("C1",),
        ),
        Case(
            name="F8",
            lie="unsupported tone: 'struggles' beside +22 against the average",
            finding=_swap_words(
                good, "and he is strong on it", "but he struggles with it"
            ),
            expected=("C7",),
        ),
    ]
    return found


def _swap_claim(finding: Finding, claim: Claim) -> Finding:
    # put a changed claim in place of the one with the same id
    claims = []
    for old in finding.claims:
        if old.id == claim.id:
            claims.append(claim)
        else:
            claims.append(old)
    if tuple(claims) == finding.claims:
        raise HarnessError(
            f"{claim.id}: the broken claim is the same as the correct one"
        )
    return dataclasses.replace(finding, claims=tuple(claims))


def _swap_words(finding: Finding, old: str, new: str) -> Finding:
    # change the sentence in exactly one place, or refuse
    if finding.template.count(old) != 1:
        raise HarnessError(f"'{old}' must appear exactly once in the template")
    return dataclasses.replace(finding, template=finding.template.replace(old, new))


def _add(finding: Finding, claim: Claim, sentence: str) -> Finding:
    return Finding(
        agent=finding.agent,
        claims=finding.claims + (claim,),
        template=finding.template + " " + sentence,
    )


# Running every gate on every case

GATES = (
    ("v2 mode", V2_MODE),
    ("v3 design", V3_DESIGN_CHECKS),
    ("v4 design", ALL_CHECKS),
)


@dataclass(frozen=True, kw_only=True)
class Result:
    """Which checks stopped one case, in each gate."""

    case: Case
    fired: dict[str, tuple[str, ...]]  # gate -> checks that failed it


def run() -> list[Result]:
    source = engines()
    setup = context(source)
    results = []
    for case in cases():
        fired = {}
        for gate, checks in GATES:
            failures = faith_check(case.finding, source, setup, checks)
            fired[gate] = _checks_in(failures)
        results.append(Result(case=case, fired=fired))
    return results


def _checks_in(failures: list[Failure]) -> tuple[str, ...]:
    names = []
    for failure in failures:
        if failure.check not in names:
            names.append(failure.check)
    return tuple(names)


def classes_caught(results: list[Result], gate: str) -> list[str]:
    """The lie classes this gate catches. F2 counts only if both versions are caught."""
    stopped = {}
    for result in results:
        if result.case.expected == ():
            continue  # a control
        lie_class = result.case.name[:2]  # "F2a" -> "F2"
        caught = result.fired[gate] != ()
        if lie_class not in stopped:
            stopped[lie_class] = caught
        else:
            stopped[lie_class] = stopped[lie_class] and caught
    found = []
    for lie_class in sorted(stopped):
        if stopped[lie_class]:
            found.append(lie_class)
    return found


def controls_passed(results: list[Result], gate: str) -> int:
    """How many correct findings this gate let through (each one stopped is a false alarm)."""
    passed = 0
    for result in results:
        if result.case.expected == () and result.fired[gate] == ():
            passed += 1
    return passed
