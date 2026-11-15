"""Testing a judgment the question takes for granted (Day 76, invariant I7).

The test is the same as C7's: a checked comparison whose whole range lies on the side of
zero the word points to. The sentence is written by code, not a model, and goes through
the gate like any finding, so it gets its receipt. It is filed under N3, which marked it.
"""

from cricketmind_v3.domain.agent_spec import claim_from
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.facts import Answer
from cricketmind_v3.domain.faith.c3_evidence import FLOORS
from cricketmind_v3.domain.faith.c7_tone import SUPERLATIVES
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.premise import PREMISE_AGENT, Premise
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.receipts import query_text
from cricketmind_v3.domain.registry import CLAIM_TYPES, lookup
from cricketmind_v3.domain.roster import ROSTER

# the comparisons that can test a judgment: counted ones, in the menu's order. A matchup
# is a what-if about two bowlers, and says nothing of how a player has done
COMPARISONS = []
for row in CLAIM_TYPES:
    if row.comparison and row.provenance == Provenance.DESCRIPTIVE:
        COMPARISONS.append(row.name)

# which comparison can test which word (D-096). "Bad" is the player against the average
# batter; "better" sets two things side by side, so only a difference between two tests
# it; "best" ranks him against everyone, and nothing here does
AGAINST_AVERAGE = ("sr_vs_average",)
BETWEEN_TWO = ("sr_difference",)
TWO_SIDED_WORDS = ("better", "worse", "stronger", "weaker")


def comparisons_for(word: str) -> tuple[str, ...]:
    if word in SUPERLATIVES:
        return ()
    if word in TWO_SIDED_WORDS:
        return BETWEEN_TWO
    return AGAINST_AVERAGE


# who the gate lets say what: the roster, and N3's one kind of claim
ALLOWED = dict(ROSTER)
ALLOWED[PREMISE_AGENT] = tuple(COMPARISONS)


def premise_finding(premise: Premise, answers: list[Answer]) -> Finding | None:
    """What the numbers say of the judgment, or None when nothing here can test it."""
    answer = _comparison(answers, comparisons_for(premise.word))
    if answer is None:
        return None
    claim = claim_from("c1", answer)
    if claim is None:
        return None
    return Finding(
        agent=PREMISE_AGENT,
        claims=(claim,),
        template=_verdict(premise, claim),
    )


def _comparison(answers: list[Answer], names: tuple[str, ...]) -> Answer | None:
    for name in names:
        found = []
        for answer in answers:
            if answer.metric == name and answer.fact is not None:
                found.append(answer)
        if found:
            found.sort(key=_asked)
            return found[0]
    return None


def _asked(answer: Answer) -> str:
    return query_text(answer.metric, answer.subject, None)


def _verdict(premise: Premise, claim: Claim) -> str:
    # no tone word and no name, so it says only what the comparison says
    row = lookup(claim.metric)
    low, high = claim.interval
    may_be_nothing = low <= 0 <= high
    if claim.n < FLOORS[row.unit]:
        if may_be_nothing:
            return (
                f"Too few {row.unit} to judge that, and no detectable difference "
                "so far: {c1}."
            )
        return f"Too few {row.unit} to judge that: {{c1}}."
    if may_be_nothing:
        return "The numbers do not back that: no detectable difference: {c1}."
    points_up = low > 0
    if points_up == premise.better:
        return "The numbers back that: {c1}."
    return "The numbers point the other way: {c1}."
