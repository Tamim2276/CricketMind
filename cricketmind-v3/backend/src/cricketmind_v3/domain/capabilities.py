"""What can this question support? (Day 62)

An agent can answer only if the question has a number for at least one of its measures.
N10 chooses among these agents (Day 63).
"""

from cricketmind_v3.domain.facts import Answer
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.registry import lookup
from cricketmind_v3.domain.roster import ROSTER
from cricketmind_v3.domain.shot_dna import ShotDNA

EXPLAINER = "N11m"


def measures_with_numbers(answers: list[Answer], dna: ShotDNA) -> set[str]:
    """What this question has a number for: what the engines answered, and a posture
    only when it was measured from a clip (I5). Having a clip is not enough."""
    found = set()
    for answer in answers:
        if answer.fact is not None:
            found.add(answer.metric)
    if dna.clip is not None:
        for name, _ in dna.posture:
            row = lookup(name)
            if row is not None and row.provenance == Provenance.MEASURED:
                found.add(name)
    return found


def capable_agents(
    kind: str, answers: list[Answer], dna: ShotDNA
) -> dict[str, tuple[str, ...]]:
    """Each agent that can answer, with the measures it has numbers for. A rule goes to
    the Explainer alone, which uses no numbers. Nobody capable gives an empty dict."""
    if kind == "rules":
        return {EXPLAINER: ()}
    ready = measures_with_numbers(answers, dna)
    capable = {}
    for agent, names in ROSTER.items():
        has = []
        for name in names:
            if name in ready:
                has.append(name)
        if has:
            capable[agent] = tuple(has)
    return capable
