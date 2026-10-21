from cricketmind_v3.domain.faith.gate import Context, Failure
from cricketmind_v3.domain.faith.text import has_phrase, sentence_of
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.registry import lookup
from cricketmind_v3.domain.render import counted

# the smallest sample a count may stand on without a warning (design parameters, D-030)
FLOORS = {"balls": 30, "innings": 10, "reviews": 20, "wickets": 30}

# ways to warn the reader; no digits, so they pass C5
QUALIFIERS = (
    "on only",
    "only a few",
    "a small sample",
    "a tiny sample",
    "a limited sample",
    "limited evidence",
    "too few",
    "too early to say",
)


def check_c3(finding: Finding, context: Context) -> list[Failure]:
    """C3: a count on a small sample must say so in its own sentence."""
    failures = []
    for claim in finding.claims:
        if claim.kind != Provenance.DESCRIPTIVE:
            continue  # a prediction shows its range, a measurement its clip
        claim_type = lookup(claim.metric)
        if claim_type is None:
            continue  # C1 reports it
        floor = FLOORS[claim_type.unit]
        if claim.n >= floor:
            continue
        sentence = sentence_of(finding.template, claim.id)
        if sentence is None or has_phrase(sentence, QUALIFIERS):
            continue
        reason = (
            f"the number rests on only {counted(claim.n, claim_type.unit)} "
            f"(the floor is {floor}); say so, e.g. 'on only a few {claim_type.unit}'"
        )
        failures.append(
            Failure(agent=finding.agent, claim=claim.id, check="C3", reason=reason)
        )
    return failures
