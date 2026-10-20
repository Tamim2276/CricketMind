from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.faith.gate import Context, Failure
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.registry import lookup
from cricketmind_v3.domain.render import shown


def check_c2(finding: Finding, context: Context) -> list[Failure]:
    """C2: asked again, for exactly this subject, does the engine give the same number?"""
    failures = []
    for claim in finding.claims:
        if lookup(claim.metric) is None:
            continue  # C1 reports it; there is no engine to ask

        # fail closed: a number nobody re-checked is not shown
        if claim.id not in context.facts:
            reason = "the number was never re-checked with its engine"
            failures.append(_fail(finding, claim.id, reason))
            continue
        fact = context.facts[claim.id]
        if fact is None:
            reason = "the engine has no number for this subject"
            failures.append(_fail(finding, claim.id, reason))
            continue

        # "within rounding" means: the reader would see the same number (D-028)
        said = shown(claim.metric, claim.value)
        real = shown(claim.metric, fact.value)
        if said != real:
            reason = (
                f"the claim says {said}, but the engine gives {real} for this subject"
            )
            failures.append(_fail(finding, claim.id, reason))

        # a bigger sample would dodge the evidence floor (C3)
        if claim.kind == Provenance.DESCRIPTIVE and claim.n != fact.n:
            reason = f"the claim rests on {claim.n}, but the engine counts {fact.n}"
            failures.append(_fail(finding, claim.id, reason))

        failures.extend(_compare_ranges(finding, claim, fact))
    return failures


def _compare_ranges(finding: Finding, claim: Claim, fact: Fact) -> list[Failure]:
    if claim.interval is None and fact.interval is None:
        return []
    if claim.interval is None:
        return [_fail(finding, claim.id, "the claim leaves out the engine's range")]
    if fact.interval is None:
        reason = "the claim gives a range the engine does not have"
        return [_fail(finding, claim.id, reason)]
    said = _range_text(claim.metric, claim.interval)
    real = _range_text(claim.metric, fact.interval)
    if said != real:
        reason = f"the claim's range is {said}, but the engine's is {real}"
        return [_fail(finding, claim.id, reason)]
    return []


def _range_text(metric: str, interval: tuple[float, float]) -> str:
    low, high = interval
    return shown(metric, low) + " to " + shown(metric, high)


def _fail(finding: Finding, claim_id: str, reason: str) -> Failure:
    return Failure(agent=finding.agent, claim=claim_id, check="C2", reason=reason)
