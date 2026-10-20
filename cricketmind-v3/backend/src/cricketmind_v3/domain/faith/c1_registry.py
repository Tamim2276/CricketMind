from cricketmind_v3.domain.faith.gate import Context, Failure
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.registry import lookup, missing_keys


def check_c1(finding: Finding, context: Context) -> list[Failure]:
    """C1: is each claim on the menu, allowed for this agent, complete, from its engine?"""
    allowed = context.allowed.get(finding.agent)
    if allowed is None:
        return [_fail(finding, "", f"{finding.agent} is not an agent on the roster")]

    failures = []
    for claim in finding.claims:
        claim_type = lookup(claim.metric)
        if claim_type is None:
            reason = f"'{claim.metric}' is not on the menu: no engine computes it"
            failures.append(_fail(finding, claim.id, reason))
            continue  # nothing else to compare it with

        if claim.metric not in allowed:
            reason = f"{finding.agent} may not assert {claim.metric}"
            failures.append(_fail(finding, claim.id, reason))

        if claim.source is None:
            reason = f"the claim names no source; {claim.metric} comes from {claim_type.source}"
            failures.append(_fail(finding, claim.id, reason))
        elif claim.source != claim_type.source:
            reason = (
                f"the claim says it comes from {claim.source}, "
                f"but {claim.metric} comes from {claim_type.source}"
            )
            failures.append(_fail(finding, claim.id, reason))

        if claim.kind != claim_type.provenance:
            reason = (
                f"{claim.metric} is {claim_type.provenance.value}, "
                f"but the claim says {claim.kind.value}"
            )
            failures.append(_fail(finding, claim.id, reason))

        missing = missing_keys(claim_type, claim.subject)
        if missing:
            reason = f"{claim.metric} needs to know: " + ", ".join(missing)
            failures.append(_fail(finding, claim.id, reason))

        if claim_type.needs_range and claim.interval is None:
            reason = (
                f"{claim.metric} must carry its range, so its uncertainty shows (C4)"
            )
            failures.append(_fail(finding, claim.id, reason))

    return failures


def _fail(finding: Finding, claim_id: str, reason: str) -> Failure:
    return Failure(agent=finding.agent, claim=claim_id, check="C1", reason=reason)
