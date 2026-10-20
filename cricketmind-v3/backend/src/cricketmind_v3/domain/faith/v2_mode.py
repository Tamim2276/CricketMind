from cricketmind_v3.domain.faith.gate import Context, Failure
from cricketmind_v3.domain.findings import Finding


def check_v2(finding: Finding, context: Context) -> list[Failure]:
    """The v2 (ICCIT) gate, kept to compare with: is each value one the engines gave?

    Like v2, it checks values only, by exact equality. It never looks at who a number
    is about, its sample, its range or the words around it.
    """
    failures = []
    for claim in finding.claims:
        if claim.value not in context.surface:
            reason = f"{claim.value} is not a number the engines gave"
            failures.append(
                Failure(agent=finding.agent, claim=claim.id, check="V2", reason=reason)
            )
    return failures
