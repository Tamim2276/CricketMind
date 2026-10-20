import dataclasses

from cricketmind_v3.application.ports import FactSource
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.faith.checks import ALL_CHECKS
from cricketmind_v3.domain.faith.gate import Check, Context, Failure, run_checks
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.registry import lookup


def recompute(finding: Finding, source: FactSource) -> dict[str, Fact | None]:
    """Ask the engine again for every claim on the menu: claim id -> its answer."""
    facts = {}
    for claim in finding.claims:
        if lookup(claim.metric) is None:
            continue  # C1 reports it; no engine to ask
        # the gate asks with the claim's metric and subject, never its value
        facts[claim.id] = source.fact(claim.metric, claim.subject, claim.clip_id)
    return facts


def faith_check(
    finding: Finding,
    source: FactSource,
    context: Context,
    checks: tuple[Check, ...] = ALL_CHECKS,
) -> list[Failure]:
    """The Faith Check node: ask the engines again, then run the checks."""
    checked = dataclasses.replace(context, facts=recompute(finding, source))
    return run_checks(finding, checked, checks)
