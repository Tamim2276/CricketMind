"""N12's report, and what it is built from (Day 75).

The verified findings in a fixed order are always possible, and are what the report falls
back to. For a model to join them, their claims are put under one set of slots, so two
agents' c1 cannot be mixed up, and one number said twice is one slot.
"""

from dataclasses import dataclass

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.render import render
from cricketmind_v3.domain.roster import NAMES, ROSTER

REPORT_AGENT = "N12"
# always in front of an explanation (D-012, Day 72)
EXPLANATION_LABEL = "General explanation, not checked against data"

# how the report was made
NOTHING = "nothing passed"
ONE_FINDING = "one finding"
FIXED_ORDER = "fixed order"
BY_MODEL = "model"


@dataclass(frozen=True, kw_only=True)
class Report:
    text: str  # with every number filled in, by the renderer
    by: str  # NOTHING, ONE_FINDING, FIXED_ORDER or BY_MODEL
    note: str | None = None  # why the fixed order was used when a model was there


def in_routing_order(
    findings: tuple[Finding, ...], chosen: tuple[str, ...]
) -> list[Finding]:
    """The router's order, whichever agent finished first."""
    ordered = []
    for agent in chosen:
        for finding in findings:
            if finding.agent == agent:
                ordered.append(finding)
    return ordered


def one_finding(finding: Finding) -> str:
    """A finding with its numbers filled in, and whose it is. An explanation always
    carries its label."""
    if not ROSTER[finding.agent]:
        return f"{EXPLANATION_LABEL}: {render(finding)}"
    return f"{NAMES[finding.agent]}: {render(finding)}"


def fixed_order(findings: list[Finding]) -> str:
    lines = []
    for finding in findings:
        lines.append(one_finding(finding))
    return "\n".join(lines)


def joined_findings(findings: list[Finding]) -> Finding:
    """Every finding under one set of slots, for the model to join: "Tactical Analysis:
    Kohli returns {c1}." on one line for each. One number said twice is one slot."""
    claims = []
    lines = []
    for finding in findings:
        new_ids = {}
        for claim in finding.claims:
            new_ids[claim.id] = _slot_for(claim, claims)
        lines.append(
            f"{NAMES[finding.agent]}: {_with_slots(finding.template, new_ids)}"
        )
    return Finding(agent=REPORT_AGENT, claims=tuple(claims), template="\n".join(lines))


def _slot_for(claim: Claim, claims: list[Claim]) -> str:
    # the same measure of the same subject has the same number: one slot
    for kept in claims:
        if kept.metric == claim.metric and kept.subject == claim.subject:
            return kept.id
    new = _renamed(claim, f"c{len(claims) + 1}")
    claims.append(new)
    return new.id


def _renamed(claim: Claim, claim_id: str) -> Claim:
    return Claim(
        id=claim_id,
        kind=claim.kind,
        metric=claim.metric,
        subject=claim.subject,
        value=claim.value,
        n=claim.n,
        interval=claim.interval,
        clip_id=claim.clip_id,
        source=claim.source,
    )


def _with_slots(template: str, new_ids: dict[str, str]) -> str:
    # one pass from left to right, so {c1} -> {c2} cannot then become {c3}
    written = ""
    start = 0
    while True:
        open_at = template.find("{", start)
        if open_at == -1:
            return written + template[start:]
        close_at = template.find("}", open_at)
        slot = template[open_at + 1 : close_at]
        written += template[start:open_at] + "{" + new_ids[slot] + "}"
        start = close_at + 1
