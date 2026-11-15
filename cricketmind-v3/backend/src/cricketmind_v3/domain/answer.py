"""What N13 shows (Day 75): the read-back first, then what passed, with its tags; what was
held back; what was missing; and a receipt for every number.

Nothing here makes a number. A number reaches the reader only from a verified finding,
through the renderer. A withheld finding's reasons stay out, as one can quote a number
nobody checked ("it typed '150'").
"""

from dataclasses import dataclass

from cricketmind_v3.domain.faith.gate import Withheld
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.premise import PREMISE_AGENT
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.readback import cant_check_answer, read_back
from cricketmind_v3.domain.reading import Reading
from cricketmind_v3.domain.receipts import Receipt
from cricketmind_v3.domain.registry import extra_keys, lookup, missing_keys
from cricketmind_v3.domain.render import render
from cricketmind_v3.domain.report import Report
from cricketmind_v3.domain.roster import NAMES, ROSTER
from cricketmind_v3.domain.routing import NOBODY, Routing
from cricketmind_v3.domain.subject import CricketSubject

NOBODY_ANSWERS = "Nothing is answered: no agent has a checked number for this question."
NOTHING_PASSED = "Nothing is shown: no finding passed the checks."

# how a subject's fields are asked for
FIELD_WORDS = {
    "player": "a batter",
    "bowler": "a bowler",
    "other_bowler": "a second bowler",
    "shot": "a shot",
    "other_shot": "a second shot",
    "team": "a team",
    "opposition": "the other side",
    "venue": "a ground",
    "match_format": "a format",
    "gender": "men's or women's cricket",
    "date": "a date",
    "k": "how many balls ahead",
    "bowler_type": "pace or spin",
}


@dataclass(frozen=True, kw_only=True)
class Parts:
    """Everything N13 may show. A question nobody can check has only the reading."""

    reading: Reading
    routing: Routing | None = None
    report: Report | None = None
    withheld: tuple[Withheld, ...] = ()
    verified: tuple[Finding, ...] = ()
    receipts: dict | None = None
    why_no_numbers: tuple[str, ...] = ()  # the evidence's reasons, and the clip's


def final_answer(parts: Parts) -> str:
    lines = [read_back(parts.reading)]
    if parts.reading.kind == "cant_check":
        lines.append(cant_check_answer(parts.reading))
        return "\n".join(lines)
    if parts.routing is None:
        return "\n".join(lines)  # a graph stopped early: only what was read is known

    lines.extend(_the_judgment(parts))
    if parts.routing.how == NOBODY:
        lines.append(NOBODY_ANSWERS)
        for why in parts.why_no_numbers:
            lines.append(f"Why: {why}.")
    elif parts.report is not None and parts.report.text:
        lines.append(parts.report.text)
    else:
        lines.append(NOTHING_PASSED)

    lines.extend(_what_was_missing(parts.routing, parts.reading.subject))
    lines.extend(_held_back(parts.withheld))
    lines.extend(receipt_lines(parts.verified, parts.receipts or {}))
    return "\n".join(lines)


def _the_judgment(parts: Parts) -> list[str]:
    # "says so before anything else" (Day 76); a rule's question judges no player
    premise = parts.reading.premise
    if premise is None or parts.reading.kind == "rules":
        return []
    for finding in parts.verified:
        if finding.agent == PREMISE_AGENT:
            return [f'On "{premise.word}": {render(finding)}']
    return [f'On "{premise.word}": nothing I can count tests that.']


def _what_was_missing(routing: Routing, subject: CricketSubject) -> list[str]:
    # who the question was for, and what it would have taken (D-084, L-078)
    lines = []
    for agent in routing.no_number:
        lines.append(what_it_needs(agent, subject))
    return lines


def what_it_needs(agent: str, subject: CricketSubject) -> str:
    """ "Injury Management could answer with a date, and without a format."

    From the menu, so it holds for every measure: of this agent's measures, the one
    nearest to fitting the question says what is missing or in the way.
    """
    best = None
    for name in ROSTER[agent]:
        row = lookup(name)
        if row.provenance == Provenance.MEASURED:
            continue  # needs a clip, not a word in the question
        missing = missing_keys(row, subject)
        extra = extra_keys(row, subject)
        size = len(missing) + len(extra)
        if best is None or size < best[0]:
            best = (size, missing, extra)
    if best is None or best[0] == 0:
        return f"{NAMES[agent]} has no number for this in the data."
    parts = []
    if best[1]:
        parts.append("with " + _in_words(best[1]))
    if best[2]:
        parts.append("without " + _in_words(best[2]))
    return f"{NAMES[agent]} could answer {', and '.join(parts)}."


def _in_words(fields: list[str]) -> str:
    words = []
    for field in fields:
        words.append(FIELD_WORDS.get(field, field))
    return ", ".join(words)


def _held_back(withheld: tuple[Withheld, ...]) -> list[str]:
    lines = []
    for each in withheld:
        if each.agent not in NAMES:
            continue  # N3's test of a judgment has its own line
        checks = ", ".join(each.checks)
        lines.append(f"Withheld: {NAMES[each.agent]}: its words did not pass {checks}.")
    return lines


def receipt_lines(findings: tuple[Finding, ...], receipts: dict) -> list[str]:
    """How to make each number shown again (D-052). One number said twice, one receipt."""
    seen = []
    lines = []
    for finding in findings:
        for claim in finding.claims:
            receipt = receipts.get((finding.agent, claim.id))
            if receipt is None or receipt in seen:
                continue
            seen.append(receipt)
            lines.append(_receipt_line(receipt))
    if lines:
        lines.insert(0, "Receipts:")
    return lines


def _receipt_line(receipt: Receipt) -> str:
    return (
        f"- {receipt.engine} version {receipt.version}, data {receipt.data[:12]}: "
        f"{receipt.query}"
    )
