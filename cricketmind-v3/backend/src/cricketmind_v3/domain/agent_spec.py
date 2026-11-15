"""What an agent is (Day 65): who it works for, how it sounds, and what it may say.

The voice comes from the agent's prompt file. What it may say comes only from the roster,
so a prompt file can never widen it, and C1 checks every claim against that same list.

What it is told about a question (Day 66): the numbers it may use, each ready as a claim
with its slot. The model writes only the sentence; the code fills in every number.
"""

from dataclasses import dataclass

from cricketmind_v3.domain.claims import Claim, ClaimError
from cricketmind_v3.domain.facts import Answer
from cricketmind_v3.domain.faith.c3_evidence import FLOORS
from cricketmind_v3.domain.faith.gate import Failure
from cricketmind_v3.domain.premise import Premise
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.readback import subject_words
from cricketmind_v3.domain.receipts import query_text
from cricketmind_v3.domain.registry import lookup
from cricketmind_v3.domain.render import METRICS, render_claim
from cricketmind_v3.domain.roster import NAMES, ROSTER, STAKEHOLDERS
from cricketmind_v3.domain.wording import WordingError, band_for, needs_hedge

# the four parts of every voice, in this order
VOICE_PARTS = (
    "Who you work for:",
    "What they want:",
    "How you sound:",
    "What you never say:",
)


@dataclass(frozen=True, kw_only=True)
class AgentSpec:
    agent: str  # "N11b"
    name: str  # "Tactical Analysis"
    stakeholder: str  # "Coaches"
    voice: str  # the words of its prompt file
    allowed: tuple[str, ...]  # the roster's measures, never the file's


def make_spec(agent: str, voice: str) -> AgentSpec:
    return AgentSpec(
        agent=agent,
        name=NAMES[agent],
        stakeholder=STAKEHOLDERS[agent],
        voice=voice.strip(),
        allowed=ROSTER[agent],
    )


def system_prompt(spec: AgentSpec, rules: str, ready: tuple[str, ...]) -> str:
    """All an agent is told before the question: its voice, the shared rules, and the
    measures it may use that have numbers for this question (`ready`). An agent with no
    measures (the Explainer) writes words only, so the rules about slots are not for it."""
    if not spec.allowed:
        return f"You are {spec.name}, a cricket specialist.\n{spec.voice}"
    lines = [
        f"You are {spec.name}, a cricket specialist.",
        spec.voice,
        "",
        rules.strip(),
        "",
        "Your measures for this question, with what each slot will say:",
    ]
    for name in spec.allowed:
        if name in ready:
            words = METRICS[name][0].replace("{}", "...")
            lines.append(f"- {name}: {words}")
    return "\n".join(lines)


def offered_claims(agent: str, answers: list[Answer]) -> tuple[Claim, ...]:
    """Every number this agent may use for this question, each ready as a claim: the same
    question always gives the same c1, c2... whichever engine finished first."""
    offered = []
    for name in ROSTER[agent]:
        same_measure = []
        for answer in answers:
            if answer.metric == name and answer.fact is not None:
                same_measure.append(answer)
        same_measure.sort(key=_asked)
        for answer in same_measure:
            claim = claim_from(f"c{len(offered) + 1}", answer)
            if claim is not None:
                offered.append(claim)
    return tuple(offered)


def _asked(answer: Answer) -> str:
    return query_text(answer.metric, answer.subject, None)


def claim_from(claim_id: str, answer: Answer) -> Claim | None:
    """A claim made from what an engine said. The model never fills these fields."""
    row = lookup(answer.metric)
    try:
        return Claim(
            id=claim_id,
            kind=row.provenance,
            metric=answer.metric,
            subject=answer.subject,
            value=answer.fact.value,
            n=answer.fact.n,
            interval=answer.fact.interval,
            source=row.source,
        )
    except ClaimError:
        return None  # a count with no sample, or a chance with no range, cannot be said


def fact_line(claim: Claim, shown: dict[str, str] | None = None) -> str:
    """One number as the agent sees it, and what the checks will ask of its sentence."""
    about = ", ".join(subject_words(claim.subject, shown))
    line = f"{{{claim.id}}} shows: {render_claim(claim)}. It is about: {about}."
    for need in _what_it_needs(claim):
        line += " " + need
    return line


def _what_it_needs(claim: Claim) -> list[str]:
    # told here from the same numbers the checks read, so a sentence can pass first time
    needs = []
    row = lookup(claim.metric)
    counted = claim.kind == Provenance.DESCRIPTIVE and row.sample
    if counted and claim.n < FLOORS[row.unit]:
        needs.append("A small sample: say so in the same sentence.")
    if row.comparison:
        low, high = claim.interval
        if low <= 0 <= high:
            needs.append('Its range includes zero: write "no detectable difference".')
    elif claim.kind == Provenance.PREDICTIVE:
        needs.extend(_chance_words(claim))
    return needs


def _chance_words(claim: Claim) -> list[str]:
    try:
        words = ", ".join(band_for(claim.value).words)
        wide = needs_hedge(claim.interval)
    except WordingError:
        return []  # C4 reports a chance outside 0 to 1
    found = [f"Words that fit this chance: {words}."]
    if wide:
        found.append('Its range is wide: add "the estimate is very uncertain".')
    return found


def agent_prompt(
    question: str,
    claims: tuple[Claim, ...],
    failures: tuple[Failure, ...] = (),
    shown: dict[str, str] | None = None,
    premise: Premise | None = None,
) -> str:
    """The question, the numbers, and, on a second try, why the last answer failed."""
    lines = [f"Question: {question}"]
    if premise is not None:
        lines.append("")
        lines.append(
            f'The question holds a judgment: "{premise.word}". The answer tests it '
            "separately, so do not repeat it, and do not explain why it would be true."
        )
    if claims:
        lines.append("")
        lines.append("The facts you may use:")
        for claim in claims:
            lines.append("- " + fact_line(claim, shown))
    if failures:
        lines.append("")
        lines.append("Your last answer did not pass the checks:")
        for failure in failures:
            lines.append(f"- {failure.check}: {failure.reason}")
        lines.append("Write it again, and fix each of these.")
    return "\n".join(lines)
