"""The agent factory (Day 66): a model's reply becomes a Finding.

The model fills one box: the sentence, with a slot where each number goes. It never types
a number, a measure or a subject. The code makes the claims from what the engines said,
and keeps the ones the sentence uses. A reply that cannot be read still becomes a Finding,
with the reason, so that C1 reports it and the gate can ask again (Day 74).
"""

from pydantic import BaseModel, ConfigDict, Field

from cricketmind_v3.application.ports import LLM, LLMError
from cricketmind_v3.domain.agent_spec import (
    AgentSpec,
    agent_prompt,
    offered_claims,
    system_prompt,
)
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.facts import Answer
from cricketmind_v3.domain.faith.gate import Failure
from cricketmind_v3.domain.findings import Finding, FindingError, references
from cricketmind_v3.domain.premise import Premise


class FindingForm(BaseModel):
    """The form the model fills: only the words."""

    model_config = ConfigDict(extra="forbid")

    template: str = Field(
        description="your few sentences; where a number you were given goes, its slot, "
        "as in {c1}"
    )


def write_finding(
    agent: str,
    question: str,
    answers: list[Answer],
    llm: LLM | None,
    spec: AgentSpec | None,
    rules: str,
    failures: tuple[Failure, ...] = (),
    shown: dict[str, str] | None = None,
    premise: Premise | None = None,
) -> Finding:
    """One try at one agent's finding: one call to the model, so the bound is simple."""
    if llm is None:
        return _unreadable(agent, "no language model is running")
    if spec is None:
        return _unreadable(agent, "this agent has no prompt file")
    claims = offered_claims(agent, answers)
    if spec.allowed and not claims:
        return _unreadable(agent, "this agent has no number for this question")
    ready = []
    for claim in claims:
        ready.append(claim.metric)
    try:
        form = llm.answer(
            system_prompt(spec, rules, tuple(ready)),
            agent_prompt(question, claims, failures, shown, premise),
            FindingForm,
        )
    except LLMError as error:
        reason = str(error).splitlines()[0]
        return _unreadable(agent, f"the model's reply did not fit: {reason}")
    return finding_from_reply(agent, form.template, claims, bool(spec.allowed))


def finding_from_reply(
    agent: str, template: str, offered: tuple[Claim, ...], needs_a_number: bool = True
) -> Finding:
    """The claims the sentence uses, and the sentence. Anything else is unreadable."""
    if not template.strip():
        return _unreadable(agent, "the reply has no sentence")
    try:
        slots = references(template)
    except FindingError as error:
        return _unreadable(agent, str(error))
    if needs_a_number and not slots:
        return _unreadable(
            agent, "the sentence uses no slot: say at least one of your numbers"
        )
    used = []
    for claim in offered:
        if claim.id in slots:
            used.append(claim)
    try:
        return Finding(agent=agent, claims=tuple(used), template=template)
    except FindingError as error:
        return _unreadable(agent, str(error))  # a slot it was not offered


def _unreadable(agent: str, why: str) -> Finding:
    return Finding(agent=agent, claims=(), template="", unreadable=why)
