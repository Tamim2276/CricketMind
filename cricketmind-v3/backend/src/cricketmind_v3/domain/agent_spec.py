"""What an agent is (Day 65): who it works for, how it sounds, and what it may say.

The voice comes from the agent's prompt file. What it may say comes only from the roster,
so a prompt file can never widen it, and C1 checks every claim against that same list.
"""

from dataclasses import dataclass

from cricketmind_v3.domain.render import METRICS
from cricketmind_v3.domain.roster import NAMES, ROSTER, STAKEHOLDERS

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
    measures it may use that have numbers for this question (`ready`)."""
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
