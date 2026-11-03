"""The agents' prompt files (Day 65): one for each in backend/prompts/, and the rules they
all share. A file that is missing or doubled stops the program at the start, not halfway
through a question."""

from pathlib import Path

from cricketmind_v3.domain.agent_spec import AgentSpec, make_spec
from cricketmind_v3.domain.capabilities import EXPLAINER
from cricketmind_v3.domain.roster import ROSTER

PROMPTS = Path(__file__).resolve().parents[3] / "prompts"
RULES_FILE = "shared_rules.txt"


class PromptFileError(ValueError):
    """An agent's prompt file is missing, doubled or empty."""


def read_rules(folder: Path = PROMPTS) -> str:
    return _read(folder / RULES_FILE)


def read_specs(folder: Path = PROMPTS) -> dict[str, AgentSpec]:
    """Every stakeholder agent's spec. The Explainer's file comes with it (Day 72)."""
    specs = {}
    for agent in ROSTER:
        if agent == EXPLAINER:
            continue
        found = list(folder.glob(f"{agent.lower()}_*.txt"))
        if len(found) != 1:
            raise PromptFileError(f"{agent}: {len(found)} prompt files in {folder}")
        specs[agent] = make_spec(agent, _read(found[0]))
    return specs


def _read(path: Path) -> str:
    if not path.exists():
        raise PromptFileError(f"no prompt file: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise PromptFileError(f"an empty prompt file: {path}")
    return text
