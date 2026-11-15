"""What one question can cost at worst (Day 79, invariant I6): each loop is bounded, so the
worst case is a sum anyone can check. A test runs that worst case with fakes and counts.

A superstep is one round of the graph: nodes running side by side count as one. The bound
for a clip is written with Phase 8, when the video nodes and their retry loop are real.
"""

from cricketmind_v3.domain.faith.gate import MAX_REGENERATIONS
from cricketmind_v3.domain.routing import MAX_ACTIVE_AGENTS

# N3, N6, N7a and N7b, N8 and N9, N10: before any agent
BEFORE_THE_AGENTS = 5
# the agents and the gate: a round each, once and again for every regeneration
AGENTS_AND_GATE = 2
# N12 and N13
AFTER_THE_GATE = 2

READER_CALLS = 2  # a reply that does not fit is asked again once (Day 58)
NAMES_IN_A_QUESTION = 3  # player, bowler, other bowler
LOOK_UPS_FOR_A_NAME = 3  # D-055: resolver.py's LOOK_UPS
ROUTER_CALLS = 2  # the same (Day 64)
REPORT_CALLS = 1  # one try (Day 75)


def agent_runs(
    agents: int = MAX_ACTIVE_AGENTS, regenerations: int = MAX_REGENERATIONS
) -> int:
    """4 x (1 + 2) = 12, however many agents the roster has. Each run asks the model once."""
    return agents * (1 + regenerations)


def text_supersteps(regenerations: int = MAX_REGENERATIONS) -> int:
    """5 + 2 x (1 + 2) + 2 = 13 rounds for a question in words."""
    return BEFORE_THE_AGENTS + AGENTS_AND_GATE * (1 + regenerations) + AFTER_THE_GATE


def model_calls(
    agents: int = MAX_ACTIVE_AGENTS, regenerations: int = MAX_REGENERATIONS
) -> int:
    """2 + 3 x 3 + 2 + 12 + 1 = 26 calls to the language model at most."""
    return (
        READER_CALLS
        + NAMES_IN_A_QUESTION * LOOK_UPS_FOR_A_NAME
        + ROUTER_CALLS
        + agent_runs(agents, regenerations)
        + REPORT_CALLS
    )
