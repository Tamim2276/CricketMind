"""CricketMind: a verification-gated multi-agent architecture for cricket analytics.

A query -- with a batting-shot clip or without one -- is turned into a
stakeholder-specific report in which every figure has been checked against the
two engines permitted to produce numbers.

Typical use:

    from cricketmind import Settings, build_cricketmind_graph, new_state

    graph = build_cricketmind_graph()
    final = graph.invoke(new_state(CLIP, QUERY, Settings(trace=True)))
    print(final["report"])

Everything about a run that is not the query lives on `Settings` and travels
in the state, so two runs cannot interfere with each other.
"""
from .agents import (AGENT_NODES, AGENT_REGISTRY, AGENTS_BY_ID, POSTURE_AGENTS,
                     agent_name, sanctioned_facts)
from .config import (CLIP, CONF_THRESHOLD, DEFAULT_AGENTS, DEFAULT_SETTINGS,
                     MAX_ACTIVE_AGENTS, MAX_REGENS, MAX_RETRIES, NUM_CLASSES,
                     NUM_FRAMES, PROJECT_ROOT, QUERY, SEED, SHOT_CLASSES,
                     SIM_DELIVERIES, SIM_TRIALS, Settings)
from .graph import build_cricketmind_graph, show_graph
from .state import (CricketMindState, FakeTensor, describe_state, new_state,
                    settings_of, trace, upsert_findings)

__all__ = [
    "AGENT_NODES", "AGENT_REGISTRY", "AGENTS_BY_ID", "POSTURE_AGENTS",
    "agent_name", "sanctioned_facts",
    "CLIP", "CONF_THRESHOLD", "DEFAULT_AGENTS", "DEFAULT_SETTINGS",
    "MAX_ACTIVE_AGENTS", "MAX_REGENS", "MAX_RETRIES", "NUM_CLASSES",
    "NUM_FRAMES", "PROJECT_ROOT", "QUERY", "SEED", "SHOT_CLASSES",
    "SIM_DELIVERIES", "SIM_TRIALS", "Settings",
    "build_cricketmind_graph", "show_graph",
    "CricketMindState", "FakeTensor", "describe_state", "new_state",
    "settings_of", "trace", "upsert_findings",
]
