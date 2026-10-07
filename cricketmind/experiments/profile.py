"""The execution profile: what each situation costs, and the bounds.

`invoke()` only hands back the final state, so it cannot show the path a query
took. `stream()` runs the same graph but reports events as they happen, and
that is what everything here is built on.

Because settings now travel in the state, a scenario is just a `Settings`
object -- there is no global to save and restore, so running these out of
order cannot change a result.
"""
from functools import lru_cache

from ..agents import AGENT_REGISTRY, AGENTS_BY_ID
from ..config import CLIP, MAX_ACTIVE_AGENTS, MAX_REGENS, MAX_RETRIES, QUERY, Settings
from ..graph import build_cricketmind_graph
from ..state import new_state


@lru_cache(maxsize=1)
def default_graph():
    """One compiled graph, shared. Compiling is pure, so this is safe to cache."""
    return build_cricketmind_graph()


def record_run(state, graph=None) -> list:
    """Run one query and write down which node ran in which step.

    With stream_mode="debug" one kind of event is a "task", which means "this
    node is running in this step" -- that is all we keep.

    Returns a list of (step, node) pairs, for example:
        [(1, "n1_video_input"), (1, "n3_text_encoder"), (2, "n2_vision_encoder"), ...]
    Two pairs with the same step number ran in parallel.
    """
    graph = graph or default_graph()
    timeline = []
    for event in graph.stream(state, stream_mode="debug"):
        if event["type"] == "task":
            timeline.append((event["step"], event["payload"]["name"]))
    return timeline


def run_scenario(clip, confidence, lie_once=(), lie_always=(), query=QUERY,
                 graph=None) -> list:
    """Record one run under one set of conditions.

    clip        a clip path, or None for a text-only query
    confidence  what the classifier scores -- one number for every attempt,
                or a sequence with one per attempt, e.g. (0.40, 0.92)
    lie_once    agent ids that lie on their first try, then correct themselves
    lie_always  agent ids that lie on every try

    No global is touched, so nothing needs putting back afterwards.
    """
    settings = Settings(confidence=confidence, lie_once=lie_once,
                        lie_always=lie_always, trace=False)
    return record_run(new_state(clip, query, settings), graph)


def profile(timeline: list) -> dict:
    """Turn a timeline from record_run into the counts for the cost table.

    steps        how many rounds the graph took (parallel nodes share a round)
    node_runs    how many times any node ran, counting loop re-runs
    agent_calls  how many times a stakeholder agent ran -- an LLM call in deployment
    retries      how many times the vision branch was sent back to N1
    regens       how many times failed agents were sent back to try again
    dropped      True if unverifiable findings were withheld at the end
    degraded     True if the query stopped early because the clip stayed unclear
    """
    ran_nodes = [node for step, node in timeline]
    return {
        "steps":       len({step for step, node in timeline}),
        "node_runs":   len(timeline),
        "agent_calls": sum(1 for node in ran_nodes if node in AGENTS_BY_ID),
        "retries":     ran_nodes.count("retry"),
        "regens":      ran_nodes.count("regen_prep"),
        "dropped":     "drop_unfaithful" in ran_nodes,
        "degraded":    "degrade" in ran_nodes,
    }


def predicted_steps(has_clip: bool, retries: int, regens: int,
                    dropped: bool, degraded: bool) -> int:
    """How many rounds a query should take, worked out from the graph's shape.

    Nothing here runs the graph. Every number is a count of rounds along the
    drawn edges:
      - a query that reaches the report takes 12 rounds with a clip, 9 without
      - each vision retry adds 5 (the retry node and the four-stage chain)
      - each regeneration adds 3 (preparation, the agents, a second check)
      - withholding unverified findings adds 1
      - a degraded run is the first vision pass (4) + 5 per retry + the exit (1)
    """
    if degraded:
        return 4 + 5 * retries + 1

    start = 12 if has_clip else 9
    return start + 5 * retries + 3 * regens + (1 if dropped else 0)


# The seven situations the paper reports.
SCENARIOS = [
    {"name": "Video, confident",
     "clip": CLIP, "confidence": 0.92},

    {"name": "Text-only (no clip)",
     "clip": None, "confidence": 0.92},

    {"name": "Video, one retry then accept",
     "clip": CLIP, "confidence": (0.40, 0.92)},

    {"name": "Video, one agent lies once",
     "clip": CLIP, "confidence": 0.92,
     "lie_once": {"n11b_tactical_analysis"}},

    {"name": "Retries used up (degrade)",
     "clip": CLIP, "confidence": 0.40},

    {"name": "Worst case: 2 retries, agents never fix",
     "clip": CLIP, "confidence": (0.40, 0.40, 0.92),
     "lie_always": {a["id"] for a in AGENT_REGISTRY}},

    {"name": "Text-only, one agent lies once",
     "clip": None, "confidence": 0.92,
     "lie_once": {"n11b_tactical_analysis"}},
]

# What each run measured. If a later change to the graph alters a cost, this
# is where you find out.
EXPECTED_COSTS = {                         # (steps, node runs, agent calls)
    "Video, confident":                        (12, 16, 2),
    "Text-only (no clip)":                     (9, 11, 1),
    "Video, one retry then accept":            (17, 21, 2),
    "Video, one agent lies once":              (15, 19, 3),
    "Retries used up (degrade)":               (15, 16, 0),
    "Worst case: 2 retries, agents never fix": (29, 35, 6),
    "Text-only, one agent lies once":          (12, 14, 2),
}

# Scenario names as they appear in the paper, in the order the paper lists them.
PAPER_ROW_NAMES = {
    "Video, confident":                        "Video, confident",
    "Text-only (no clip)":                     "Text-only",
    "Video, one retry then accept":            "Video, 1 retry",
    "Video, one agent lies once":              "Video, 1 fabrication",
    "Text-only, one agent lies once":          "Text-only, 1 fabrication",
    "Retries used up (degrade)":               "Retries exhausted",
    "Worst case: 2 retries, agents never fix": "Worst case",
}

# The bounds, fixed when the graph is constructed.
WORST_STEPS = predicted_steps(has_clip=True, retries=MAX_RETRIES,
                              regens=MAX_REGENS, dropped=True, degraded=False)
WORST_AGENT_CALLS = MAX_ACTIVE_AGENTS * (1 + MAX_REGENS)


def run_all_scenarios(graph=None) -> list:
    """Run every scenario and return one profile row per scenario."""
    rows = []
    for sc in SCENARIOS:
        timeline = run_scenario(sc["clip"], sc["confidence"],
                                lie_once=sc.get("lie_once", ()),
                                lie_always=sc.get("lie_always", ()),
                                graph=graph)
        rows.append({"scenario": sc["name"], **profile(timeline)})
    return rows
