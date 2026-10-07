"""The shared graph state, its reducers, and the tracing helper.

Every node in the graph reads from and writes into one `CricketMindState`. A
node returns a **partial** dict containing only the keys it changed; LangGraph
merges that into the running state.

How a key is merged depends on whether it carries a reducer:

| Kind                              | Merge behaviour     | Used for                      |
|-----------------------------------|---------------------|-------------------------------|
| plain key                         | last write wins     | scalars only one node writes  |
| `Annotated[list, operator.add]`   | concatenate         | fields written in parallel    |
| `Annotated[list, upsert_findings]`| replace by agent id | findings, which can re-run    |
"""
import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from .config import DEFAULT_SETTINGS, Settings


@dataclass(frozen=True)
class FakeTensor:
    """Stand-in for a torch tensor: carries the real shape, no data.

    Keeps the stub graph free of GPU work while still making shape contracts
    between nodes explicit and checkable.
    """
    shape: tuple
    note: str = ""

    def __repr__(self) -> str:
        return f"<FakeTensor {self.shape} {self.note}>"


def upsert_findings(left: list, right: list) -> list:
    """Decides how to combine agent findings when more than one is written.

    Every agent writes its finding into `agent_findings`. Two different things
    have to happen, depending on WHO wrote it:

      1. Findings from DIFFERENT agents must pile up.
         Coaches writes, Players writes  ->  the report has 2 findings.

      2. If the SAME agent writes twice, the new finding must REPLACE the old.
         Coaches writes, Coaches writes again  ->  the report has 1 finding.

    Rule 2 is the one that matters. When an agent fails the Faith Check it gets
    re-run, so it produces a finding twice - once with the fabricated number and
    once corrected. If we simply glued the two lists together, the rejected
    finding would still be sitting in the list and would reach the user. That is
    exactly what the Faith Check was supposed to prevent.

    The way we get both behaviours is to keep ONE SLOT PER AGENT, named after
    that agent. Writing into a slot that already exists overwrites whatever was
    in it, and the slot stays where it was, so the report does not get
    reshuffled just because one agent had to retry.
    """
    # LangGraph passes `left` as None on the very first write, before anything
    # is in state. Looping over None would crash, so turn it into an empty list.
    if left is None:
        left = []
    if right is None:
        right = []

    # The slots. The key is the agent id ("n11b"), the value is its finding.
    # A dictionary can never hold the same key twice, and that is what stops an
    # agent from ending up with two findings.
    slots = {}

    # Put the findings we already had into their slots.
    for finding in left:
        slots[finding["agent"]] = finding

    # Add the new findings. This single line covers both rules above:
    #   - agent id not seen before -> a new slot is created  (rule 1)
    #   - agent id already there   -> its slot is overwritten (rule 2)
    for finding in right:
        slots[finding["agent"]] = finding

    # The slot names were only there to spot duplicates. Drop them and hand
    # back a plain list of findings, in the order the agents first appeared.
    return list(slots.values())


class CricketMindState(TypedDict, total=False):
    """Shared state for the CricketMind graph (Architecture v2).

    total=False because nodes contribute keys progressively -- at START only
    video_path, query and settings exist, and asserting otherwise would make
    every node responsible for fields it knows nothing about.
    """

    # inputs
    video_path: str
    query: str
    settings: Any                    # the run's Settings; see config.Settings

    # N1/N2/N4/N5: perception
    frames: list                     # N1 -> NUM_FRAMES sampled frames
    frame_features: Any              # N2 -> (T, 1280) per-frame embeddings
    temporal_features: Any           # N4 -> (1280,) transformer-pooled vector
    shot_label: str                  # N5 -> one of SHOT_CLASSES
    shot_confidence: float           # N5 -> top-1 probability

    # control
    retry_count: int                 # Conf. Check loop counter
    regen_count: int                 # Faith Check loop counter
    status: str                      # ok | low_confidence | unverified_claims_dropped

    # N3/N6: grounding
    query_embedding: Any             # N3 -> embedded query
    query_filters: dict              # N3 -> {"bowler_type": "pace", ...}
    shot_dna: dict                   # N6 -> label + posture metrics + filters

    # N7a/N7b: retrieval (parallel writers -> need reducers)
    stats_hits: Annotated[list, operator.add]     # N7a  Cricsheet ball-by-ball
    context_hits: Annotated[list, operator.add]   # N7b  Kaggle historical

    # N8/N9: shared analytical services
    # The ONLY sanctioned sources of numbers. The Faith Check validates every
    # agent claim against these two dicts and nothing else.
    stats_results: dict              # N8 -> deterministic aggregates
    sim_results: dict                # N9 -> Monte Carlo outcomes

    # N10: dispatch
    active_agents: list              # which of N11a-N11l the router woke
    routing_info: dict               # who proposed what, and what rules removed

    # N11a-N11l: findings (parallel writers, re-runnable)
    agent_findings: Annotated[list, upsert_findings]

    # Faith Check
    # plain list, NOT a reducer: only faith_check writes this, and on the
    # second pass through the regenerate loop operator.add would append the
    # new failures to the stale ones, so a corrected agent would still look
    # broken. Reducers are for fields written by parallel nodes.
    faith_failures: list

    # N12/N13
    report: str


def settings_of(state) -> Settings:
    """The Settings for this run, or the defaults if none were supplied."""
    return state.get("settings") or DEFAULT_SETTINGS


def trace(state, node: str, msg: str) -> None:
    """Print a node's own log line, if this run asked for tracing.

    Takes the state rather than reading a global, so one run printing does not
    make another run print.
    """
    if settings_of(state).trace:
        print(f"  [{node:5s}] {msg}")


def new_state(video_path, query: str, settings: Settings = None) -> CricketMindState:
    """Build a valid initial state. Collections start empty rather than absent
    so a reducer never receives None on the first write.

    `video_path` may be None. A question like "what if Rohit had opened instead
    of Kohli?" has no clip to analyse, and the graph routes those straight to N6
    without touching the vision branch.
    """
    return {
        "video_path": video_path,
        "query": query,
        "settings": settings or DEFAULT_SETTINGS,
        "retry_count": 0,
        "regen_count": 0,
        "status": "ok",
        "stats_hits": [],
        "context_hits": [],
        "active_agents": [],
        "agent_findings": [],
        "faith_failures": [],
    }


def describe_state(state: dict, title: str = "STATE") -> None:
    """Compact state dump -- long tensors and lists are summarised, not printed."""
    print(f"--- {title} " + "-" * max(0, 56 - len(title)))
    for k, v in state.items():
        if isinstance(v, (list, tuple)):
            shown = f"[{len(v)} item(s)]"
        elif isinstance(v, dict):
            shown = "{" + ", ".join(list(v)[:4]) + ("..." if len(v) > 4 else "") + "}"
        elif isinstance(v, str) and len(v) > 48:
            shown = v[:45] + "..."
        else:
            shown = v
        print(f"  {k:18s} {shown}")
