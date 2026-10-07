"""N10 Supervisor Router: the LLM proposes, the rules decide.

Routing has two stages, kept apart on purpose:

  1. `propose_agents` -- who SHOULD answer, best first. Either the keyword
     scorer or the language model, depending on `Settings.router_mode`.
  2. the rules, applied in `n10_supervisor_router` and identical for both
     routers: agents whose evidence does not exist are removed, at most
     MAX_ACTIVE_AGENTS are kept, and if nobody is left the default pair answers.

Because the rules run after the proposal, the cap, the evidence rule and the
bounds that follow from them hold whichever router proposed.
"""
from langgraph.graph import END

from ..agents import AGENT_REGISTRY, AGENTS_BY_ID
from ..config import DEFAULT_AGENTS, MAX_ACTIVE_AGENTS
from ..llm_router import llm_route
from ..state import CricketMindState, settings_of, trace


def score_agents(state: CricketMindState) -> dict:
    """Score every agent against the query, the Shot DNA and the analytics.

    Returns {agent_id: (score, [reasons])}. Split out from the router so the
    selection logic can be inspected on its own -- this is the part the
    language model replaces, and it is easier to compare against a rubric when
    it is not tangled up with graph plumbing.
    """
    q = (state.get("query") or "").lower()
    dna = state.get("shot_dna") or {}
    stats = (state.get("stats_results") or {}).get("by_shot", {})
    scores = {}

    # 1. keyword evidence from the query
    for a in AGENT_REGISTRY:
        hits = [k for k in a["keywords"] if k in q]
        if hits:
            scores[a["id"]] = [len(hits), [f"query mentions {', '.join(hits)}"]]

    # 2. structural signal: a shaky classification is a technique question.
    #    A text-only query has no visual confidence at all, so the signal does
    #    not apply. Note dna.get("confidence", 1.0) would return None here, not
    #    the default -- the key exists, its value is None.
    conf = dna.get("confidence")
    if conf is not None and conf < 0.80:
        e = scores.setdefault("n11c_personal_performance", [0, []])
        e[0] += 1
        e[1].append(f"low visual confidence ({conf:.2f})")

    # 3. structural signal: a wide spread between shots is tactical
    if len(stats) >= 2:
        rates = [a["strike_rate"] for a in stats.values()]
        if max(rates) - min(rates) >= 20:
            e = scores.setdefault("n11b_tactical_analysis", [0, []])
            e[0] += 1
            e[1].append(f"SR spread {max(rates) - min(rates):.0f} between shots")

    return {k: (v[0], v[1]) for k, v in scores.items()}


def agents_missing_facts(state: CricketMindState) -> set:
    """Agent ids whose required citations this query cannot supply.

    Kept separate from the scoring so the reason an agent stayed asleep is
    "its evidence does not exist", not "it scored low".
    """
    dna = state.get("shot_dna") or {}
    if dna.get("posture"):
        return set()
    return {a["id"] for a in AGENT_REGISTRY if "weight_transfer" in a["cites"]}


def propose_agents(state: CricketMindState) -> dict:
    """Stage 1 of N10: who SHOULD answer, before any rule is applied.

    router_mode "keyword": the keyword scorer, best score first, ties broken by
    registry order so runs are reproducible.
    router_mode "llm": the local language model. If it cannot be reached or
    replies badly, the keyword scorer is used instead and the run records that
    it happened.

    Returns {"router": who proposed, "ranked": agent ids, best first,
             "reason": the LLM's reason (or None),
             "scored": the keyword evidence (or None),
             "seconds": how long the LLM took (or None)}.
    """
    cfg = settings_of(state)

    router = "keyword"
    if cfg.router_mode == "llm":
        try:
            answer = llm_route(state.get("query") or "", cfg)
            return {"router": "llm", "ranked": answer["agents"],
                    "reason": answer["reason"], "scored": None,
                    "seconds": answer["seconds"]}
        except (OSError, ValueError, KeyError) as e:   # unreachable, or a bad reply
            trace(state, "N10", f"LLM router failed ({type(e).__name__}) -> keyword router")
            router = "keyword (LLM unavailable)"

    scored = score_agents(state)
    order = {a["id"]: i for i, a in enumerate(AGENT_REGISTRY)}
    ranked = sorted(scored, key=lambda aid: (-scored[aid][0], order[aid]))
    return {"router": router, "ranked": ranked, "reason": None, "scored": scored,
            "seconds": None}


def n10_supervisor_router(state: CricketMindState) -> dict:
    """N10 --- Supervisor Router. Chooses which agents run; does not run them.

    Writes `active_agents`, plus `routing_info` recording who proposed what
    and what the rules removed.
    """
    proposal = propose_agents(state)
    blocked = agents_missing_facts(state)

    skipped = [a for a in proposal["ranked"] if a in blocked]
    allowed = [a for a in proposal["ranked"] if a not in blocked]
    if skipped:
        trace(state, "N10", f"no evidence for {len(skipped)}: "
                            + ", ".join(AGENTS_BY_ID[a]["stakeholder"] for a in skipped))

    if allowed:
        chosen = allowed[:MAX_ACTIVE_AGENTS]
        trace(state, "N10", f"woke {len(chosen)}/{len(AGENT_REGISTRY)}: "
                            + ", ".join(AGENTS_BY_ID[c]["stakeholder"] for c in chosen))
        if proposal["scored"] is not None:          # keyword router: its evidence
            for aid in chosen:
                trace(state, "  ->", f"{aid}: {'; '.join(proposal['scored'][aid][1])}")
        else:                                        # LLM router: its reason
            trace(state, "  ->", f"LLM: {proposal['reason']}")
    else:
        chosen = [a for a in DEFAULT_AGENTS if a not in blocked]
        trace(state, "N10", "no signal -> default "
                            f"{[AGENTS_BY_ID[c]['stakeholder'] for c in chosen]}")

    return {"active_agents": chosen,
            "routing_info": {"router":    proposal["router"],
                             "proposed":  proposal["ranked"],
                             "blocked":   skipped,
                             "reason":    proposal["reason"],
                             "defaulted": not allowed,
                             "seconds":   proposal["seconds"]}}


def dispatch_agents(state: CricketMindState) -> list:
    """Conditional edge that returns a LIST -- LangGraph's fan-out.

    Every name returned is executed in the same superstep. Returning [END]
    rather than [] matters: an empty list leaves the graph with nothing to do
    and no way to finish.
    """
    chosen = state.get("active_agents") or []
    return chosen if chosen else [END]
