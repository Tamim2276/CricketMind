"""The routing comparison: keyword router versus language-model router.

29 prompts. For each of the twelve stakeholders, a *keyword* prompt that uses
the agent's trigger words and a *plain* prompt that avoids all of them, plus
five tricky prompts. Each prompt is routed once by the deterministic keyword
router and (optionally) several times by the LLM router.

Also records a prompt's journey through the whole graph, stage by stage, with
stream_mode="updates".
"""
import json
import time
from collections import Counter

from ..agents import AGENTS_BY_ID, POSTURE_AGENTS, agent_name
from ..config import CLIP, PROJECT_ROOT, QUERY, Settings
from ..llm_router import ollama_ready
from ..nodes.routing import score_agents
from ..state import new_state
from .profile import default_graph

LLM_RESULTS = PROJECT_ROOT / "experiments" / "routing" / "llm_routing_runs.json"

# Every prompt is about Kohli (or no named player): the synthetic corpus holds
# only Kohli's deliveries, and a player with no data is not handled yet.
ROUTING_PROMPTS = [
    # a -- Franchises / Boards
    {"id": "a-keyword", "target": "n11a_club_operations", "style": "keyword", "clip": CLIP,
     "prompt": "Should we keep Kohli in the squad or release him before the auction?"},
    {"id": "a-plain", "target": "n11a_club_operations", "style": "plain", "clip": CLIP,
     "prompt": "Should our team keep Kohli next season, or let him go to save money?"},
    # b -- Coaches
    {"id": "b-keyword", "target": "n11b_tactical_analysis", "style": "keyword", "clip": CLIP,
     "prompt": "What field should we set to exploit Kohli's weakness against spin?"},
    {"id": "b-plain", "target": "n11b_tactical_analysis", "style": "plain", "clip": CLIP,
     "prompt": "How do we get Kohli out early when he faces spin?"},
    # c -- Players
    {"id": "c-keyword", "target": "n11c_personal_performance", "style": "keyword", "clip": CLIP,
     "prompt": "How is Kohli's technique on the cover drive trending this season?"},
    {"id": "c-plain", "target": "n11c_personal_performance", "style": "plain", "clip": CLIP,
     "prompt": "Has Kohli's batting got better or worse this season, and what should he change?"},
    # d -- Scouts
    {"id": "d-keyword", "target": "n11d_global_scouting", "style": "keyword", "clip": CLIP,
     "prompt": "Which emerging batters in other leagues play the pull shot like Kohli?"},
    {"id": "d-plain", "target": "n11d_global_scouting", "style": "plain", "clip": CLIP,
     "prompt": "Find young overseas batters who could become the next Kohli."},
    # e -- Fans
    {"id": "e-keyword", "target": "n11e_fan_concierge", "style": "keyword", "clip": CLIP,
     "prompt": "Should I pick Kohli as my fantasy captain for the next match?"},
    {"id": "e-plain", "target": "n11e_fan_concierge", "style": "plain", "clip": CLIP,
     "prompt": "I want to watch Kohli bat live; which home game should I go to?"},
    # f -- Stadiums
    {"id": "f-keyword", "target": "n11f_stadium_operations", "style": "keyword", "clip": CLIP,
     "prompt": "Should the curator prepare a greener pitch to trouble Kohli against pace?"},
    {"id": "f-plain", "target": "n11f_stadium_operations", "style": "plain", "clip": CLIP,
     "prompt": "How should the ground staff get the wicket ready if we want Kohli to struggle?"},
    # g -- Sponsors
    {"id": "g-keyword", "target": "n11g_sponsorship_intelligence", "style": "keyword", "clip": CLIP,
     "prompt": "Which sponsor brand gets the most logo visibility when Kohli plays the cover drive?"},
    {"id": "g-plain", "target": "n11g_sponsorship_intelligence", "style": "plain", "clip": CLIP,
     "prompt": "Is paying to put our company name on Kohli's bat good value for money?"},
    # h -- Broadcasters
    {"id": "h-keyword", "target": "n11h_content_production", "style": "keyword", "clip": CLIP,
     "prompt": "Make a highlight clip with a wagon wheel overlay of Kohli's pull shots"},
    {"id": "h-plain", "target": "n11h_content_production", "style": "plain", "clip": CLIP,
     "prompt": "What on-screen visuals should the TV team show when Kohli walks out to bat?"},
    # i -- Governing Bodies
    {"id": "i-keyword", "target": "n11i_compliance", "style": "keyword", "clip": CLIP,
     "prompt": "Does Kohli's bat meet ICC eligibility rules?"},
    {"id": "i-plain", "target": "n11i_compliance", "style": "plain", "clip": CLIP,
     "prompt": "Did Kohli break any of the game's official regulations in this innings?"},
    # j -- Academies
    {"id": "j-keyword", "target": "n11j_talent_development", "style": "keyword", "clip": CLIP,
     "prompt": "Design an academy drill so U-19 batters learn the cover drive"},
    {"id": "j-plain", "target": "n11j_talent_development", "style": "plain", "clip": CLIP,
     "prompt": "How can we teach teenage batters to hit the ball through the covers like Kohli?"},
    # k -- Medical Teams
    {"id": "k-keyword", "target": "n11k_injury_management", "style": "keyword", "clip": CLIP,
     "prompt": "Is Kohli's workload a fatigue risk before the final?"},
    {"id": "k-plain", "target": "n11k_injury_management", "style": "plain", "clip": CLIP,
     "prompt": "Is Kohli too tired or sore to play the final safely?"},
    # l -- Player Agents / Managers
    {"id": "l-keyword", "target": "n11l_player_representation", "style": "keyword", "clip": CLIP,
     "prompt": "What is Kohli's market valuation before the auction?"},
    {"id": "l-plain", "target": "n11l_player_representation", "style": "plain", "clip": CLIP,
     "prompt": "How much money should Kohli ask for in his next deal?"},
    # tricky prompts -- no single intended stakeholder
    {"id": "no-signal", "target": None, "style": "tricky", "clip": CLIP,
     "prompt": "Tell me about this delivery"},
    {"id": "many-asked", "target": None, "style": "tricky", "clip": CLIP,
     "prompt": "Scout, sponsor, injury, broadcast, fantasy and auction view of Kohli's drive"},
    {"id": "running-example", "target": None, "style": "tricky", "clip": CLIP,
     "prompt": QUERY},
    {"id": "text-compare", "target": None, "style": "tricky", "clip": None,
     "prompt": "Compare Kohli's cover drive and pull shot against pace in the powerplay"},
    {"id": "text-academy", "target": None, "style": "tricky", "clip": None,
     "prompt": "Design an academy drill so U-19 batters learn the cover drive"},
]


def route_prompt(p: dict, router_mode: str = "keyword", graph=None,
                 ollama_url: str = None) -> dict:
    """Run one prompt through the whole graph and record the router's choice.

    The clip is always classified confidently and no agent lies, so the only
    thing that varies is the question.

    Returns a row: the prompt's id, style and target; which router ran; the
    agents it proposed, the ones the rules removed, and the ones woken; a
    reason for each woken agent -- the keyword evidence, "llm", or "default";
    and how the target was woken: "signal", "default", "no", or None for
    prompts without a target.
    """
    graph = graph or default_graph()
    settings = Settings(confidence=0.92, router_mode=router_mode, trace=False)
    if ollama_url is not None:
        settings = settings.with_(ollama_url=ollama_url)

    final = graph.invoke(new_state(p["clip"], p["prompt"], settings))

    info = final["routing_info"]
    woken = final["active_agents"]

    if info["defaulted"]:                     # nobody proposed survived the rules
        why = {a: "default" for a in woken}
    elif info["router"] == "llm":             # the model's single reason covers them all
        why = {a: "llm" for a in woken}
    else:                                     # the keyword scorer's evidence, per agent
        scored = score_agents(final)
        why = {a: "; ".join(r.replace("query mentions ", "") for r in scored[a][1])
               for a in woken}

    if p["target"] is None:
        target_woken = None
    elif p["target"] not in woken:
        target_woken = "no"
    elif why[p["target"]] == "default":
        target_woken = "default"
    else:
        target_woken = "signal"

    return {"id": p["id"], "style": p["style"], "target": p["target"],
            "clip": p["clip"], "status": final["status"],
            "router": info["router"], "proposed": info["proposed"],
            "blocked": info["blocked"], "reason": info["reason"],
            "seconds": info["seconds"],
            "woken": woken, "why": why, "target_woken": target_woken}


def route_all(router_mode: str = "keyword", graph=None) -> list:
    """Route all 29 prompts with one router."""
    return [route_prompt(p, router_mode, graph) for p in ROUTING_PROMPTS]


def load_or_run_llm(runs: int = 3, rerun: bool = False, graph=None,
                    settings: Settings = None) -> list:
    """The LLM router's answers: the saved ones, or fresh ones from Ollama.

    The saved answers are only valid for this exact model, run count and
    prompt list; anything else is re-run or skipped. Returns None when there
    is nothing saved and Ollama is not available.
    """
    cfg = settings or Settings(router_mode="llm")
    setup = {"model": cfg.router_model, "runs": runs,
             "prompts": [[p["id"], p["prompt"]] for p in ROUTING_PROMPTS]}

    saved = json.loads(LLM_RESULTS.read_text(encoding="utf-8")) if LLM_RESULTS.exists() else None
    if saved and saved["setup"] == setup and not rerun:
        return saved["runs"]

    if not ollama_ready(cfg):
        return None

    out = []
    for run in range(runs):
        start = time.perf_counter()
        out.append([route_prompt(p, "llm", graph) for p in ROUTING_PROMPTS])
        print(f"run {run + 1}/{runs}: {time.perf_counter() - start:.0f} s")

    LLM_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    LLM_RESULTS.write_text(json.dumps({"setup": setup, "runs": out}, indent=1),
                           encoding="utf-8")
    return out


def compare(keyword_rows: list, llm_runs: list) -> dict:
    """The totals behind the summary table.

    Returns {"per_run": {label: (keyword values, LLM values per run)},
             "stable": [one bool per prompt], "seconds": [every LLM call]}.
    """
    per_run = {
        "keyword prompts, target woken by the router":
            ([sum(r["style"] == "keyword" and r["target_woken"] == "signal"
                  for r in keyword_rows)],
             [sum(r["style"] == "keyword" and r["target_woken"] == "signal" for r in run)
              for run in llm_runs]),
        "plain prompts, target woken by the router":
            ([sum(r["style"] == "plain" and r["target_woken"] == "signal"
                  for r in keyword_rows)],
             [sum(r["style"] == "plain" and r["target_woken"] == "signal" for r in run)
              for run in llm_runs]),
        "prompts that fell back to the default pair":
            ([sum(all(w == "default" for w in r["why"].values()) for r in keyword_rows)],
             [sum(all(w == "default" for w in r["why"].values()) for r in run)
              for run in llm_runs]),
        "agents woken per prompt (mean)":
            ([sum(len(r["woken"]) for r in keyword_rows) / len(keyword_rows)],
             [sum(len(r["woken"]) for r in run) / len(run) for run in llm_runs]),
    }
    first = llm_runs[0]
    stable = [all(run[i]["woken"] == first[i]["woken"] for run in llm_runs)
              for i in range(len(ROUTING_PROMPTS))]
    seconds = [r["seconds"] for run in llm_runs for r in run]
    return {"per_run": per_run, "stable": stable, "seconds": seconds}


# --- one query through the graph, stage by stage ----------------------------

def record_writes(state, graph=None) -> list:
    """Run one query and record what every node wrote into the state.

    Like record_run, but with stream_mode="updates": each event is
    {node_name: the dictionary that node returned}. Returns (node, update)
    pairs in the order the nodes finished.
    """
    graph = graph or default_graph()
    writes = []
    for event in graph.stream(state, stream_mode="updates"):
        for node, update in event.items():
            writes.append((node, update or {}))
    return writes


def describe_write(node: str, update: dict) -> str:
    """One short, readable line for what a node wrote."""
    u = update
    if node == "n1_video_input":
        return f"{len(u['frames'])} frames sampled"
    if node in ("n2_vision_encoder", "n4_temporal_fusion"):
        key = "frame_features" if node == "n2_vision_encoder" else "temporal_features"
        return f"features {u[key].shape}"
    if node == "n3_text_encoder":
        parts = [f"{k}={'+'.join(v) if isinstance(v, list) else v}"
                 for k, v in u["query_filters"].items()]
        return ", ".join(parts) or "no filters"
    if node == "n5_classifier_head":
        return f"{u['shot_label']} (p={u['shot_confidence']:.2f})"
    if node == "n6_shot_dna":
        dna = u["shot_dna"]
        if dna["posture"]:
            return (f"{dna['shot']} + posture "
                    f"(weight transfer {dna['posture']['weight_transfer']}) + filters")
        return "query filters only (no clip)"
    if node == "n7a_stats_retrieval":
        return f"{len(u['stats_hits'])} matching deliveries"
    if node == "n7b_context_retrieval":
        return f"{len(u['context_hits'])} context rows"
    if node == "n8_stats_engine":
        return ", ".join(f"{shot} SR {s['strike_rate']} ({s['balls']} balls)"
                         for shot, s in u["stats_results"]["by_shot"].items())
    if node == "n9_whatif_simulator":
        s = u["sim_results"]
        lo, hi = s["ci95"]
        return f"{s['scenario']}: {s['p_dismissal']:.1%} out (95% CI {lo:.1%}-{hi:.1%})"
    if node == "n10_supervisor_router":
        return "wakes " + ", ".join(agent_name(a) for a in u["active_agents"])
    if node in AGENTS_BY_ID:
        f = u["agent_findings"][0]
        return f"{agent_name(node)} cites {f['cites']}"
    if node == "faith_check":
        n = len(u["faith_failures"])
        return "all cited figures match the engines" if n == 0 else f"{n} citation(s) rejected"
    if node == "n12_supervisor_synthesis":
        return f"report written ({len(u['report'].splitlines())} lines)"
    if node == "n13_final_output":
        return f"status = {u['status']}"
    return "wrote " + ", ".join(u) if u else "wrote nothing"


FLOW_PROMPTS = {
    "A": QUERY,                                                   # a coaching comparison
    "B": "Is Kohli's workload a fatigue risk before the final?",  # a medical question
}

FLOW_STAGES = [
    ("N3 reads the question",   ["n3_text_encoder"]),
    ("N1-N5 classify the clip", ["n5_classifier_head"]),
    ("N6 builds the Shot DNA",  ["n6_shot_dna"]),
    ("N7a finds deliveries",    ["n7a_stats_retrieval"]),
    ("N7b finds context",       ["n7b_context_retrieval"]),
    ("N8 computes statistics",  ["n8_stats_engine"]),
    ("N9 simulates",            ["n9_whatif_simulator"]),
    ("N10 picks the experts",   ["n10_supervisor_router"]),
    ("N11 agents answer",       list(AGENTS_BY_ID)),
    ("Faith Check",             ["faith_check"]),
    ("N13 output",              ["n13_final_output"]),
]

# The paper keeps the stages whose output differs between the two queries,
# plus the Faith Check; identical rows are left out to fit the page limit.
PAPER_FLOW_STAGES = [s for s in FLOW_STAGES if s[0] not in
                     ("N1-N5 classify the clip", "N6 builds the Shot DNA",
                      "N7b finds context", "N13 output")]


def record_flows(graph=None) -> dict:
    """Record both flow prompts, stage by stage."""
    settings = Settings(confidence=0.92, trace=False)
    return {label: record_writes(new_state(CLIP, q, settings), graph)
            for label, q in FLOW_PROMPTS.items()}


def prompt_style_counts() -> Counter:
    return Counter(p["style"] for p in ROUTING_PROMPTS)


__all__ = ["ROUTING_PROMPTS", "route_prompt", "route_all", "load_or_run_llm",
           "compare", "record_writes", "describe_write", "record_flows",
           "FLOW_PROMPTS", "FLOW_STAGES", "PAPER_FLOW_STAGES",
           "prompt_style_counts", "LLM_RESULTS", "POSTURE_AGENTS"]
