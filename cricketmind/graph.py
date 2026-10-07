"""The complete CricketMind graph: every node of Architecture v2.

Thirty nodes: one per component of Figure 1, together with the nodes that
drive its two correction loops (`retry`, `regen_prep`) and the withhold path
(`drop_unfaithful`).

Two notes that cost real debugging time:

**An edge is a trigger, not a data dependency.** A node fires as soon as *any*
incoming edge is satisfied -- LangGraph has no all-parents barrier. That is why
N6 has no edge from N3: it reads N3's output from *state*, and the gate's
`accept` route is its only trigger.

**`defer=True` where parents sit at different depths** (N6, N8, N9, N10), and
NOT on `faith_check`: the agents are siblings from one fan-out, so they finish
in the same superstep. Deferring it also made `get_graph()` draw a phantom
`n13 -> faith_check` edge and lose `n13 -> END`.
"""
from langgraph.graph import END, START, StateGraph

from .agents import AGENT_NODES
from .nodes.analytics import n8_stats_engine, n9_whatif_simulator
from .nodes.grounding import n6_shot_dna, n7a_stats_retrieval, n7b_context_retrieval
from .nodes.output import n12_supervisor_synthesis, n13_final_output
from .nodes.perception import (conf_gate, has_video, low_confidence_exit,
                               n1_video_input, n2_vision_encoder, n3_text_encoder,
                               n4_temporal_fusion, n5_classifier_head, retry_prep)
from .nodes.routing import dispatch_agents, n10_supervisor_router
from .nodes.verification import (dispatch_regen, drop_unfaithful, faith_gate,
                                 n_faith_check, regen_prep)
from .state import CricketMindState


def build_cricketmind_graph():
    """Compile the full graph. Settings travel in the state, not in here."""
    g = StateGraph(CricketMindState)

    # --- perception -------------------------------------------------
    g.add_node("n1_video_input",        n1_video_input)
    g.add_node("n2_vision_encoder",     n2_vision_encoder)
    g.add_node("n4_temporal_fusion",    n4_temporal_fusion)
    g.add_node("n5_classifier_head",    n5_classifier_head)
    g.add_node("n3_text_encoder",       n3_text_encoder)
    g.add_node("retry",   retry_prep)
    g.add_node("degrade", low_confidence_exit)

    # --- grounding, retrieval, analytics ----------------------------
    g.add_node("n6_shot_dna",           n6_shot_dna,           defer=True)
    g.add_node("n7a_stats_retrieval",   n7a_stats_retrieval)
    g.add_node("n7b_context_retrieval", n7b_context_retrieval)
    g.add_node("n8_stats_engine",       n8_stats_engine,       defer=True)
    g.add_node("n9_whatif_simulator",   n9_whatif_simulator,   defer=True)
    g.add_node("n10_supervisor_router", n10_supervisor_router, defer=True)

    # --- the twelve stakeholder agents ------------------------------
    for aid, fn in AGENT_NODES.items():
        g.add_node(aid, fn)

    # --- verification and synthesis ---------------------------------
    g.add_node("faith_check",      n_faith_check)
    g.add_node("regen_prep",       regen_prep)
    g.add_node("drop_unfaithful",  drop_unfaithful)
    g.add_node("n12_supervisor_synthesis", n12_supervisor_synthesis)
    g.add_node("n13_final_output",         n13_final_output)

    # --- edges ------------------------------------------------------
    # Two ways in. Only a query carrying a clip enters the vision branch;
    # a text-only question goes straight to N6 and grounds on its filters.
    g.add_conditional_edges(START, has_video, {
        "video":     "n1_video_input",
        "text_only": "n6_shot_dna",
    })
    g.add_edge(START, "n3_text_encoder")     # every query has text
    g.add_edge("n1_video_input",     "n2_vision_encoder")
    g.add_edge("n2_vision_encoder",  "n4_temporal_fusion")
    g.add_edge("n4_temporal_fusion", "n5_classifier_head")

    g.add_conditional_edges("n5_classifier_head", conf_gate, {
        "accept":    "n6_shot_dna",
        "retry":     "retry",
        "exhausted": "degrade",
    })
    g.add_edge("retry", "n1_video_input")

    g.add_edge("n6_shot_dna", "n7a_stats_retrieval")
    g.add_edge("n6_shot_dna", "n7b_context_retrieval")
    g.add_edge("n7a_stats_retrieval",   "n8_stats_engine")
    g.add_edge("n7b_context_retrieval", "n9_whatif_simulator")
    g.add_edge("n8_stats_engine",     "n10_supervisor_router")
    g.add_edge("n9_whatif_simulator", "n10_supervisor_router")

    g.add_conditional_edges("n10_supervisor_router", dispatch_agents,
                            list(AGENT_NODES) + [END])
    for aid in AGENT_NODES:
        g.add_edge(aid, "faith_check")

    g.add_conditional_edges("faith_check", faith_gate, {
        "faithful":   "n12_supervisor_synthesis",
        "regenerate": "regen_prep",
        "exhausted":  "drop_unfaithful",
    })
    g.add_conditional_edges("regen_prep", dispatch_regen, list(AGENT_NODES))
    g.add_edge("drop_unfaithful", "n12_supervisor_synthesis")

    g.add_edge("n12_supervisor_synthesis", "n13_final_output")
    g.add_edge("n13_final_output", END)
    g.add_edge("degrade", END)                # terminal: no report is produced

    return g.compile()


def show_graph(compiled, png: bool = True, mermaid: bool = False):
    """Render a compiled LangGraph.

    png=True     try mermaid.ink for an inline image, fall back to ASCII offline
    mermaid=True also print the Mermaid source (handy for pasting into a thesis)
    """
    from IPython.display import Image, display

    g = compiled.get_graph()
    if png:
        try:
            display(Image(g.draw_mermaid_png()))
        except Exception as e:
            print(f"(no PNG: {type(e).__name__} -- falling back to ASCII)\n")
            print(g.draw_ascii())
    else:
        print(g.draw_ascii())

    if mermaid:
        print("\n--- Mermaid source " + "-" * 40)
        print(g.draw_mermaid())
