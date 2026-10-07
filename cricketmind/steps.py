"""The seven intermediate graphs, kept as snapshots of how the system was built.

Nothing in the running system uses these. They exist because the walkthrough
notebook -- and the thesis chapter it becomes -- shows the graph growing one
stage at a time, and because each one is the smallest graph that demonstrates
a particular LangGraph behaviour:

| builder                   | what it shows                                  |
|---------------------------|------------------------------------------------|
| `build_vision_graph`      | a linear chain, N1 -> N2 -> N4 -> N5           |
| `build_gated_graph`       | a conditional edge and a bounded retry loop    |
| `build_grounded_graph`    | two branches off START, joining at N6          |
| `build_retrieval_graph`   | a fan-out whose parallel writes need reducers  |
| `build_analytics_graph`   | `defer=True` for parents at different depths   |
| `build_router_graph`      | dispatch computed from state                   |
| `build_agent_graph`       | a conditional edge returning a LIST (fan-out)  |

`build_cricketmind_graph` in graph.py is the real one.
"""
from langgraph.graph import END, START, StateGraph

from .agents import AGENT_NODES
from .nodes.analytics import n8_stats_engine, n9_whatif_simulator
from .nodes.grounding import n6_shot_dna, n7a_stats_retrieval, n7b_context_retrieval
from .nodes.perception import (conf_gate, low_confidence_exit, n1_video_input,
                               n2_vision_encoder, n3_text_encoder,
                               n4_temporal_fusion, n5_classifier_head, retry_prep)
from .nodes.routing import dispatch_agents, n10_supervisor_router
from .state import CricketMindState


def _vision_chain(g):
    """N1 -> N2 -> N4 -> N5, the part every step graph shares."""
    g.add_node("n1_video_input", n1_video_input)
    g.add_node("n2_vision_encoder", n2_vision_encoder)
    g.add_node("n4_temporal_fusion", n4_temporal_fusion)
    g.add_node("n5_classifier_head", n5_classifier_head)
    g.add_edge("n1_video_input", "n2_vision_encoder")
    g.add_edge("n2_vision_encoder", "n4_temporal_fusion")
    g.add_edge("n4_temporal_fusion", "n5_classifier_head")
    return g


def build_vision_graph():
    """Step 2: the vision branch only, linear."""
    g = _vision_chain(StateGraph(CricketMindState))
    g.add_edge(START, "n1_video_input")
    g.add_edge("n5_classifier_head", END)
    return g.compile()


def build_gated_graph():
    """Step 3: vision path + Conf. Check + bounded retry loop."""
    g = _vision_chain(StateGraph(CricketMindState))
    g.add_node("retry", retry_prep)
    g.add_node("degrade", low_confidence_exit)

    g.add_edge(START, "n1_video_input")
    g.add_conditional_edges("n5_classifier_head", conf_gate, {
        "accept":    END,
        "retry":     "retry",
        "exhausted": "degrade",
    })
    g.add_edge("retry", "n1_video_input")          # the red dashed loop
    g.add_edge("degrade", END)
    return g.compile()


def build_grounded_graph():
    """Step 4: parallel text branch + gated vision branch, joining at N6.

    Note what is NOT here: an edge from n3_text_encoder to n6_shot_dna. N6
    reads N3's output out of state. Giving it an edge would make N6 fire the
    moment N3 finished, long before the vision branch produced a shot label.
    """
    g = _vision_chain(StateGraph(CricketMindState))
    g.add_node("n3_text_encoder", n3_text_encoder)
    g.add_node("retry", retry_prep)
    g.add_node("degrade", low_confidence_exit)
    g.add_node("n6_shot_dna", n6_shot_dna, defer=True)

    g.add_edge(START, "n1_video_input")
    g.add_edge(START, "n3_text_encoder")           # the whole of "run in parallel"
    g.add_conditional_edges("n5_classifier_head", conf_gate, {
        "accept":    "n6_shot_dna",                # the gate is N6's ONLY trigger
        "retry":     "retry",
        "exhausted": "degrade",
    })
    g.add_edge("retry", "n1_video_input")
    g.add_edge("n6_shot_dna", END)
    g.add_edge("degrade", END)
    return g.compile()


def _grounded_core(g):
    """Everything up to and including the N7a / N7b fan-out."""
    _vision_chain(g)
    g.add_node("n3_text_encoder", n3_text_encoder)
    g.add_node("n6_shot_dna", n6_shot_dna, defer=True)
    g.add_node("n7a_stats_retrieval", n7a_stats_retrieval)
    g.add_node("n7b_context_retrieval", n7b_context_retrieval)
    g.add_node("retry", retry_prep)
    g.add_node("degrade", low_confidence_exit)

    g.add_edge(START, "n1_video_input")
    g.add_edge(START, "n3_text_encoder")
    g.add_conditional_edges("n5_classifier_head", conf_gate, {
        "accept":    "n6_shot_dna",
        "retry":     "retry",
        "exhausted": "degrade",
    })
    g.add_edge("retry", "n1_video_input")
    g.add_edge("n6_shot_dna", "n7a_stats_retrieval")
    g.add_edge("n6_shot_dna", "n7b_context_retrieval")
    g.add_edge("degrade", END)
    return g


def build_retrieval_graph():
    """Step 5: everything from Step 4, plus the parallel retrieval fan-out."""
    g = _grounded_core(StateGraph(CricketMindState))
    g.add_edge("n7a_stats_retrieval", END)
    g.add_edge("n7b_context_retrieval", END)
    return g.compile()


def _analytics_core(g):
    """Retrieval plus the two engines.

    defer=True on both: N9 is triggered by N7b but reads stats_hits, which N7a
    writes, so both retrievers must have finished before either engine runs.
    """
    _grounded_core(g)
    g.add_node("n8_stats_engine", n8_stats_engine, defer=True)
    g.add_node("n9_whatif_simulator", n9_whatif_simulator, defer=True)
    g.add_edge("n7a_stats_retrieval", "n8_stats_engine")
    g.add_edge("n7b_context_retrieval", "n9_whatif_simulator")
    return g


def build_analytics_graph():
    """Step 6: retrieval fan-out feeding the two analytical services."""
    g = _analytics_core(StateGraph(CricketMindState))
    g.add_edge("n8_stats_engine", END)
    g.add_edge("n9_whatif_simulator", END)
    return g.compile()


def _router_core(g):
    """Analytics plus the supervisor router."""
    _analytics_core(g)
    # defer=True: two incoming edges, and LangGraph has no all-parents barrier
    g.add_node("n10_supervisor_router", n10_supervisor_router, defer=True)
    g.add_edge("n8_stats_engine", "n10_supervisor_router")
    g.add_edge("n9_whatif_simulator", "n10_supervisor_router")
    return g


def build_router_graph():
    """Step 7: everything so far, plus the supervisor router."""
    g = _router_core(StateGraph(CricketMindState))
    g.add_edge("n10_supervisor_router", END)
    return g.compile()


def build_agent_graph():
    """Step 8: the router now fans out to the chosen stakeholder agents."""
    g = _router_core(StateGraph(CricketMindState))
    for aid, fn in AGENT_NODES.items():
        g.add_node(aid, fn)

    # the fan-out: one conditional edge, many possible destinations
    g.add_conditional_edges("n10_supervisor_router", dispatch_agents,
                            list(AGENT_NODES) + [END])
    for aid in AGENT_NODES:
        g.add_edge(aid, END)
    return g.compile()


STEP_GRAPHS = {
    "2 vision":    build_vision_graph,
    "3 gated":     build_gated_graph,
    "4 grounded":  build_grounded_graph,
    "5 retrieval": build_retrieval_graph,
    "6 analytics": build_analytics_graph,
    "7 router":    build_router_graph,
    "8 agents":    build_agent_graph,
}
