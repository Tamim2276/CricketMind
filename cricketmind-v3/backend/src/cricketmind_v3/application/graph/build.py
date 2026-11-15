"""The v4 flowchart, wired (Day 55). Every arrow here is one in the v4 figure. The
branches come on their own days: clip or text (Day 56); the honest decline (Day 59); who
answers (Day 73); regenerate (Day 74); the classifier's status (Day 118)."""

from langgraph.graph import END, START, StateGraph

from cricketmind_v3.application.engine_source import ENGINE_VERSIONS
from cricketmind_v3.application.graph import nodes, routes
from cricketmind_v3.application.graph.context import GraphContext
from cricketmind_v3.application.graph.state import GraphState
from cricketmind_v3.domain.registry import RegistryError, check_registry
from cricketmind_v3.domain.render import METRICS
from cricketmind_v3.domain.roster import ROSTER


def check_the_menu() -> None:
    """I2: a measure with no engine or no words stops the build, not a question (Day 71).
    N6's engine comes with the clip (Phase 8); nothing else may be missing."""
    sources = set(ENGINE_VERSIONS)
    sources.add("N6")
    problems = check_registry(ROSTER, sources, set(METRICS))
    if problems:
        raise RegistryError("; ".join(problems))


def build_graph():
    check_the_menu()
    graph = StateGraph(GraphState, context_schema=GraphContext)
    for node in nodes.EVERY_NODE:
        if node is nodes.n6_shot_dna:
            # it waits until every node still running has finished, so it runs once:
            # after the video and the question, or after the question alone
            graph.add_node(node, defer=True)
        else:
            graph.add_node(node)  # named after its function

    # a branch: with a clip, the clip and the question side by side; without, the
    # question only
    graph.add_conditional_edges(
        START, routes.clip_or_text, ["n1_video_input", "n3_question_reader"]
    )
    graph.add_edge("n1_video_input", "n2_n5_shot_classifier")
    graph.add_edge("n2_n5_shot_classifier", "confidence_gate")
    graph.add_edge("n1_video_input", "n2b_pose_estimator")

    graph.add_edge("confidence_gate", "n6_shot_dna")
    graph.add_edge("n2b_pose_estimator", "n6_shot_dna")
    # a branch (Day 59): a question nobody can check goes straight to the answer. A
    # clip makes a question video, so Shot DNA never sees one
    graph.add_conditional_edges(
        "n3_question_reader",
        routes.after_reading,
        ["n6_shot_dna", "n13_final_output"],
    )
    graph.add_edge("n6_shot_dna", "n7a_evidence_retrieval")
    graph.add_edge("n6_shot_dna", "n7b_context_retrieval")
    retrievers = ["n7a_evidence_retrieval", "n7b_context_retrieval"]
    graph.add_edge(retrievers, "n8_descriptive_engine")
    graph.add_edge(retrievers, "n9_predictive_engine")
    graph.add_edge(
        ["n8_descriptive_engine", "n9_predictive_engine"], "n10_supervisor_router"
    )

    # a branch (Day 73): only the chosen agents run, side by side; with nobody chosen,
    # no agent runs. They finish together, so the gate runs once, after them all
    agents = []
    for agent in nodes.AGENTS:
        agents.append(agent.__name__)
        graph.add_edge(agent.__name__, "faith_gate")
    graph.add_conditional_edges(
        "n10_supervisor_router", routes.to_the_chosen, [*agents, "n13_final_output"]
    )

    # a loop, and it is bounded (Day 74): the failing agents go back at most twice
    graph.add_conditional_edges(
        "faith_gate", routes.after_gate, [*agents, "n12_report_synthesis"]
    )
    graph.add_edge("n12_report_synthesis", "n13_final_output")
    graph.add_edge("n13_final_output", END)
    return graph.compile()
