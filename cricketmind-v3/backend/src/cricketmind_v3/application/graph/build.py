"""The v4 flowchart, wired (Day 55). Every arrow here is one in the v4 figure. The
branches come on their own days: clip or text (Day 56); the classifier's status (Day
118), decline, regenerate and withhold later."""

from langgraph.graph import END, START, StateGraph

from cricketmind_v3.application.graph import nodes, routes
from cricketmind_v3.application.graph.context import GraphContext
from cricketmind_v3.application.graph.state import GraphState


def build_graph():
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
    graph.add_edge("n3_question_reader", "n6_shot_dna")
    graph.add_edge("n6_shot_dna", "n7a_evidence_retrieval")
    graph.add_edge("n6_shot_dna", "n7b_context_retrieval")
    retrievers = ["n7a_evidence_retrieval", "n7b_context_retrieval"]
    graph.add_edge(retrievers, "n8_descriptive_engine")
    graph.add_edge(retrievers, "n9_predictive_engine")
    graph.add_edge(
        ["n8_descriptive_engine", "n9_predictive_engine"], "n10_supervisor_router"
    )

    # every agent for now; Day 73 sends the question only to the chosen ones
    agents = []
    for agent in nodes.AGENTS:
        graph.add_edge("n10_supervisor_router", agent.__name__)
        agents.append(agent.__name__)
    graph.add_edge(agents, "faith_gate")

    graph.add_edge("faith_gate", "n12_report_synthesis")
    graph.add_edge("n12_report_synthesis", "n13_final_output")
    graph.add_edge("n13_final_output", END)
    return graph.compile()
