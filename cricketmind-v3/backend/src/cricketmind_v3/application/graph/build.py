"""The v4 flowchart, wired (Day 55). Every arrow here is one in the v4 figure; the
branches (clip or not, retry, decline, regenerate, withhold) come on their own days."""

from langgraph.graph import END, START, StateGraph

from cricketmind_v3.application.graph import nodes
from cricketmind_v3.application.graph.state import GraphState


def build_graph():
    graph = StateGraph(GraphState)
    for node in nodes.EVERY_NODE:
        graph.add_node(node)  # named after its function

    # the clip and the question are read side by side
    graph.add_edge(START, "n1_video_input")
    graph.add_edge(START, "n3_question_reader")
    graph.add_edge("n1_video_input", "n2_n5_shot_classifier")
    graph.add_edge("n2_n5_shot_classifier", "confidence_gate")
    graph.add_edge("n1_video_input", "n2b_pose_estimator")

    # a list waits for all of them, so Shot DNA runs once, with everything in
    graph.add_edge(
        ["confidence_gate", "n2b_pose_estimator", "n3_question_reader"], "n6_shot_dna"
    )
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
