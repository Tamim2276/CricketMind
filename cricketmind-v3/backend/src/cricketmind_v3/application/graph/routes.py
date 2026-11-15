"""The branches of the flowchart: each looks at the notebook and says where to go next."""

from cricketmind_v3.application.graph.nodes import NODE_OF
from cricketmind_v3.application.graph.state import GraphState
from cricketmind_v3.domain.faith.gate import MAX_REGENERATIONS


def clip_or_text(state: GraphState) -> list[str]:
    """Day 56: with a clip, the video and the question are read side by side; without
    one, only the question, and the video nodes never run."""
    if state.get("clip") is None:
        return ["n3_question_reader"]
    return ["n1_video_input", "n3_question_reader"]


def after_reading(state: GraphState) -> list[str]:
    """Day 59: a question no ball-by-ball record can answer goes straight to the honest
    answer; the engines and the agents never see it."""
    if state["reading"].kind == "cant_check":
        return ["n13_final_output"]
    return ["n6_shot_dna"]


def to_the_chosen(state: GraphState) -> list[str]:
    """Day 73: only the agents N10 chose run, side by side. With nobody chosen no agent
    runs, and the answer says what was missing (D-084)."""
    chosen = state["routing"].chosen
    if not chosen:
        return ["n13_final_output"]
    found = []
    for agent in chosen:
        found.append(NODE_OF[agent])
    return found


def after_gate(state: GraphState) -> list[str]:
    """Day 74: only the agents that failed go back, with the reasons, and at most twice.
    After that, whatever still fails is withheld, and N12 goes on with what passed."""
    if state["checks_run"] > MAX_REGENERATIONS:
        return ["n12_report_synthesis"]
    failing = []
    for failure in state["failures"]:
        node = NODE_OF.get(failure.agent)
        if node is not None and node not in failing:
            failing.append(node)
    if failing:
        return failing
    return ["n12_report_synthesis"]
