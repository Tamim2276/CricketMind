"""The branches of the flowchart: each looks at the notebook and says where to go next."""

from cricketmind_v3.application.graph.state import GraphState


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
