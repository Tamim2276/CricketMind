"""Day 55: the v4 flowchart, run end to end. Day 56: the branch for a clip, and N3."""

import pytest
from langgraph.graph import END, START

from cricketmind_v3.application.graph import nodes
from cricketmind_v3.application.graph.build import build_graph
from cricketmind_v3.application.graph.context import GraphContext
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.subject import CricketSubject

QUESTION = "Kohli out within 6 balls in the T20 powerplay against pace?"
CONTEXT = GraphContext(names=cricket_gazetteer({"V Kohli": ("Kohli", "Virat Kohli")}))
VIDEO = (
    "n1_video_input",
    "n2_n5_shot_classifier",
    "confidence_gate",
    "n2b_pose_estimator",
)

# every arrow of the v4 figure, written out by hand from the figure
ARROWS = [
    (START, "n1_video_input"),
    (START, "n3_question_reader"),
    ("n1_video_input", "n2_n5_shot_classifier"),
    ("n2_n5_shot_classifier", "confidence_gate"),
    ("n1_video_input", "n2b_pose_estimator"),
    ("confidence_gate", "n6_shot_dna"),
    ("n2b_pose_estimator", "n6_shot_dna"),
    ("n3_question_reader", "n6_shot_dna"),
    ("n6_shot_dna", "n7a_evidence_retrieval"),
    ("n6_shot_dna", "n7b_context_retrieval"),
    ("n7a_evidence_retrieval", "n8_descriptive_engine"),
    ("n7b_context_retrieval", "n8_descriptive_engine"),
    ("n7a_evidence_retrieval", "n9_predictive_engine"),
    ("n7b_context_retrieval", "n9_predictive_engine"),
    ("n8_descriptive_engine", "n10_supervisor_router"),
    ("n9_predictive_engine", "n10_supervisor_router"),
    ("faith_gate", "n12_report_synthesis"),
    ("n12_report_synthesis", "n13_final_output"),
    ("n13_final_output", END),
]
for agent in nodes.AGENTS:
    ARROWS.append(("n10_supervisor_router", agent.__name__))
    ARROWS.append((agent.__name__, "faith_gate"))
BRANCHES = [(START, "n1_video_input"), (START, "n3_question_reader")]


def run(clip: str | None = None) -> dict:
    return build_graph().invoke({"question": QUESTION, "clip": clip}, context=CONTEXT)


def node_names() -> list[str]:
    found = []
    for node in nodes.EVERY_NODE:
        found.append(node.__name__)
    return found


def test_the_graph_runs_end_to_end():
    state = run()
    assert state["answer"] == "stub: nothing is answered yet"
    assert state["question"] == QUESTION  # passed along untouched
    assert state["clip"] is None


def test_the_twelve_agents_and_the_explainer():
    assert len(nodes.AGENTS) == 13
    assert len(nodes.EVERY_NODE) == 27  # 11 before the agents, 13, and 3 after


def test_with_a_clip_every_node_runs_exactly_once():
    # once: Shot DNA waits for whoever is still coming, not once per arrow; and all of
    # them: the reducer keeps every signature written side by side
    assert sorted(run("clip.mp4")["visited"]) == sorted(node_names())


def test_without_a_clip_the_video_nodes_never_run():
    expected = []
    for name in node_names():
        if name not in VIDEO:
            expected.append(name)
    assert sorted(run()["visited"]) == sorted(expected)


def test_n3_reads_the_question():
    expected = CricketSubject(
        player="V Kohli", k=6, match_format="T20", phase="powerplay", bowler_type="pace"
    )
    assert run()["subject"] == expected
    assert run("clip.mp4")["subject"] == expected


def test_the_graph_needs_its_context():
    with pytest.raises(ValueError, match="context"):
        build_graph().invoke({"question": QUESTION, "clip": None})


def test_a_node_is_named_after_its_function():
    found = []
    for name in build_graph().get_graph().nodes:
        if name not in (START, END):
            found.append(name)
    assert sorted(found) == sorted(node_names())


def test_the_arrows_are_the_v4_figure():
    found = []
    branches = []
    for edge in build_graph().get_graph().edges:
        found.append((edge.source, edge.target))
        if edge.conditional:
            branches.append((edge.source, edge.target))
    assert sorted(found) == sorted(ARROWS)
    assert sorted(branches) == sorted(BRANCHES)


def test_no_node_runs_before_the_ones_it_waits_for():
    for clip in ("clip.mp4", None):
        visited = run(clip)["visited"]
        for before, after in ARROWS:
            if before not in visited or after not in visited:
                continue  # START, END, and the video without a clip
            assert visited.index(before) < visited.index(after), (clip, before, after)


def test_the_picture_names_every_node():
    picture = build_graph().get_graph().draw_mermaid()
    for name in node_names():
        assert name in picture
