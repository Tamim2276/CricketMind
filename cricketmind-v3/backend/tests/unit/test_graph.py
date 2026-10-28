"""Day 55: the v4 flowchart with every node a stub, run end to end."""

from langgraph.graph import END, START

from cricketmind_v3.application.graph import nodes
from cricketmind_v3.application.graph.build import build_graph

QUESTION = "Kohli out within 6 balls?"

# every arrow of the v4 figure without a branch, written out by hand from the figure
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


def run() -> dict:
    return build_graph().invoke({"question": QUESTION, "clip": None})


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


def test_every_node_runs_exactly_once():
    # once: the lists in build.py make a node wait for all its arrows, not run per arrow;
    # and all of them: the reducer keeps every signature written side by side
    assert sorted(run()["visited"]) == sorted(node_names())


def test_a_node_is_named_after_its_function():
    found = []
    for name in build_graph().get_graph().nodes:
        if name not in (START, END):
            found.append(name)
    assert sorted(found) == sorted(node_names())


def test_the_arrows_are_the_v4_figure():
    found = []
    for edge in build_graph().get_graph().edges:
        found.append((edge.source, edge.target))
    assert sorted(found) == sorted(ARROWS)


def test_no_node_runs_before_the_ones_it_waits_for():
    visited = run()["visited"]
    for before, after in ARROWS:
        if before == START or after == END:
            continue
        assert visited.index(before) < visited.index(after), (before, after)


def test_the_picture_names_every_node():
    picture = build_graph().get_graph().draw_mermaid()
    for name in node_names():
        assert name in picture
