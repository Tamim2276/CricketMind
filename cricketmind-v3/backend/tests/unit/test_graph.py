"""Day 55: the v4 flowchart, run end to end. Day 56: the branch for a clip, and N3."""

import dataclasses
import json
from datetime import date

import pytest
from langgraph.graph import END, START

from cricketmind_v3.application.engine_source import EngineFactSource, Repositories
from cricketmind_v3.application.fact_surface import surface
from cricketmind_v3.application.graph import nodes
from cricketmind_v3.application.graph.build import build_graph
from cricketmind_v3.application.graph.context import GraphContext
from cricketmind_v3.application.ports import (
    FakeDeliveryRepository,
    FakeFactSource,
    FakeLLM,
    FakeResultRepository,
    FakeReviewRepository,
    FakeVenueRepository,
    FakeWorkloadRepository,
)
from cricketmind_v3.application.reader import QuestionForm
from cricketmind_v3.application.receipt_checker import check_receipt
from cricketmind_v3.application.retrieval import NO_DATABASE
from cricketmind_v3.domain.balls import Ball
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.names import Player
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.routing import BY_MODEL, DEFAULT, NOBODY, PROPOSED
from cricketmind_v3.domain.shot_dna import NO_CLIP, NOT_READ_YET
from cricketmind_v3.domain.subject import CricketSubject

QUESTION = "Kohli out within 6 balls in the T20 powerplay against pace?"
CONTEXT = GraphContext(names=cricket_gazetteer({"V Kohli": ("Kohli", "Virat Kohli")}))
BALL = Ball(
    match_type="T20",
    gender="male",
    batter="V Kohli",
    runs_batter=1,
    wides=0,
    noballs=0,
    player_out=None,
    wicket_kind=None,
)
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
    ("n3_question_reader", "n13_final_output"),
]
for agent in nodes.AGENTS:
    ARROWS.append(("n10_supervisor_router", agent.__name__))
    ARROWS.append((agent.__name__, "faith_gate"))
BRANCHES = [
    (START, "n1_video_input"),
    (START, "n3_question_reader"),
    ("n3_question_reader", "n6_shot_dna"),
    ("n3_question_reader", "n13_final_output"),  # the honest decline (Day 59)
]


def run(clip: str | None = None) -> dict:
    return build_graph().invoke({"question": QUESTION, "clip": clip}, context=CONTEXT)


def node_names() -> list[str]:
    found = []
    for node in nodes.EVERY_NODE:
        found.append(node.__name__)
    return found


def test_the_graph_runs_end_to_end():
    state = run()
    assert state["answer"] == (
        "I read this as: a prediction: V Kohli, out within 6 balls, T20, "
        "the powerplay, against pace.\nstub: nothing is answered yet"
    )
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


def test_n6_writes_the_shot_dna():
    # Day 60: the subject N3 read, and why nothing is seen yet
    text = run()["shot_dna"]
    assert text.subject == run()["subject"]
    assert text.empty_because == NO_CLIP
    video = run("clip.mp4")["shot_dna"]
    assert video.clip == "clip.mp4"
    assert video.empty_because == NOT_READ_YET


def test_n7a_files_the_balls_under_the_subject():
    # Day 60: no database in CONTEXT, and so it says; with one, the balls come back
    (found,) = run()["evidence"]
    assert found.subject == CricketSubject(
        player="V Kohli", match_format="T20", phase="powerplay", bowler_type="pace"
    )
    assert found.none_because == NO_DATABASE
    repo = FakeDeliveryRepository()
    repo.add(found.subject, [BALL, BALL])
    context = dataclasses.replace(CONTEXT, deliveries=repo)
    state = build_graph().invoke({"question": QUESTION, "clip": None}, context=context)
    assert state["evidence"][0].count == 2


def test_n7b_describes_who_the_question_is_about():
    # Day 60: CONTEXT has no player register, so Kohli is not found in it
    assert run()["context"].not_found == ("V Kohli",)
    kohli = Player(person_id="ba607b88", name="V Kohli", teams=("India",))
    context = dataclasses.replace(CONTEXT, players=(kohli,))
    state = build_graph().invoke({"question": QUESTION, "clip": None}, context=context)
    assert state["context"].people == ("V Kohli; India",)


def test_n8_and_n9_write_the_fact_surface_side_by_side():
    # Day 61: both write the same page, and the reducer keeps both
    past = CricketSubject(
        player="V Kohli", match_format="T20", phase="powerplay", bowler_type="pace"
    )
    ahead = dataclasses.replace(past, k=6)
    engines = FakeFactSource()
    engines.add("strike_rate", past, Fact(value=130.4, n=2855))
    engines.add(
        "p_dismissal_within_k", ahead, Fact(value=0.18, n=2855, interval=(0.1, 0.3))
    )
    context = dataclasses.replace(CONTEXT, engines=engines)
    state = build_graph().invoke({"question": QUESTION, "clip": None}, context=context)
    assert surface(state["answers"]) in ((130.4, 0.18), (0.18, 130.4))


def test_without_engines_the_surface_is_empty():
    assert run()["answers"] == []


def test_a_declined_question_asks_the_engines_nothing():
    engines = FakeFactSource()
    context = dataclasses.replace(CONTEXT, engines=engines)
    state = build_graph().invoke(
        {"question": "Is Kohli injured?", "clip": None}, context=context
    )
    assert state.get("answers", []) == []


def test_the_gate_keeps_its_own_receipts_in_the_notebook():
    # Day 61: a finding put in by hand, as no agent writes one until Day 67
    kohli = CricketSubject(player="V Kohli")
    repo = FakeDeliveryRepository()
    repo.add(kohli, [BALL, BALL])
    engines = EngineFactSource(
        Repositories(
            deliveries=repo,
            venues=FakeVenueRepository(),
            workload=FakeWorkloadRepository(date(2026, 10, 14)),
            reviews=FakeReviewRepository(),
            results=FakeResultRepository(),
        ),
        data="test data",
    )
    claim = Claim(
        id="c1",
        kind=Provenance.DESCRIPTIVE,
        metric="strike_rate",
        subject=kohli,
        value=999.0,  # what the agent says; the receipt keeps what the engine says
        n=2,
    )
    finding = Finding(agent="N11b", claims=(claim,), template="{c1}.")
    context = dataclasses.replace(CONTEXT, engines=engines)
    state = build_graph().invoke(
        {"question": QUESTION, "clip": None, "findings": [finding]}, context=context
    )
    receipt = state["receipts"][("N11b", "c1")]
    assert receipt.engine == "N8.M1"
    assert receipt.value == 100.0
    assert receipt.n == 2
    assert check_receipt(receipt, engines) == []  # anyone can make it again


def test_without_engines_or_findings_no_receipts():
    assert run()["receipts"] == {}


def test_n10_says_who_can_answer():
    # Day 62: from the numbers N8 and N9 really gave
    past = CricketSubject(
        player="V Kohli", match_format="T20", phase="powerplay", bowler_type="pace"
    )
    engines = FakeFactSource()
    engines.add("runs", past, Fact(value=3724.0, n=2855))
    context = dataclasses.replace(CONTEXT, engines=engines)
    state = build_graph().invoke({"question": QUESTION, "clip": None}, context=context)
    assert state["capable"] == {"N11c": ("runs",), "N11e": ("runs",)}


def test_with_no_engines_nobody_can_answer():
    assert run()["capable"] == {}


def test_a_rule_reaches_the_explainer_through_the_filter():
    state = run_question("What is LBW?")
    assert state["capable"] == {"N11m": ()}


def test_n10_chooses_from_those_who_can_answer():
    # Day 63: the default pair, as nothing in the words points elsewhere
    past = CricketSubject(
        player="V Kohli", match_format="T20", phase="powerplay", bowler_type="pace"
    )
    engines = FakeFactSource()
    engines.add("strike_rate", past, Fact(value=130.4, n=2855))
    context = dataclasses.replace(CONTEXT, engines=engines)
    state = build_graph().invoke({"question": QUESTION, "clip": None}, context=context)
    assert state["routing"].chosen == ("N11b", "N11c")
    assert state["routing"].how == DEFAULT


def test_the_words_of_the_question_reach_the_router():
    past = CricketSubject(player="V Kohli")
    engines = FakeFactSource()
    engines.add("strike_rate", past, Fact(value=130.4, n=2855))
    context = dataclasses.replace(CONTEXT, engines=engines)
    state = build_graph().invoke(
        {"question": "Is Kohli a fantasy pick?", "clip": None}, context=context
    )
    assert state["routing"].chosen == ("N11e",)
    assert state["routing"].how == PROPOSED


def test_with_no_numbers_the_router_chooses_nobody():
    routing = run()["routing"]
    assert routing.chosen == ()
    assert routing.how == NOBODY


def test_a_rule_is_routed_to_the_explainer():
    assert run_question("What is LBW?")["routing"].chosen == ("N11m",)


def test_the_model_reads_the_question_and_then_proposes_who_answers():
    # Day 64: one model, two jobs: N3's reading first, then N10's proposal
    reading = dict.fromkeys(QuestionForm.model_fields)
    reading.update(kind="stats", player="Kohli")
    proposal = {"reason": "a fan's question", "agents": ["N11e", "N11k"]}
    engines = FakeFactSource()
    engines.add(
        "strike_rate", CricketSubject(player="V Kohli"), Fact(value=130.4, n=2855)
    )
    llm = FakeLLM([json.dumps(reading), json.dumps(proposal)])
    context = GraphContext(
        llm=llm,
        players=(Player(person_id="ba607b88", name="V Kohli", matches=600),),
        engines=engines,
    )
    state = build_graph().invoke(
        {"question": "Would you start Kohli in my team?", "clip": None}, context=context
    )
    routing = state["routing"]
    assert routing.by == BY_MODEL
    assert routing.chosen == ("N11e",)
    assert routing.no_number == ("N11k",)
    assert routing.reason == "a fan's question"
    assert len(llm.asked) == 2
    assert "Would you start Kohli in my team?" in llm.asked[1][1]
    # the reason is for us: no answer shows it
    assert "a fan's question" not in state["answer"]


def test_every_agent_still_runs_until_day_73():
    # the choice is only written down for now
    visited = run()["visited"]
    for agent in nodes.AGENTS:
        assert agent.__name__ in visited


def test_n3_reads_with_the_model_when_it_has_one():
    reply = dict.fromkeys(QuestionForm.model_fields)
    reply.update(kind="prediction", player="Kohli", k=6)
    context = GraphContext(
        names=CONTEXT.names,
        llm=FakeLLM([json.dumps(reply)]),
        players=(Player(person_id="ba607b88", name="V Kohli", matches=600),),
    )
    state = build_graph().invoke(
        {"question": "Kohli out within 6 balls?", "clip": None}, context=context
    )
    assert state["reading"].by == "model"
    assert state["subject"] == CricketSubject(player="V Kohli", k=6)


def test_n3_reads_by_rules_without_a_model():
    assert run()["reading"].by == "rules"


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


# Day 59: a question nobody can check takes the short path


def run_question(question: str, clip: str | None = None) -> dict:
    return build_graph().invoke({"question": question, "clip": clip}, context=CONTEXT)


def test_a_question_nobody_can_check_goes_straight_to_the_answer():
    state = run_question("Is Kohli injured?")
    assert state["reading"].kind == "cant_check"
    assert state["visited"] == ["n3_question_reader", "n13_final_output"]
    assert state["answer"].startswith(
        "I read this as: a question no ball-by-ball record can answer: V Kohli."
    )
    assert "I can't check that" in state["answer"]
    assert "You could ask instead: V Kohli's record" in state["answer"]


def test_a_clip_makes_it_a_video_question():
    state = run_question("Is Kohli injured?", "clip.mp4")
    assert state["reading"].kind == "video"
    assert state["visited"].count("n13_final_output") == 1


def test_the_engines_and_agents_never_see_a_declined_question():
    visited = run_question("Kohli's salary?")["visited"]
    for name in node_names():
        if name not in ("n3_question_reader", "n13_final_output"):
            assert name not in visited
