"""Every node of the v4 graph (Day 55): each signs the notebook, and the stubs do nothing
else. The days that follow make them real, one at a time: N3 on Day 56. A node's name is
its function's name."""

from langgraph.runtime import Runtime

from cricketmind_v3.application.fact_surface import ask_engines
from cricketmind_v3.application.faith_check import receipts_of, recompute
from cricketmind_v3.application.graph.context import GraphContext
from cricketmind_v3.application.graph.state import GraphState
from cricketmind_v3.application.reader import read_question
from cricketmind_v3.application.retrieval import retrieve
from cricketmind_v3.application.router import route
from cricketmind_v3.domain.capabilities import capable_agents
from cricketmind_v3.domain.question_context import question_context
from cricketmind_v3.domain.readback import cant_check_answer, read_back
from cricketmind_v3.domain.shot_dna import stub_shot_dna


def given(runtime: Runtime[GraphContext]) -> GraphContext:
    """What the node was given; a graph run without it stops here, not later."""
    if runtime.context is None:
        raise ValueError("the graph was run without its context (a GraphContext)")
    return runtime.context


# A. the clip: its frames, the shot, how sure, the body


def n1_video_input(state: GraphState) -> dict:
    return {"visited": ["n1_video_input"]}


def n2_n5_shot_classifier(state: GraphState) -> dict:
    # N2 to N5 in v4: CricShot's shot-classification agent, as one node (D-072)
    return {"visited": ["n2_n5_shot_classifier"]}


def confidence_gate(state: GraphState) -> dict:
    return {"visited": ["confidence_gate"]}


def n2b_pose_estimator(state: GraphState) -> dict:
    return {"visited": ["n2b_pose_estimator"]}


# B. the question, and the shot it is about


def n3_question_reader(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    # the model reads and the code checks (Day 58); without a model, the rules (Day 56)
    context = given(runtime)
    reading = read_question(
        state["question"],
        context.llm,
        context.players,
        context.names,
        context.grounds,
        context.teams,
        state.get("clip"),
    )
    return {
        "visited": ["n3_question_reader"],
        "subject": reading.subject,
        "reading": reading,
    }


def n6_shot_dna(state: GraphState) -> dict:
    # Day 60: a stub, as nothing reads the clip until Phase 8
    dna = stub_shot_dna(state["subject"], state.get("clip"))
    return {"visited": ["n6_shot_dna"], "shot_dna": dna}


# C. the evidence, and D. the engines


def n7a_evidence_retrieval(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    # Day 60: the balls of each side of the question, from the subject in the Shot DNA
    context = given(runtime)
    kind = state["reading"].kind
    evidence = retrieve(state["shot_dna"].subject, kind, context.deliveries)
    return {"visited": ["n7a_evidence_retrieval"], "evidence": evidence}


def n7b_context_retrieval(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    # Day 60: who the people are and where the ground is, in the registers' words
    context = given(runtime)
    found = question_context(
        state["shot_dna"].subject, context.players, context.grounds, context.teams
    )
    return {"visited": ["n7b_context_retrieval"], "context": found}


def n8_descriptive_engine(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    # Day 61: every count that fits the question
    context = given(runtime)
    answers = ask_engines(
        state["shot_dna"].subject, state["reading"].kind, "N8", context.engines
    )
    return {"visited": ["n8_descriptive_engine"], "answers": answers}


def n9_predictive_engine(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    # Day 61: every simulation that fits the question
    context = given(runtime)
    answers = ask_engines(
        state["shot_dna"].subject, state["reading"].kind, "N9", context.engines
    )
    return {"visited": ["n9_predictive_engine"], "answers": answers}


# E. the router, and F. the twelve agents and the Explainer


def n10_supervisor_router(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    # Day 62: who can answer. Days 63, 64: who should, at most four: the model proposes,
    # the rules decide. Every agent still runs until Day 73 sends only the chosen
    context = given(runtime)
    capable = capable_agents(
        state["reading"].kind, state.get("answers", []), state["shot_dna"]
    )
    routing = route(state["question"], capable, state["evidence"], context.llm)
    return {
        "visited": ["n10_supervisor_router"],
        "capable": capable,
        "routing": routing,
    }


def n11a_club_operations(state: GraphState) -> dict:
    return {"visited": ["n11a_club_operations"]}


def n11b_tactical_analysis(state: GraphState) -> dict:
    return {"visited": ["n11b_tactical_analysis"]}


def n11c_personal_performance(state: GraphState) -> dict:
    return {"visited": ["n11c_personal_performance"]}


def n11d_global_scouting(state: GraphState) -> dict:
    return {"visited": ["n11d_global_scouting"]}


def n11e_fan_concierge(state: GraphState) -> dict:
    return {"visited": ["n11e_fan_concierge"]}


def n11f_stadium_operations(state: GraphState) -> dict:
    return {"visited": ["n11f_stadium_operations"]}


def n11g_bowling_performance(state: GraphState) -> dict:
    return {"visited": ["n11g_bowling_performance"]}


def n11h_content_production(state: GraphState) -> dict:
    return {"visited": ["n11h_content_production"]}


def n11i_match_officials(state: GraphState) -> dict:
    return {"visited": ["n11i_match_officials"]}


def n11j_talent_development(state: GraphState) -> dict:
    return {"visited": ["n11j_talent_development"]}


def n11k_injury_management(state: GraphState) -> dict:
    return {"visited": ["n11k_injury_management"]}


def n11l_player_representation(state: GraphState) -> dict:
    return {"visited": ["n11l_player_representation"]}


def n11m_explainer(state: GraphState) -> dict:
    return {"visited": ["n11m_explainer"]}


# G. the checks, the report, the answer


def faith_gate(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    # Day 61: it asks the engines again and keeps the receipts; it checks nothing until
    # Day 74
    context = given(runtime)
    receipts = {}
    if context.engines is not None:
        for finding in state.get("findings", []):
            facts = recompute(finding, context.engines)
            receipts.update(receipts_of(finding, facts))
    return {"visited": ["faith_gate"], "receipts": receipts}


def n12_report_synthesis(state: GraphState) -> dict:
    return {"visited": ["n12_report_synthesis"]}


def n13_final_output(state: GraphState) -> dict:
    # every answer starts with what was read (Day 59); a question nobody can check
    # gets the honest answer, and the rest a stub until Day 75
    reading = state["reading"]
    body = "stub: nothing is answered yet"
    if reading.kind == "cant_check":
        body = cant_check_answer(reading)
    answer = read_back(reading) + "\n" + body
    return {"visited": ["n13_final_output"], "answer": answer}


AGENTS = (
    n11a_club_operations,
    n11b_tactical_analysis,
    n11c_personal_performance,
    n11d_global_scouting,
    n11e_fan_concierge,
    n11f_stadium_operations,
    n11g_bowling_performance,
    n11h_content_production,
    n11i_match_officials,
    n11j_talent_development,
    n11k_injury_management,
    n11l_player_representation,
    n11m_explainer,
)

EVERY_NODE = (
    n1_video_input,
    n2_n5_shot_classifier,
    confidence_gate,
    n2b_pose_estimator,
    n3_question_reader,
    n6_shot_dna,
    n7a_evidence_retrieval,
    n7b_context_retrieval,
    n8_descriptive_engine,
    n9_predictive_engine,
    n10_supervisor_router,
    *AGENTS,
    faith_gate,
    n12_report_synthesis,
    n13_final_output,
)
