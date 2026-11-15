"""Every node of the v4 graph (Day 55): each signs the notebook, and the stubs do nothing
else. The days that follow make them real, one at a time: N3 on Day 56. A node's name is
its function's name."""

from langgraph.runtime import Runtime

from cricketmind_v3.application.agent_factory import write_finding
from cricketmind_v3.application.fact_surface import ask_engines
from cricketmind_v3.application.faith_check import check_with_receipts
from cricketmind_v3.application.graph.context import GraphContext
from cricketmind_v3.application.graph.state import GraphState
from cricketmind_v3.application.reader import read_question
from cricketmind_v3.application.retrieval import retrieve
from cricketmind_v3.application.router import route
from cricketmind_v3.application.synthesis import write_report
from cricketmind_v3.domain.answer import Parts, final_answer
from cricketmind_v3.domain.capabilities import capable_agents
from cricketmind_v3.domain.faith.gate import (
    MAX_REGENERATIONS,
    Context,
    withheld_from,
)
from cricketmind_v3.domain.findings import latest_findings
from cricketmind_v3.domain.premise_check import ALLOWED, premise_finding
from cricketmind_v3.domain.question_context import question_context
from cricketmind_v3.domain.report import in_routing_order
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
    # the rules decide. Day 73: only those run
    context = given(runtime)
    capable = capable_agents(
        state["reading"].kind, state.get("answers", []), state["shot_dna"]
    )
    routing = route(state["question"], capable, state["evidence"], context.llm)
    written = {
        "visited": ["n10_supervisor_router"],
        "capable": capable,
        "routing": routing,
    }
    # Day 76: a judgment in the question is tested, whoever answers. N3 marked it; this is
    # the first node with the numbers to test it
    premise = state["reading"].premise
    if premise is not None:
        tested = premise_finding(premise, state.get("answers", []))
        if tested is not None:
            written["findings"] = [tested]
    return written


def run_agent(agent: str, node: str, state: GraphState, context: GraphContext) -> dict:
    # Days 66-72: one try at this agent's finding. The model writes the words, and the
    # code fills in every number from the page N8 and N9 wrote. Day 74: sent back by the
    # gate, it hears why
    mine = []
    for failure in state.get("failures", ()):
        if failure.agent == agent:
            mine.append(failure)
    finding = write_finding(
        agent,
        state["question"],
        state.get("answers", []),
        context.llm,
        context.specs.get(agent),
        context.rules,
        failures=tuple(mine),
        shown=dict(state["reading"].shown),
        premise=state["reading"].premise,
    )
    return {"visited": [node], "findings": [finding]}


def n11a_club_operations(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    return run_agent("N11a", "n11a_club_operations", state, given(runtime))


def n11b_tactical_analysis(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    return run_agent("N11b", "n11b_tactical_analysis", state, given(runtime))


def n11c_personal_performance(
    state: GraphState, runtime: Runtime[GraphContext]
) -> dict:
    return run_agent("N11c", "n11c_personal_performance", state, given(runtime))


def n11d_global_scouting(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    return run_agent("N11d", "n11d_global_scouting", state, given(runtime))


def n11e_fan_concierge(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    return run_agent("N11e", "n11e_fan_concierge", state, given(runtime))


def n11f_stadium_operations(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    return run_agent("N11f", "n11f_stadium_operations", state, given(runtime))


def n11g_bowling_performance(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    return run_agent("N11g", "n11g_bowling_performance", state, given(runtime))


def n11h_content_production(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    return run_agent("N11h", "n11h_content_production", state, given(runtime))


def n11i_match_officials(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    return run_agent("N11i", "n11i_match_officials", state, given(runtime))


def n11j_talent_development(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    return run_agent("N11j", "n11j_talent_development", state, given(runtime))


def n11k_injury_management(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    return run_agent("N11k", "n11k_injury_management", state, given(runtime))


def n11l_player_representation(
    state: GraphState, runtime: Runtime[GraphContext]
) -> dict:
    return run_agent("N11l", "n11l_player_representation", state, given(runtime))


def n11m_explainer(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    return run_agent("N11m", "n11m_explainer", state, given(runtime))


# G. the checks, the report, the answer


def faith_gate(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    # Day 61: it asks the engines again and keeps the receipts. Day 74: the seven checks
    # on each agent's last try. An agent that passed is never checked again
    context = given(runtime)
    gate = Context(allowed=ALLOWED, names=context.gate_names or context.names)
    verified = list(state.get("verified", ()))
    passed = []
    for finding in verified:
        passed.append(finding.agent)
    receipts = dict(state.get("receipts", {}))
    failures = []
    for finding in latest_findings(state.get("findings", [])):
        if finding.agent in passed:
            continue
        wrong, theirs = check_with_receipts(finding, context.engines, gate)
        receipts.update(theirs)
        if wrong:
            failures.extend(wrong)
        else:
            verified.append(finding)
    checks_run = state.get("checks_run", 0) + 1
    withheld = ()
    if checks_run > MAX_REGENERATIONS:
        withheld = withheld_from(failures)
    return {
        "visited": ["faith_gate"],
        "receipts": receipts,
        "failures": tuple(failures),
        "verified": tuple(verified),
        "withheld": withheld,
        "checks_run": checks_run,
    }


def n12_report_synthesis(state: GraphState, runtime: Runtime[GraphContext]) -> dict:
    # Day 75: only what passed, in the router's order; the model joins, the code checks
    context = given(runtime)
    findings = in_routing_order(state.get("verified", ()), state["routing"].chosen)
    names = context.gate_names or context.names
    report = write_report(state["question"], findings, context.llm, names)
    return {"visited": ["n12_report_synthesis"], "report": report}


def n13_final_output(state: GraphState) -> dict:
    # every answer starts with what was read (Day 59); then what passed, with its tags;
    # what was held back; what was missing; and the receipts (Day 75)
    parts = Parts(
        reading=state["reading"],
        routing=state.get("routing"),
        report=state.get("report"),
        withheld=state.get("withheld", ()),
        verified=state.get("verified", ()),
        receipts=state.get("receipts", {}),
        why_no_numbers=why_no_numbers(state),
    )
    return {"visited": ["n13_final_output"], "answer": final_answer(parts)}


def why_no_numbers(state: GraphState) -> tuple[str, ...]:
    # each different reason the evidence gave, and the clip's
    reasons = []
    dna = state.get("shot_dna")
    if dna is not None and dna.clip is not None and dna.empty_because is not None:
        reasons.append(dna.empty_because)
    if state["reading"].kind == "video" and state.get("clip") is None:
        reasons.append("the question is about a clip, and no clip came with it")
    for found in state.get("evidence", ()):
        if found.none_because is not None and found.none_because not in reasons:
            reasons.append(found.none_because)
    return tuple(reasons)


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

# "N11b" -> "n11b_tactical_analysis": who the router means, as a node (Day 73)
NODE_OF = {}
for agent_node in AGENTS:
    NODE_OF[agent_node.__name__.split("_")[0].capitalize()] = agent_node.__name__

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
