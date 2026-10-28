"""Every node of the v4 graph as a stub (Day 55): each one only signs the notebook. The
days that follow make them real, one at a time. A node's name is its function's name."""

from cricketmind_v3.application.graph.state import GraphState

# A. the clip: its frames, the shot, how sure, the body


def n1_video_input(state: GraphState) -> dict:
    return {"visited": ["n1_video_input"]}


def n2_n5_shot_classifier(state: GraphState) -> dict:
    # N2 to N5 in v4: one exported model from the CricShot study
    return {"visited": ["n2_n5_shot_classifier"]}


def confidence_gate(state: GraphState) -> dict:
    return {"visited": ["confidence_gate"]}


def n2b_pose_estimator(state: GraphState) -> dict:
    return {"visited": ["n2b_pose_estimator"]}


# B. the question, and the shot it is about


def n3_question_reader(state: GraphState) -> dict:
    return {"visited": ["n3_question_reader"]}


def n6_shot_dna(state: GraphState) -> dict:
    return {"visited": ["n6_shot_dna"]}


# C. the evidence, and D. the engines


def n7a_evidence_retrieval(state: GraphState) -> dict:
    return {"visited": ["n7a_evidence_retrieval"]}


def n7b_context_retrieval(state: GraphState) -> dict:
    return {"visited": ["n7b_context_retrieval"]}


def n8_descriptive_engine(state: GraphState) -> dict:
    return {"visited": ["n8_descriptive_engine"]}


def n9_predictive_engine(state: GraphState) -> dict:
    return {"visited": ["n9_predictive_engine"]}


# E. the router, and F. the twelve agents and the Explainer


def n10_supervisor_router(state: GraphState) -> dict:
    return {"visited": ["n10_supervisor_router"]}


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


def faith_gate(state: GraphState) -> dict:
    return {"visited": ["faith_gate"]}


def n12_report_synthesis(state: GraphState) -> dict:
    return {"visited": ["n12_report_synthesis"]}


def n13_final_output(state: GraphState) -> dict:
    return {"visited": ["n13_final_output"], "answer": "stub: nothing is answered yet"}


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
