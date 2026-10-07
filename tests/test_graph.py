"""The compiled graph: its shape, its two entry paths, and both correction loops."""
import pytest

from cricketmind import (AGENT_REGISTRY, CLIP, MAX_ACTIVE_AGENTS, QUERY, Settings,
                         build_cricketmind_graph, new_state)
from cricketmind.experiments.profile import record_run


@pytest.fixture(scope="module")
def graph():
    return build_cricketmind_graph()


@pytest.fixture(scope="module")
def nodes(graph):
    return {n for n in graph.get_graph().nodes if not n.startswith("__")}


def test_thirty_nodes(nodes):
    """One per component of Fig. 1, plus the loop-control and withhold nodes."""
    assert len(nodes) == 30


def test_every_architecture_node_present(nodes):
    expected = {
        "n1_video_input", "n2_vision_encoder", "n3_text_encoder",
        "n4_temporal_fusion", "n5_classifier_head", "n6_shot_dna",
        "n7a_stats_retrieval", "n7b_context_retrieval",
        "n8_stats_engine", "n9_whatif_simulator", "n10_supervisor_router",
        "faith_check", "n12_supervisor_synthesis", "n13_final_output",
        "retry", "degrade", "regen_prep", "drop_unfaithful",
    } | {a["id"] for a in AGENT_REGISTRY}
    assert nodes == expected


def test_video_path_produces_a_report(graph):
    final = graph.invoke(new_state(CLIP, QUERY))
    assert final["status"] == "ok"
    assert "CRICKETMIND REPORT" in final["report"]
    assert final["shot_dna"]["modality"] == "video"


def test_text_only_path_skips_the_vision_branch(graph):
    final = graph.invoke(new_state(None, QUERY))
    assert final["status"] == "ok"
    assert final["shot_dna"]["modality"] == "text"
    assert final["shot_dna"]["posture"] is None, "posture must not be invented"
    assert "no clip supplied" in final["report"]


def test_text_only_does_not_wake_posture_agents(graph):
    """An agent whose cited facts do not exist for this query is removed."""
    from cricketmind import POSTURE_AGENTS
    final = graph.invoke(new_state(None, QUERY))
    assert not set(final["active_agents"]) & POSTURE_AGENTS


def test_n3_runs_on_both_paths(graph):
    for clip in (CLIP, None):
        final = graph.invoke(new_state(clip, QUERY))
        assert final["query_embedding"] is not None, f"N3 skipped for clip={clip}"


def test_degrade_is_terminal(graph):
    """A clip that exhausts the retry budget produces no report at all."""
    final = graph.invoke(new_state(CLIP, QUERY, Settings(confidence=0.30)))
    assert final["status"] == "low_confidence"
    assert final.get("report") is None
    assert final.get("shot_dna") is None, "N6 must not run when the gate refused"

    ran = {node for step, node in record_run(new_state(CLIP, QUERY, Settings(confidence=0.30)))}
    assert "degrade" in ran
    assert not ran & {"n12_supervisor_synthesis", "n13_final_output"}


def test_retry_loop_recovers(graph):
    final = graph.invoke(new_state(CLIP, QUERY, Settings(confidence=(0.41, 0.55, 0.88))))
    assert final["status"] == "ok"
    assert final["retry_count"] == 2
    assert final["shot_dna"]["confidence"] == 0.88


def test_router_never_wakes_everyone(graph):
    everything = ("budget auction squad tactic bowling technique shot scout league "
                  "fan fantasy pitch prep crowd sponsor logo roi commentary overlay "
                  "eligibility icc academy drill injury workload valuation market")
    final = graph.invoke(new_state(CLIP, everything))
    assert len(final["active_agents"]) == MAX_ACTIVE_AGENTS
    assert len(final["active_agents"]) < len(AGENT_REGISTRY)


def test_only_routed_agents_run(graph):
    final = graph.invoke(new_state(CLIP, QUERY))
    ran = {f["agent"] for f in final["agent_findings"]}
    assert ran == set(final["active_agents"])
    assert len(ran) < len(AGENT_REGISTRY)
