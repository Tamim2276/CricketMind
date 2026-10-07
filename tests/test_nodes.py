"""The node bodies: perception, retrieval, and the two sanctioned engines."""
import pytest

from cricketmind import (CLIP, FakeTensor, NUM_FRAMES, QUERY, SHOT_CLASSES,
                         Settings, build_cricketmind_graph, new_state)
from cricketmind.corpus import FAKE_CRICSHEET
from cricketmind.nodes.analytics import n8_stats_engine, n9_whatif_simulator
from cricketmind.nodes.grounding import n7a_stats_retrieval
from cricketmind.nodes.perception import conf_gate, n3_text_encoder


@pytest.fixture(scope="module")
def run():
    graph = build_cricketmind_graph()
    return graph.invoke(new_state(CLIP, QUERY))


# --- N3 ---------------------------------------------------------------------

@pytest.mark.parametrize("question,expected", [
    ("What if Rohit faced the pace attack instead of Kohli?", "Rohit"),
    ("Compare Kohli's cover drive vs his pull shot", "Kohli"),
    ("How does Babar play spin compared with Root?", "Babar"),
    ("Tell me about this delivery", None),
])
def test_n3_takes_the_player_named_first(question, expected):
    """Checking the names in a fixed order used to always pick Kohli."""
    got = n3_text_encoder({"query": question})["query_filters"].get("player")
    assert got == expected


def test_n3_extracts_the_running_example(run):
    f = run["shot_dna"]["filters"]
    assert f["bowler_type"] == "pace"
    assert f["phase"] == "powerplay"
    assert f["player"] == "Kohli"
    assert set(f["shots"]) == {"Cover Drive", "Pull"}


# --- the confidence gate ----------------------------------------------------

@pytest.mark.parametrize("conf,tries,route", [
    (0.92, 0, "accept"),
    (0.40, 0, "retry"),
    (0.40, 1, "retry"),
    (0.40, 2, "exhausted"),
    (0.70, 0, "accept"),        # the threshold itself is accepted
])
def test_conf_gate_routes(conf, tries, route):
    assert conf_gate({"shot_confidence": conf, "retry_count": tries}) == route


# --- retrieval --------------------------------------------------------------

def test_n7a_filters_out_the_noise(run):
    hits = run["stats_hits"]
    assert len(hits) == 100
    assert {r["shot"] for r in hits} == {"Cover Drive", "Pull"}
    assert all(r["bowler_type"] == "pace" for r in hits), "spin noise leaked in"
    assert all(r["phase"] == "powerplay" for r in hits), "death-overs noise leaked in"
    assert len(hits) < len(FAKE_CRICSHEET)


def test_n7b_returns_context(run):
    assert len(run["context_hits"]) == 4


def test_n7a_refuses_a_text_query_naming_no_shot():
    state = {"shot_dna": {"filters": {"player": "Kohli"}, "shot": None}}
    with pytest.raises(AssertionError, match="nothing to retrieve"):
        n7a_stats_retrieval(state)


# --- the two engines --------------------------------------------------------

def test_n8_counts_the_rows_rather_than_hardcoding(run):
    by_shot = run["stats_results"]["by_shot"]
    assert by_shot["Cover Drive"]["strike_rate"] == 142.0
    assert by_shot["Pull"]["strike_rate"] == 112.0
    assert run["stats_results"]["best_shot"] == "Cover Drive"
    assert run["stats_results"]["worst_shot"] == "Pull"
    assert run["stats_results"]["sample_size"] == 100

    # recompute by hand from the retrieved rows
    runs_ = sum(r["runs"] for r in run["stats_hits"] if r["shot"] == "Pull")
    balls = sum(1 for r in run["stats_hits"] if r["shot"] == "Pull")
    assert round(100.0 * runs_ / balls, 1) == 112.0


def test_n9_agrees_with_the_closed_form(run):
    sim = run["sim_results"]
    lo, hi = sim["ci95"]
    assert sim["target_shot"] == "Pull"
    assert abs(sim["p_dismissal"] - sim["analytic_check"]) < 0.02
    assert lo <= sim["analytic_check"] <= hi


def test_engines_refuse_to_run_without_evidence():
    for engine in (n8_stats_engine, n9_whatif_simulator):
        with pytest.raises(AssertionError):
            engine({"stats_hits": []})


# --- shapes -----------------------------------------------------------------

def test_vision_shapes_line_up(run):
    assert run["frame_features"].shape == (NUM_FRAMES, 1280)
    assert run["temporal_features"].shape == (1280,)
    assert isinstance(run["frame_features"], FakeTensor)


def test_classifier_label_is_a_real_class(run):
    assert run["shot_label"] in SHOT_CLASSES


def test_classifier_rejects_an_unknown_label():
    graph = build_cricketmind_graph()
    with pytest.raises(Exception):
        graph.invoke(new_state(CLIP, QUERY, Settings(label="Helicopter Shot")))
