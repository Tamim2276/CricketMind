"""The Faith Check: what it catches, what it withholds, and where it stops."""
import pytest

from cricketmind import (AGENT_REGISTRY, CLIP, MAX_REGENS, QUERY, Settings,
                         build_cricketmind_graph, new_state, sanctioned_facts)
from cricketmind.experiments.boundary import R3_AGENT, run_boundary_cases

LIAR = "n11b_tactical_analysis"


@pytest.fixture(scope="module")
def graph():
    return build_cricketmind_graph()


def test_honest_run_passes_first_time(graph):
    final = graph.invoke(new_state(CLIP, QUERY))
    assert final["faith_failures"] == []
    assert final["regen_count"] == 0
    assert final["status"] == "ok"


def test_every_cited_figure_came_from_an_engine(graph):
    final = graph.invoke(new_state(CLIP, QUERY))
    facts = sanctioned_facts(final)
    for f in final["agent_findings"]:
        for key, value in f["cites"].items():
            assert key in facts, f"{f['agent']} cited an unsanctioned key {key!r}"
            assert facts[key] == value, f"{f['agent']} cited {key}={value}, engines say {facts[key]}"


def test_a_fabrication_is_caught_and_corrected(graph):
    """One regeneration, and the fabricated value never reaches the report."""
    final = graph.invoke(new_state(CLIP, QUERY, Settings(lie_once={LIAR})))
    assert final["regen_count"] == 1
    assert final["faith_failures"] == []        # corrected on the second attempt
    assert final["status"] == "ok"
    assert "999.0" not in final["report"]


def test_a_persistent_liar_is_withheld(graph):
    """The budget is spent, the finding is dropped, and the report says so."""
    final = graph.invoke(new_state(CLIP, QUERY, Settings(lie_always={LIAR})))
    assert final["regen_count"] == MAX_REGENS
    assert final["status"] == "unverified_claims_dropped"
    assert "999.0" not in final["report"]
    assert "WITHHELD" in final["report"]

    rejected = [f for f in final["agent_findings"] if f.get("rejected")]
    assert {f["agent"] for f in rejected} == {LIAR}


def test_regeneration_re_runs_only_the_failing_agent(graph):
    from cricketmind.experiments.profile import profile, record_run
    counts = profile(record_run(new_state(CLIP, QUERY, Settings(lie_once={LIAR}))))
    honest = profile(record_run(new_state(CLIP, QUERY)))
    assert counts["agent_calls"] == honest["agent_calls"] + 1
    assert counts["regens"] == 1


def test_no_fabrication_survives_even_when_everyone_lies(graph):
    everyone = {a["id"] for a in AGENT_REGISTRY}
    final = graph.invoke(new_state(CLIP, QUERY, Settings(lie_always=everyone)))
    assert "999.0" not in final["report"]
    assert final["status"] == "unverified_claims_dropped"


# --- the boundary (R3) ------------------------------------------------------

def test_boundary_cases():
    """A fabricated citation is caught; a swapped subject and an uncited
    figure both pass and reach the user. That is the gate's boundary."""
    by_case = {r["case"]: r for r in run_boundary_cases()}

    assert not by_case["honest"]["rejected"]
    assert by_case["honest"]["claim_in_report"]

    assert by_case["fabricated"]["rejected"]
    assert not by_case["fabricated"]["claim_in_report"]

    assert not by_case["misattributed"]["rejected"]
    assert by_case["misattributed"]["claim_in_report"], "the swapped claim should reach the user"

    assert not by_case["uncited"]["rejected"]
    assert by_case["uncited"]["claim_in_report"], "the uncited figure should reach the user"

    # the misattributed answer cites exactly what the honest one cites
    assert by_case["misattributed"]["cites"] == by_case["honest"]["cites"]


def test_boundary_overrides_do_not_leak(graph):
    """Settings.template_overrides is per-run: the registry is never mutated."""
    from cricketmind import AGENTS_BY_ID
    from cricketmind.experiments.boundary import R3_HONEST
    run_boundary_cases()
    assert AGENTS_BY_ID[R3_AGENT]["template"] == R3_HONEST
