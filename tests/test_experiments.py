"""The measured results the paper reports, and the isolation property that
makes them safe to re-run in any order.
"""
import pytest

from cricketmind import (AGENTS_BY_ID, CLIP, MAX_ACTIVE_AGENTS, QUERY, Settings,
                         new_state)
from cricketmind.experiments.profile import (EXPECTED_COSTS, SCENARIOS,
                                             WORST_AGENT_CALLS, WORST_STEPS,
                                             predicted_steps, profile,
                                             record_run, run_all_scenarios)
from cricketmind.experiments.routing_eval import (POSTURE_AGENTS, ROUTING_PROMPTS,
                                                  route_all)
from cricketmind.nodes.perception import n3_text_encoder


@pytest.fixture(scope="module")
def cost_rows():
    return run_all_scenarios()


# --- the execution profile --------------------------------------------------

def test_every_scenario_costs_what_it_cost_before(cost_rows):
    for r in cost_rows:
        got = (r["steps"], r["node_runs"], r["agent_calls"])
        assert got == EXPECTED_COSTS[r["scenario"]], r["scenario"]


def test_the_formula_predicts_every_measured_run(cost_rows):
    for sc, r in zip(SCENARIOS, cost_rows):
        formula = predicted_steps(has_clip=sc["clip"] is not None,
                                  retries=r["retries"], regens=r["regens"],
                                  dropped=r["dropped"], degraded=r["degraded"])
        assert formula == r["steps"], f"{r['scenario']}: {r['steps']} != {formula}"


def test_bounds_hold_and_the_worst_case_reaches_them(cost_rows):
    assert WORST_STEPS == 29 and WORST_AGENT_CALLS == 12
    assert all(r["steps"] <= WORST_STEPS for r in cost_rows)
    assert all(r["agent_calls"] <= WORST_AGENT_CALLS for r in cost_rows)
    assert max(r["steps"] for r in cost_rows) == WORST_STEPS


# --- the property the refactor bought ---------------------------------------

def test_scenarios_do_not_leak_into_each_other():
    """Running a scenario must not change what the next one measures.

    This used to depend on a try/finally around four module-level globals.
    Settings now travel in the state, so it holds by construction -- running
    the scenarios in reverse must give the same answers.
    """
    forward = {r["scenario"]: profile_of(r) for r in run_all_scenarios()}
    backward = {}
    for sc in reversed(SCENARIOS):
        timeline = record_run(new_state(
            sc["clip"], QUERY,
            Settings(confidence=sc["confidence"],
                     lie_once=sc.get("lie_once", ()),
                     lie_always=sc.get("lie_always", ()))))
        backward[sc["name"]] = profile_of({"scenario": sc["name"], **profile(timeline)})
    assert forward == backward


def profile_of(row):
    return (row["steps"], row["node_runs"], row["agent_calls"])


def test_a_crashing_run_cannot_poison_the_next_one():
    """`confidence` of the wrong type crashes N5; the next run is unaffected."""
    with pytest.raises(Exception):
        record_run(new_state(CLIP, QUERY, Settings(confidence="not a number")))
    clean = profile(record_run(new_state(CLIP, QUERY)))
    assert (clean["steps"], clean["agent_calls"]) == (12, 2)


# --- the prompt set ---------------------------------------------------------

def test_prompt_set_is_built_as_the_paper_describes():
    styles = [p["style"] for p in ROUTING_PROMPTS]
    assert len(ROUTING_PROMPTS) == 29
    assert styles.count("keyword") == styles.count("plain") == 12
    assert {p["target"] for p in ROUTING_PROMPTS if p["target"]} == set(AGENTS_BY_ID)


def test_keyword_prompts_use_their_trigger_words_and_plain_ones_do_not():
    for p in ROUTING_PROMPTS:
        if not p["target"]:
            continue
        q = p["prompt"].lower()
        found = [k for k in AGENTS_BY_ID[p["target"]]["keywords"] if k in q]
        if p["style"] == "keyword":
            assert found, f"{p['id']}: a keyword prompt must use the target's keywords"
        else:
            assert not found, f"{p['id']}: a plain prompt must not use {found}"


def test_every_prompt_is_answerable_with_the_corpus():
    for p in ROUTING_PROMPTS:
        parsed = n3_text_encoder({"query": p["prompt"]})["query_filters"]
        assert parsed.get("player") in (None, "Kohli"), p["id"]
        if p["clip"] is None:
            assert parsed.get("shots"), f"{p['id']}: a text-only prompt must name a shot"


# --- the keyword router on all 29 -------------------------------------------

@pytest.fixture(scope="module")
def keyword_rows():
    return route_all("keyword")


def test_keyword_router_wakes_every_target_on_its_own_prompt(keyword_rows):
    hits = sum(r["style"] == "keyword" and r["target_woken"] == "signal"
               for r in keyword_rows)
    assert hits == 12


def test_keyword_router_misses_every_plain_prompt(keyword_rows):
    """The result the LLM router is compared against."""
    hits = sum(r["style"] == "plain" and r["target_woken"] == "signal"
               for r in keyword_rows)
    assert hits == 0


def test_the_rules_hold_on_all_29_prompts(keyword_rows):
    for r in keyword_rows:
        assert r["status"] == "ok", r["id"]
        assert 1 <= len(r["woken"]) <= MAX_ACTIVE_AGENTS, r["id"]
        if r["clip"] is None:
            assert not set(r["woken"]) & POSTURE_AGENTS, r["id"]


def test_two_questions_take_different_routes_through_one_pipeline():
    from cricketmind.experiments.routing_eval import record_flows
    flows = record_flows()
    woken = {label: next(u["active_agents"] for n, u in w
                         if n == "n10_supervisor_router")
             for label, w in flows.items()}
    hits = {label: next(len(u["stats_hits"]) for n, u in w
                        if n == "n7a_stats_retrieval")
            for label, w in flows.items()}
    assert not set(woken["A"]) & set(woken["B"]), "should wake different experts"
    assert hits["A"] != hits["B"], "should retrieve different evidence"
