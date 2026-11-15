"""Day 79: the worst case is a sum, and a run with fakes reaches it.

The fakes make every part fail as often as it can: the reader's first reply does not fit,
the router's neither, and four agents lie every time."""

import json

from cricketmind_v3.application.graph.build import build_graph
from cricketmind_v3.application.graph.nodes import AGENTS
from cricketmind_v3.application.ports import FakeLLM, FakeLLMByWho, LLMError
from cricketmind_v3.application.resolver import LOOK_UPS, find_player
from cricketmind_v3.domain import cost
from cricketmind_v3.domain.faith.gate import MAX_REGENERATIONS
from cricketmind_v3.domain.names import Clues, Player
from cricketmind_v3.domain.roster import NAMES, ROSTER
from cricketmind_v3.domain.routing import MAX_ACTIVE_AGENTS
from tests.world import (
    QUESTION,
    READER,
    REPORTER,
    ROUTER,
    asked_of,
    context_of,
    reading,
    routed,
    words,
)

LIE = "Kohli returns {c1}, close to 150."


def four_who_can_say_strike_rate() -> list[str]:
    found = []
    for agent, measures in ROSTER.items():
        if "strike_rate" in measures:
            found.append(agent)
    return found


def worst_llm(chosen: list[str]) -> FakeLLMByWho:
    replies = {
        READER: ["not json", reading()],
        ROUTER: ["not json", routed(*chosen)],
        REPORTER: [words("never reached")],
    }
    for agent in chosen:
        replies[NAMES[agent]] = [words(LIE)] * 10
    return FakeLLMByWho(replies)


def supersteps(context) -> int:
    # each task the graph starts says which round it is in
    steps = set()
    stream = build_graph().stream(
        {"question": QUESTION, "clip": None}, context=context, stream_mode="debug"
    )
    for event in stream:
        steps.add(event["step"])
    return len(steps)


def test_the_bound_is_the_sum_in_the_invariant():
    # I6: 4 x (1 + 2), however many agents the roster has
    assert cost.agent_runs() == 12
    assert len(AGENTS) == 13
    assert cost.text_supersteps() == 13
    assert cost.model_calls() == 26


def test_the_bound_follows_its_parts():
    assert cost.agent_runs(agents=2, regenerations=1) == 4
    assert cost.text_supersteps(regenerations=0) == 9


def test_the_parts_are_the_real_limits():
    assert cost.LOOK_UPS_FOR_A_NAME == LOOK_UPS
    assert MAX_ACTIVE_AGENTS == 4
    assert MAX_REGENERATIONS == 2


def test_the_worst_case_reaches_twelve_agent_runs():
    chosen = four_who_can_say_strike_rate()[:4]
    context = context_of(worst_llm(chosen))
    state = build_graph().invoke({"question": QUESTION, "clip": None}, context=context)
    runs = 0
    for name in state["visited"]:
        if name.startswith("n11"):
            runs += 1
    assert runs == cost.agent_runs()
    assert len(state["withheld"]) == 4


def test_the_worst_case_reaches_the_superstep_bound():
    chosen = four_who_can_say_strike_rate()[:4]
    assert supersteps(context_of(worst_llm(chosen))) == cost.text_supersteps()


def test_the_worst_case_keeps_to_the_call_bound():
    chosen = four_who_can_say_strike_rate()[:4]
    llm = worst_llm(chosen)
    build_graph().invoke({"question": QUESTION, "clip": None}, context=context_of(llm))
    # the reader twice, the router twice, the agents twelve times; no name needed asking,
    # and nothing passed, so N12 had nothing to join
    assert len(asked_of(llm, READER)) == cost.READER_CALLS
    assert len(asked_of(llm, ROUTER)) == cost.ROUTER_CALLS
    assert len(llm.asked) == cost.READER_CALLS + cost.ROUTER_CALLS + cost.agent_runs()
    assert len(llm.asked) <= cost.model_calls()


def test_more_agents_on_the_roster_cost_nothing_more():
    # the bound does not count the roster: more names asked for, the same four run
    everyone = four_who_can_say_strike_rate()
    assert len(everyone) > 4
    llm = worst_llm(everyone)
    state = build_graph().invoke(
        {"question": QUESTION, "clip": None}, context=context_of(llm)
    )
    assert len(state["routing"].chosen) == 4
    assert len(asked_of(llm, ROUTER)) == 2
    assert len(llm.asked) == 2 + 2 + cost.agent_runs()


def test_n12_adds_its_one_call_when_there_is_something_to_join():
    llm = FakeLLMByWho(
        {
            READER: [reading()],
            ROUTER: [routed("N11b", "N11c")],
            REPORTER: [LLMError("no answer"), words("never asked")],
            NAMES["N11b"]: [words("Kohli returns {c1}.")],
            NAMES["N11c"]: [words("Kohli returns {c1}.")],
        }
    )
    build_graph().invoke({"question": QUESTION, "clip": None}, context=context_of(llm))
    assert len(asked_of(llm, REPORTER)) == cost.REPORT_CALLS


def test_a_name_costs_three_look_ups_at_most():
    # the model keeps offering a fuller name: the search still stops at three
    players = (
        Player(person_id="a", name="A Smith", matches=10),
        Player(person_id="b", name="B Smith", matches=10),
    )
    steps = []
    for _ in range(5):
        steps.append(json.dumps({"search": "Smith", "choose": None, "clue": None}))
    llm = FakeLLM(steps)
    find_player("player", "Smith", "Is Smith out?", Clues(), players, llm)
    assert len(llm.asked) <= cost.LOOK_UPS_FOR_A_NAME


def test_a_plain_question_costs_far_less_than_the_worst():
    llm = FakeLLMByWho(
        {
            READER: [reading()],
            ROUTER: [routed("N11b")],
            NAMES["N11b"]: [words("Kohli returns {c1}.")],
        }
    )
    context = context_of(llm)
    state = build_graph().invoke({"question": QUESTION, "clip": None}, context=context)
    assert state["checks_run"] == 1
    assert len(llm.asked) == 3  # N3, N10, one agent; one finding needs no joining
    assert supersteps(context_of(_again())) == cost.text_supersteps(regenerations=0)


def _again() -> FakeLLMByWho:
    return FakeLLMByWho(
        {
            READER: [reading()],
            ROUTER: [routed("N11b")],
            NAMES["N11b"]: [words("Kohli returns {c1}.")],
        }
    )
