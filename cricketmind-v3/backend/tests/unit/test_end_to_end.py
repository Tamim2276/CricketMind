"""Day 77: six text-only questions through the whole graph, with fakes.

Each is a kind of question, from the read-back to the receipts. The numbers and the
model's words are made up (tests/world.py): this is how the graph behaves, not how well
any model writes. Day 78 runs a real model, once the exam is locked.
"""

from cricketmind_v3.application.ports import FakeFactSource
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.report import BY_MODEL, EXPLANATION_LABEL, ONE_FINDING
from cricketmind_v3.domain.routing import NOBODY
from cricketmind_v3.domain.subject import CricketSubject
from tests.world import (
    ABOVE,
    KOHLI,
    ask,
    asked_of,
    engines,
    model,
    reading,
    words,
)

AHEAD = CricketSubject(player="V Kohli", match_format="T20", k=6)
WIDE = Fact(value=0.41, n=50, interval=(0.15, 0.7))


def order_of(answer: str, *parts: str) -> list[int]:
    places = []
    for part in parts:
        assert part in answer, part
        places.append(answer.index(part))
    return places


def test_scenario_1_a_stats_question_two_agents_joined():
    source = engines(("sr_vs_average", KOHLI, ABOVE))
    llm = model(
        {
            "Tactical Analysis": [words("Kohli returns {c1}.")],
            "Personal Performance": [words("Kohli is strong against the rest: {c2}.")],
        },
        report=[words("Kohli returns {c1}, and is strong against the rest: {c2}.")],
    )
    state = ask(llm, source=source)

    assert state["reading"].kind == "stats"
    assert state["routing"].chosen == ("N11b", "N11c")
    assert len(state["verified"]) == 2
    assert state["report"].by == BY_MODEL
    assert state["answer"].splitlines() == [
        "I read this as: a question about what happened: V Kohli, T20.",
        (
            "Kohli returns a strike rate of 135.1 [D, 9841 balls], and is strong against "
            "the rest: a strike-rate difference of +12.0 from the average batter "
            "[D, 9841 balls, range +8.0 to +16.0]."
        ),
        "Receipts:",
        '- N8.M1 version 1, data a1b2c3d4e5f6: {"metric":"strike_rate","number":1}',
        '- N8.M1 version 1, data a1b2c3d4e5f6: {"metric":"sr_vs_average","number":2}',
    ]


def test_scenario_2_a_prediction_with_a_wide_range():
    # the first try overstates it; the gate says why; the second fits the band
    source = engines(("p_dismissal_within_k", AHEAD, WIDE))
    llm = model(
        {
            "Tactical Analysis": [
                words("Kohli will likely be out within six balls: {c2}."),
                words(
                    "It is about even that Kohli is out within six balls: {c2}, "
                    "though the estimate is very uncertain."
                ),
            ]
        },
        chosen=("N11b",),
        read=reading(kind="prediction", k=6),
    )
    state = ask(llm, source=source, question="Kohli out within 6 balls in T20s?")

    (first, second) = asked_of(llm, "Tactical Analysis")
    assert "Words that fit this chance: about even" in first[1]
    assert "the estimate is very uncertain" in first[1]
    assert "'will' is too strong" in second[1]
    assert state["checks_run"] == 2
    assert state["withheld"] == ()
    assert state["report"].by == ONE_FINDING
    assert (
        "It is about even that Kohli is out within six balls: a 41% chance "
        "[P, range 15% to 70%], though the estimate is very uncertain."
    ) in state["answer"]


def test_scenario_3_a_rules_question():
    source = FakeFactSource()
    llm = model(
        {"Explainer": [words("LBW means leg before wicket.")]},
        read=reading(kind="rules", player=None, match_format=None),
    )
    state = ask(llm, source=source, question="What is LBW?")

    assert state.get("answers", []) == []  # a rule asks the engines nothing
    assert state["capable"] == {"N11m": ()}
    assert state["routing"].chosen == ("N11m",)
    assert state["answer"].splitlines() == [
        "I read this as: a question about the rules.",
        f"{EXPLANATION_LABEL}: LBW means leg before wicket.",
    ]
    assert (
        asked_of(llm, "You send cricket questions") == []
    )  # one agent: nothing to ask


def test_scenario_4_a_question_nobody_can_check():
    source = FakeFactSource()
    llm = model({}, read=reading(kind="cant_check"))
    state = ask(llm, source=source, question="Is Kohli injured?")

    assert state["visited"] == ["n3_question_reader", "n13_final_output"]
    assert len(llm.asked) == 1  # only N3
    assert "I can't check that" in state["answer"]
    assert "You could ask instead: V Kohli's record" in state["answer"]


def test_scenario_5_one_agent_fixes_its_lie_and_one_never_does():
    llm = model(
        {
            "Tactical Analysis": [
                words("Kohli returns {c1}, near 150."),
                words("Kohli returns {c1}."),
            ],
            "Personal Performance": [words("Kohli is outstanding: {c1}.")] * 5,
        },
    )
    state = ask(llm)

    assert state["visited"].count("n11b_tactical_analysis") == 2
    assert state["visited"].count("n11c_personal_performance") == 3
    assert state["checks_run"] == 3
    assert state["report"].by == ONE_FINDING
    lines = state["answer"].splitlines()
    assert lines[1] == (
        "Tactical Analysis: Kohli returns a strike rate of 135.1 [D, 9841 balls]."
    )
    assert lines[2] == "Withheld: Personal Performance: its words did not pass C7."
    # nothing unchecked reached the reader
    assert "150" not in state["answer"]
    assert "outstanding" not in state["answer"]


def test_scenario_6_nobody_can_answer():
    source = FakeFactSource()
    llm = model({}, read=reading(player=None, bowler="Bumrah", match_format="T20"))
    state = ask(llm, source=source, question="Is Bumrah overworked in T20s?")

    assert state["routing"].how == NOBODY
    assert state["routing"].no_number == ("N11k",)
    for name in state["visited"]:
        assert not name.startswith("n11")
    (read, nothing, why, missing) = state["answer"].splitlines()
    assert (
        read
        == "I read this as: a question about what happened: the bowler JJ Bumrah, T20."
    )
    assert nothing.startswith("Nothing is answered")
    assert why.startswith("Why:")
    assert missing == (
        "Injury Management could answer with a date, and without a format."
    )


def test_an_answer_keeps_its_parts_in_order():
    # the read-back, the judgment, the report, what was missing, what was held, receipts
    source = engines(("sr_vs_average", KOHLI, ABOVE))
    llm = model(
        {
            "Tactical Analysis": [words("Kohli returns {c1}.")],
            "Personal Performance": [words("Kohli is outstanding: {c1}.")] * 3,
        },
    )
    state = ask(llm, source=source, question="Why is Kohli so bad in T20s?")
    places = order_of(
        state["answer"],
        "I read this as:",
        'On "bad":',
        "Tactical Analysis:",
        "Withheld:",
        "Receipts:",
    )
    assert places == sorted(places)
