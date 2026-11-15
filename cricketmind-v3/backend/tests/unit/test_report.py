"""Day 75: N12 joins what passed. The model writes, the code checks, and anything wrong
goes out as the findings themselves, in the router's order."""

import json

import pytest

from cricketmind_v3.application.ports import FakeLLM, LLMError
from cricketmind_v3.application.synthesis import ReportForm, write_report
from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.report import (
    BY_MODEL,
    EXPLANATION_LABEL,
    FIXED_ORDER,
    NOTHING,
    ONE_FINDING,
    fixed_order,
    in_routing_order,
    joined_findings,
    one_finding,
)
from cricketmind_v3.domain.subject import CricketSubject
from tests.world import (
    NAMES,
    QUESTION,
    ask,
    model,
    reading,
    words,
)

KOHLI = CricketSubject(player="V Kohli", match_format="T20")
AHEAD = CricketSubject(player="V Kohli", match_format="T20", k=6)
RATE = Claim(
    id="c1",
    kind=Provenance.DESCRIPTIVE,
    metric="strike_rate",
    subject=KOHLI,
    value=135.1,
    n=9841,
    source="N8.M1",
)
RUNS = Claim(
    id="c1",
    kind=Provenance.DESCRIPTIVE,
    metric="runs",
    subject=KOHLI,
    value=13295.0,
    n=9841,
    source="N8.M1",
)
FEW = Claim(
    id="c2",
    kind=Provenance.DESCRIPTIVE,
    metric="runs",
    subject=AHEAD,
    value=20.0,
    n=6,
    source="N8.M1",
)
CHANCE = Claim(
    id="c2",
    kind=Provenance.PREDICTIVE,
    metric="p_dismissal_within_k",
    subject=AHEAD,
    value=0.18,
    interval=(0.15, 0.21),
    source="N9.T1",
)
TACTICAL = Finding(agent="N11b", claims=(RATE,), template="Kohli returns {c1}.")
PERSONAL = Finding(agent="N11c", claims=(RUNS,), template="Kohli has {c1}.")
BOTH = [TACTICAL, PERSONAL]
GOOD = "Kohli returns {c1}, and has {c2}."


def report(findings, *replies):
    return write_report(QUESTION, findings, FakeLLM(list(replies)), NAMES)


def reply(template: str) -> str:
    return json.dumps({"template": template})


# What there is to report


def test_nothing_passed_gives_an_empty_report():
    made = report([])
    assert made.by == NOTHING
    assert made.text == ""


def test_one_finding_goes_out_as_it_stands_and_the_model_is_not_asked():
    llm = FakeLLM([reply("anything")])
    made = write_report(QUESTION, [TACTICAL], llm, NAMES)
    assert made.by == ONE_FINDING
    assert made.text == (
        "Tactical Analysis: Kohli returns a strike rate of 135.1 [D, 9841 balls]."
    )
    assert llm.asked == []


def test_an_explanation_always_carries_its_label():
    explained = Finding(
        agent="N11m", claims=(), template="LBW means leg before wicket."
    )
    made = report([explained])
    assert made.text == (f"{EXPLANATION_LABEL}: LBW means leg before wicket.")
    assert EXPLANATION_LABEL == "General explanation, not checked against data"
    assert one_finding(explained).startswith(EXPLANATION_LABEL)


def test_with_no_model_the_findings_go_out_in_order():
    made = write_report(QUESTION, BOTH, None, NAMES)
    assert made.by == FIXED_ORDER
    assert made.note is None
    assert made.text == fixed_order(BOTH)
    assert made.text == (
        "Tactical Analysis: Kohli returns a strike rate of 135.1 [D, 9841 balls].\n"
        "Personal Performance: Kohli has a total of 13295 runs [D, 9841 balls]."
    )


# One set of slots


def test_two_agents_c1_become_two_slots():
    joined = joined_findings(BOTH)
    assert joined.template == (
        "Tactical Analysis: Kohli returns {c1}.\nPersonal Performance: Kohli has {c2}."
    )
    assert joined.claims[0].metric == "strike_rate"
    assert joined.claims[1].metric == "runs"
    assert joined.claims[1].id == "c2"


def test_one_number_said_twice_is_one_slot():
    again = Finding(agent="N11d", claims=(RATE,), template="It is {c1}.")
    joined = joined_findings([TACTICAL, again])
    assert len(joined.claims) == 1
    assert (
        joined.template
        == "Tactical Analysis: Kohli returns {c1}.\nGlobal Scouting: It is {c1}."
    )


def test_renumbering_never_runs_into_itself():
    # {c1} -> {c2} and {c2} -> {c3}: the first must not then become {c3}
    second = Finding(
        agent="N11c",
        claims=(RUNS, FEW),
        template="{c2} before {c1}, then {c1}.",
    )
    joined = joined_findings([TACTICAL, second])
    assert joined.template.endswith(
        "Personal Performance: {c3} before {c2}, then {c2}."
    )


def test_a_joined_claim_keeps_every_part_of_the_number():
    chance = Finding(agent="N11b", claims=(CHANCE,), template="{c2}.")
    (claim,) = joined_findings([chance]).claims
    assert claim.id == "c1"
    assert claim.value == 0.18
    assert claim.interval == (0.15, 0.21)
    assert claim.source == "N9.T1"
    assert claim.kind == Provenance.PREDICTIVE


# The model joins


def test_the_model_joins_them_and_the_renderer_fills_the_numbers():
    made = report(BOTH, reply(GOOD))
    assert made.by == BY_MODEL
    assert made.note is None
    assert made.text == (
        "Kohli returns a strike rate of 135.1 [D, 9841 balls], and has a total of "
        "13295 runs [D, 9841 balls]."
    )


def test_the_form_has_only_the_words():
    assert list(ReportForm.model_fields) == ["template"]


def test_the_model_is_told_the_question_and_the_findings():
    llm = FakeLLM([reply(GOOD)])
    write_report(QUESTION, BOTH, llm, NAMES)
    (system, prompt, shape) = llm.asked[0]
    assert shape == "ReportForm"
    assert "type no figure" in system
    assert "Keep every warning" in system
    assert "only part of the question" in system
    assert QUESTION in prompt
    assert "Tactical Analysis: Kohli returns {c1}." in prompt


def test_the_model_gets_one_try():
    llm = FakeLLM([reply("It has {c1}."), reply(GOOD)])
    write_report(QUESTION, BOTH, llm, NAMES)
    assert len(llm.asked) == 1


# Anything wrong, and the findings go out as they are

BAD_REPORTS = [
    ("a typed number", "Kohli returns {c1}, close to 150, and has {c2}.", "C5"),
    ("praise with nothing behind it", "Kohli is strong: {c1}, {c2}.", "C7"),
    ("a slot it lost", "Kohli returns {c1}.", "lost c2"),
    ("a slot it made up", "Kohli returns {c1}, {c2}, {c9}.", "c9"),
    ("a slot never closed", "Kohli returns {c1}, {c2", "never closed"),
    ("a player nobody asked about", "Rohit returns {c1}, and {c2}.", "C6"),
]


@pytest.mark.parametrize(("what", "template", "said"), BAD_REPORTS)
def test_a_bad_report_falls_back(what, template, said):
    made = report(BOTH, reply(template))
    assert made.by == FIXED_ORDER, what
    assert said in made.note, what
    assert made.text == fixed_order(BOTH), what


def test_a_chance_called_certain_falls_back():
    chance = Finding(agent="N11b", claims=(CHANCE,), template="It is unlikely: {c2}.")
    both = [chance, PERSONAL]
    made = report(both, reply("He will be out: {c1}, and {c2}."))
    assert made.by == FIXED_ORDER
    assert made.note.startswith("C4")


def test_a_lost_warning_falls_back():
    # joining sentences can drop "on only a few balls": C3 is why N12 passes it too
    few = Finding(
        agent="N11c", claims=(FEW,), template="On only a few balls Kohli has {c2}."
    )
    made = report([TACTICAL, few], reply("Kohli returns {c1}, and has {c2}."))
    assert made.by == FIXED_ORDER
    assert made.note.startswith("C3")


def test_a_kept_warning_passes():
    few = Finding(
        agent="N11c", claims=(FEW,), template="On only a few balls Kohli has {c2}."
    )
    made = report(
        [TACTICAL, few], reply("Kohli returns {c1}, and on only a few balls has {c2}.")
    )
    assert made.by == BY_MODEL


def test_a_model_that_is_down_falls_back():
    made = report(BOTH, LLMError("Ollama could not answer"))
    assert made.by == FIXED_ORDER
    assert "Ollama could not answer" in made.note


def test_a_reply_that_does_not_fit_falls_back():
    made = report(BOTH, '{"wrong": 1}')
    assert made.by == FIXED_ORDER
    assert "did not fit" in made.note


# The order


def test_the_routers_order_whoever_finished_first():
    ordered = in_routing_order((PERSONAL, TACTICAL), ("N11b", "N11c"))
    assert ordered == [TACTICAL, PERSONAL]


def test_only_those_chosen_are_reported():
    # a finding of no chosen agent (N3's test of a judgment, Day 76) is not an agent's
    other = Finding(agent="N3", claims=(), template="words")
    assert in_routing_order((other, TACTICAL), ("N11b",)) == [TACTICAL]


# Through the graph


def test_the_report_holds_only_what_passed():
    llm = model(
        {
            "Tactical Analysis": [words("Kohli returns {c1}, close to 150.")] * 3,
            "Personal Performance": [words("Kohli returns {c1}.")],
        },
        read=reading(),
    )
    state = ask(llm)
    assert state["report"].by == ONE_FINDING
    assert state["report"].text.startswith("Personal Performance:")
    assert "150" not in state["answer"]


def test_two_that_pass_are_joined_by_the_model():
    llm = model(
        {
            "Tactical Analysis": [words("Kohli returns {c1}.")],
            "Personal Performance": [words("Kohli returns {c1}.")],
        },
        report=[words("Both say Kohli returns {c1}.")],
    )
    state = ask(llm)
    assert state["report"].by == BY_MODEL
    assert "Both say Kohli returns a strike rate of 135.1" in state["answer"]
