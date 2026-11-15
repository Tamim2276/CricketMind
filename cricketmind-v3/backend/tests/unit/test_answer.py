"""Day 75: what N13 shows: the read-back, what passed with its tags, what was held back,
what was missing, and the receipts. Nothing is invented on the way."""

from cricketmind_v3.application.graph.build import build_graph
from cricketmind_v3.application.ports import FakeFactSource, FakeLLM
from cricketmind_v3.domain.answer import (
    NOBODY_ANSWERS,
    NOTHING_PASSED,
    Parts,
    final_answer,
    receipt_lines,
    what_it_needs,
)
from cricketmind_v3.domain.faith.gate import Withheld
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.reading import Reading
from cricketmind_v3.domain.report import EXPLANATION_LABEL, FIXED_ORDER, Report
from cricketmind_v3.domain.routing import DEFAULT, NOBODY, Routing
from cricketmind_v3.domain.shot_dna import NOT_READ_YET
from cricketmind_v3.domain.subject import CricketSubject
from tests.world import (
    KOHLI,
    ask,
    context_of,
    model,
    reading,
    routed,
    words,
)

BUMRAH = CricketSubject(bowler="JJ Bumrah")
READ = Reading(subject=KOHLI, by="model", kind="stats")
ROUTED = Routing(chosen=("N11b",), how=DEFAULT, by="keywords")
NO_ONE = Routing(chosen=(), how=NOBODY, by="keywords")
REPORT = Report(text="Tactical Analysis: words.", by=FIXED_ORDER)


# The order of an answer


def test_the_read_back_comes_first():
    lines = final_answer(
        Parts(reading=READ, routing=ROUTED, report=REPORT)
    ).splitlines()
    assert lines[0] == "I read this as: a question about what happened: V Kohli, T20."
    assert lines[1] == "Tactical Analysis: words."


def test_a_question_nobody_can_check_gets_only_the_honest_answer():
    declined = Reading(subject=KOHLI, by="model", kind="cant_check")
    answer = final_answer(Parts(reading=declined))
    assert "I can't check that" in answer
    assert "Nothing is answered" not in answer


# What was held back


def test_a_withheld_finding_is_listed_with_its_checks():
    withheld = Withheld(agent="N11c", checks=("C4", "C5"), reasons=("a reason",))
    answer = final_answer(
        Parts(reading=READ, routing=ROUTED, report=REPORT, withheld=(withheld,))
    )
    assert "Withheld: Personal Performance: its words did not pass C4, C5." in answer


def test_a_withheld_findings_reasons_never_show():
    # a reason can quote a number nobody checked
    withheld = Withheld(agent="N11c", checks=("C5",), reasons=("it typed '150'",))
    answer = final_answer(
        Parts(reading=READ, routing=ROUTED, report=REPORT, withheld=(withheld,))
    )
    assert "150" not in answer


def test_when_nothing_passed_it_says_so():
    withheld = Withheld(agent="N11b", checks=("C7",), reasons=())
    empty = Report(text="", by="nothing passed")
    answer = final_answer(
        Parts(reading=READ, routing=ROUTED, report=empty, withheld=(withheld,))
    )
    assert NOTHING_PASSED in answer
    assert "Withheld: Tactical Analysis" in answer


# Nobody can answer (D-084)


def test_nobody_can_answer_and_it_says_why():
    answer = final_answer(
        Parts(
            reading=READ,
            routing=NO_ONE,
            why_no_numbers=("no ball fits every filter", "no clip"),
        )
    )
    assert NOBODY_ANSWERS in answer
    assert "Why: no ball fits every filter." in answer
    assert "Why: no clip." in answer


# What was missing (L-078)


def test_the_medical_agent_needs_a_date():
    assert (
        what_it_needs("N11k", BUMRAH) == "Injury Management could answer with a date."
    )


def test_it_also_says_what_is_in_the_way():
    subject = CricketSubject(bowler="JJ Bumrah", match_format="T20")
    assert what_it_needs("N11k", subject) == (
        "Injury Management could answer with a date, and without a format."
    )


def test_a_reviews_question_needs_a_team():
    assert what_it_needs("N11i", CricketSubject(gender="male")) == (
        "Match Officials could answer with a team."
    )


def test_it_names_the_nearest_of_several_measures():
    # Fan Concierge has many: a team's win share needs a team and men's or women's
    subject = CricketSubject(team="India")
    assert what_it_needs("N11e", subject) == (
        "Fan Concierge could answer with men's or women's cricket."
    )


def test_nothing_missing_means_the_data_has_no_number():
    subject = CricketSubject(bowler="JJ Bumrah", date="2025-01-04")
    assert what_it_needs("N11k", subject) == (
        "Injury Management has no number for this in the data."
    )


def test_a_clip_measure_is_never_what_is_missing():
    # weight_transfer needs a clip, not a word in the question: it has no missing keys, and
    # it must not be taken as the measure nearest to fitting
    assert what_it_needs("N11c", CricketSubject(team="India")) == (
        "Personal Performance could answer with a batter, and without a team."
    )
    assert what_it_needs("N11c", CricketSubject()) == (
        "Personal Performance could answer with a batter."
    )


def test_who_was_wanted_is_told_beside_the_answer():
    routing = Routing(chosen=("N11g",), how=DEFAULT, by="keywords", no_number=("N11k",))
    subject = CricketSubject(bowler="JJ Bumrah", match_format="T20")
    read = Reading(subject=subject, by="model", kind="stats")
    answer = final_answer(Parts(reading=read, routing=routing, report=REPORT))
    assert "Injury Management could answer with a date, and without a format." in answer


def test_who_was_wanted_is_told_when_nobody_answers():
    routing = Routing(chosen=(), how=NOBODY, by="keywords", no_number=("N11k",))
    read = Reading(subject=BUMRAH, by="model", kind="stats")
    answer = final_answer(Parts(reading=read, routing=routing))
    assert "Injury Management could answer with a date." in answer


# Receipts


def test_every_number_shown_has_its_receipt():
    state = ask(
        model(
            {"Tactical Analysis": [words("Kohli returns {c1}.")]},
            chosen=("N11b",),
        )
    )
    lines = state["answer"].splitlines()
    assert "Receipts:" in lines
    assert lines[-1].startswith("- N8.M1 version 1, data a1b2c3d4e5f6:")
    assert '"metric":"strike_rate"' in lines[-1]


def test_one_number_said_twice_has_one_receipt():
    state = ask(
        model(
            {
                "Tactical Analysis": [words("Kohli returns {c1}.")],
                "Personal Performance": [words("Kohli returns {c1}.")],
            },
            report=[words("Kohli returns {c1}.")],
        )
    )
    lines = receipt_lines(state["verified"], state["receipts"])
    assert len(lines) == 2  # the heading, and one receipt


def test_no_receipts_no_heading():
    finding = Finding(agent="N11m", claims=(), template="Words.")
    assert receipt_lines((finding,), {}) == []


def test_a_withheld_numbers_receipt_is_not_shown():
    # the gate keeps it; only what passed is receipted in the answer
    state = ask(
        model(
            {
                "Tactical Analysis": [words("Kohli returns {c1}, near 150.")] * 3,
            },
            chosen=("N11b",),
        )
    )
    assert state["receipts"] != {}
    assert "Receipts:" not in state["answer"]


# Through the graph


def test_a_number_reaches_the_reader_with_its_tag():
    state = ask(
        model({"Tactical Analysis": [words("Kohli returns {c1}.")]}, chosen=("N11b",))
    )
    assert (
        "Tactical Analysis: Kohli returns a strike rate of 135.1 [D, 9841 balls]."
        in state["answer"]
    )


def test_the_routers_reason_never_shows():
    state = ask(
        model({"Tactical Analysis": [words("Kohli returns {c1}.")]}, chosen=("N11b",))
    )
    assert "a test" not in state["answer"]  # the reason in tests/world.py


def test_a_rules_question_gets_the_explainer_and_its_label():
    llm = model(
        {"Explainer": [words("LBW means leg before wicket.")]},
        read=reading(kind="rules", player=None, match_format=None),
    )
    state = ask(llm, question="What is LBW?")
    assert state["routing"].chosen == ("N11m",)
    assert state["answer"].splitlines()[1] == (
        f"{EXPLANATION_LABEL}: LBW means leg before wicket."
    )


def test_an_explanation_with_a_digit_is_never_shown():
    llm = model(
        {"Explainer": [words("An over has 6 balls.")] * 3},
        read=reading(kind="rules", player=None, match_format=None),
    )
    state = ask(llm, question="What is an over?")
    assert "6 balls" not in state["answer"]
    assert "Withheld: Explainer: its words did not pass C5." in state["answer"]


def test_a_rule_with_a_clean_second_try_is_shown():
    llm = model(
        {"Explainer": [words("An over has 6 balls."), words("An over has six balls.")]},
        read=reading(kind="rules", player=None, match_format=None),
    )
    state = ask(llm, question="What is an over?")
    assert "An over has six balls." in state["answer"]
    assert EXPLANATION_LABEL in state["answer"]


def test_nobody_through_the_graph_says_why_and_what_is_missing():
    # a date is what the medical agent lacks; the word "overworked" points to it
    subject_reply = reading(player=None, bowler="Bumrah", match_format=None)
    source = FakeFactSource()
    llm = model({}, read=subject_reply)
    state = ask(llm, source=source, question="Is Bumrah overworked?")
    assert state["routing"].how == NOBODY
    assert NOBODY_ANSWERS in state["answer"]
    assert "Injury Management could answer with a date." in state["answer"]


def test_a_video_question_with_no_clip_says_so():
    llm = model({}, read=reading(kind="video"))
    state = ask(llm, source=FakeFactSource(), question="Was his head still?")
    assert "the question is about a clip, and no clip came with it" in state["answer"]


def test_a_clip_not_read_yet_says_so():
    llm = FakeLLM([reading(kind="video"), routed()])
    context = context_of(llm, FakeFactSource())
    state = build_graph().invoke(
        {"question": "Was his head still?", "clip": "clip.mp4"}, context=context
    )
    assert NOT_READ_YET in state["answer"]
