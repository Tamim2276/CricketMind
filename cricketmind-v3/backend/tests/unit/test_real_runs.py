"""Days 67, 78, 79: the runs the real model will do, tried with fakes.

Written now and run after the exam is locked (D-070): a fake stands in, so nothing here
is a real answer. What these test: each result is kept as it comes, a power cut loses
little, and no script can run before the lock.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.pipeline.lock_exam import ExamLockedError
from cricketmind_v3.pipeline.real_runs import (
    already_done,
    one_agent,
    read_questions,
    run_all,
    whole_graph,
)
from tests.world import (
    context_of,
    model,
    reading,
    words,
)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
THESIS = Path(__file__).resolve().parents[3] / "thesis"


def quiet(_: str) -> None:
    pass


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Keeping results


def test_every_result_is_kept_as_it_comes(tmp_path):
    out = tmp_path / "runs.jsonl"
    seen = []

    def run_one(question: str) -> dict:
        seen.append(question)
        if len(seen) == 2:
            assert len(out.read_text(encoding="utf-8").splitlines()) == 1
        return {"answer": question.upper()}

    assert run_all(["one", "two"], out, run_one, quiet) == 2
    rows = out.read_text(encoding="utf-8").splitlines()
    assert json.loads(rows[1]) == {"answer": "TWO", "question": "two"}


def test_a_run_goes_on_from_where_it_stopped(tmp_path):
    out = tmp_path / "runs.jsonl"

    def stops_on_two(question: str) -> dict:
        if question == "two":
            raise RuntimeError("a power cut")
        return {"answer": "kept"}

    with pytest.raises(RuntimeError):
        run_all(["one", "two", "three"], out, stops_on_two, quiet)
    assert already_done(out) == {"one"}

    seen = []

    def run_one(question: str) -> dict:
        seen.append(question)
        return {"answer": "kept"}

    assert run_all(["one", "two", "three"], out, run_one, quiet) == 2
    assert seen == ["two", "three"]
    assert already_done(out) == {"one", "two", "three"}


def test_no_file_yet_means_nothing_done(tmp_path):
    assert already_done(tmp_path / "none.jsonl") == set()


def test_a_questions_file_keeps_questions_and_leaves_out_notes(tmp_path):
    path = tmp_path / "questions.txt"
    path.write_text("# a note\nOne?\n\n  Two?  \n", encoding="utf-8")
    assert read_questions(path) == ["One?", "Two?"]


def test_the_practice_sets_are_the_size_the_plan_asks():
    assert len(read_questions(THESIS / "practice" / "day67_questions.txt")) == 5
    assert len(read_questions(THESIS / "practice" / "day78_questions.txt")) == 10


# Day 67's run


def test_one_agent_keeps_every_try_and_its_reasons():
    llm = model(
        {
            "Tactical Analysis": [
                words("Kohli returns {c1}, close to 150."),
                words("Kohli returns {c1}."),
            ]
        }
    )
    row = one_agent("N11b", "Kohli's strike rate in T20s?", context_of(llm), quiet)
    assert row["passed"] is True
    assert len(row["tries"]) == 2
    assert row["tries"][0]["failures"][0]["check"] == "C5"
    assert row["tries"][1]["failures"] == []
    assert row["answer"] == "Kohli returns a strike rate of 135.1 [D, 9841 balls]."
    assert row["read_back"].startswith("I read this as:")


def test_one_agent_stops_at_three_tries():
    llm = model({"Tactical Analysis": [words("It is {c1}, near 150.")] * 9})
    row = one_agent("N11b", "Kohli's strike rate in T20s?", context_of(llm), quiet)
    assert row["passed"] is False
    assert len(row["tries"]) == 3
    assert "answer" not in row


# Day 78's run


def test_the_whole_graph_says_what_each_part_did():
    llm = model(
        {"Tactical Analysis": [words("It is {c1}, near 150."), words("It is {c1}.")]},
        chosen=("N11b",),
    )
    row = whole_graph("Kohli's strike rate in T20s?", context_of(llm), quiet)
    assert row["read_by"] == "model"
    assert row["kind"] == "stats"
    assert row["chosen"] == ["N11b"]
    assert row["checks_run"] == 2
    assert row["withheld"] == []
    assert row["report_by"] == "one finding"
    assert len(row["tries"]) == 2
    assert "135.1" in row["answer"]
    json.dumps(row)  # it must fit in the results file


def test_a_question_nobody_can_check_has_nothing_to_report_and_still_saves():
    llm = model({}, read=reading(kind="cant_check"))
    row = whole_graph("Is Kohli injured?", context_of(llm), quiet)
    assert row["chosen"] == []
    assert row["checks_run"] == 0
    assert row["report_by"] is None
    json.dumps(row)


def test_a_withheld_finding_keeps_its_reasons_for_us():
    llm = model(
        {"Tactical Analysis": [words("It is {c1}, near 150.")] * 3}, chosen=("N11b",)
    )
    row = whole_graph("Kohli's strike rate in T20s?", context_of(llm), quiet)
    assert row["withheld"][0]["agent"] == "N11b"
    assert "150" in row["withheld"][0]["reasons"][0]  # in the file, never in the answer
    assert "150" not in row["answer"]
    json.dumps(row)


def test_the_premise_reaches_the_first_agent_too():
    llm = model({"Tactical Analysis": [words("Kohli returns {c1}.")]})
    one_agent("N11b", "Why is Kohli so bad in T20s?", context_of(llm), quiet)
    agent_call = llm.asked[-1]
    assert "holds a judgment" in agent_call[1]


def test_a_subject_is_kept_as_asked():
    subject = CricketSubject(player="V Kohli", match_format="T20")
    llm = model({"Tactical Analysis": [words("Kohli returns {c1}.")]})
    row = one_agent("N11b", "Kohli's strike rate in T20s?", context_of(llm), quiet)
    assert row["passed"]
    assert subject.player in row["read_back"]


# The guard: no real answer before the lock (D-070)


@pytest.mark.parametrize(
    ("script", "arguments"),
    [
        ("first_agent", ("no.duckdb", "no-model")),
        ("real_runs", ("no.duckdb", "no-model")),
        ("ask", ("no.duckdb", "no-model", "What is LBW?")),
    ],
)
def test_no_script_gives_a_real_answer_before_the_lock(script, arguments, monkeypatch):
    module = load_script(script)

    def not_locked() -> None:
        raise ExamLockedError("lock the exam first")

    monkeypatch.setattr(module, "require_locked_exam", not_locked)
    # the guard comes first: nothing is opened and no model is asked before it
    with pytest.raises(ExamLockedError):
        module.main(*arguments)


def test_the_router_script_still_gives_no_answers():
    # it reads and routes with the real model, and gives no agent a prompt file
    text = (SCRIPTS / "route_questions.py").read_text(encoding="utf-8")
    assert "specs=" not in text
