"""Running the real model on practice questions (Days 67 and 78), after the exam is locked.

Each result is written to its file as soon as it exists, and a question already there is
skipped: a power cut loses one question at most. Nothing here scores anything.
"""

import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path

from cricketmind_v3.application.agent_factory import write_finding
from cricketmind_v3.application.fact_surface import ask_engines
from cricketmind_v3.application.faith_check import check_with_receipts
from cricketmind_v3.application.graph.build import build_graph
from cricketmind_v3.application.graph.context import GraphContext
from cricketmind_v3.application.reader import read_question
from cricketmind_v3.domain.faith.gate import MAX_REGENERATIONS, Context
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer, gate_gazetteer
from cricketmind_v3.domain.names import name_list
from cricketmind_v3.domain.readback import read_back
from cricketmind_v3.domain.render import render
from cricketmind_v3.domain.roster import ROSTER
from cricketmind_v3.domain.shot_dna import stub_shot_dna
from cricketmind_v3.infrastructure.duckdb_engines import engines_on
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from cricketmind_v3.infrastructure.prompt_files import read_rules, read_specs
from cricketmind_v3.pipeline.grounds import load_grounds
from cricketmind_v3.pipeline.players import load_players
from cricketmind_v3.pipeline.teams import load_teams


def live_context(db, llm) -> GraphContext:
    """Everything the graph needs, on the real database. Each question's run gets its own
    cursors: one for N7a and one for the engines (D-082)."""
    players = tuple(load_players(db))
    grounds = tuple(load_grounds(db))
    teams = tuple(load_teams(db))
    team_names = []
    for team in teams:
        team_names.append(team.name)
    ground_names = []
    for ground in grounds:
        ground_names.append(ground.name)
    said = name_list(list(players))
    return GraphContext(
        names=cricket_gazetteer(said),
        gate_names=gate_gazetteer(said, tuple(team_names), tuple(ground_names)),
        llm=llm,
        players=players,
        grounds=grounds,
        teams=teams,
        deliveries=DuckDBDeliveryRepository(db.cursor()),
        engines=engines_on(db.cursor()),
        specs=read_specs(),
        rules=read_rules(),
    )


def read_questions(path: Path) -> list[str]:
    """One question a line; a line starting with # is a note."""
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            questions.append(line)
    return questions


def already_done(out: Path) -> set[str]:
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["question"])
    return done


def run_all(
    questions: list[str], out: Path, run_one: Callable[[str], dict], say=print
) -> int:
    """Each question not yet in the file; how many were run this time."""
    done = already_done(out)
    ran = 0
    for question in questions:
        if question in done:
            say(f"already done: {question}")
            continue
        row = run_one(question)
        row["question"] = question
        with open(out, "a", encoding="utf-8") as results:
            results.write(json.dumps(row) + "\n")  # kept before the next one starts
        ran += 1
    return ran


def one_agent(agent: str, question: str, context: GraphContext, say=print) -> dict:
    """Day 67: one agent on one question, with the real model and all seven checks. The
    numbers come as the graph gets them; tries and reasons are all kept."""
    started = time.time()
    reading = read_question(
        question,
        context.llm,
        context.players,
        context.names,
        context.grounds,
        context.teams,
    )
    say(f"\n{question}\n{read_back(reading)}")
    subject = stub_shot_dna(reading.subject, None).subject
    answers = ask_engines(subject, reading.kind, "N8", context.engines)
    answers += ask_engines(subject, reading.kind, "N9", context.engines)
    gate = Context(allowed=ROSTER, names=context.gate_names or context.names)
    tries = []
    failures = ()
    finding = None
    for _ in range(1 + MAX_REGENERATIONS):
        finding = write_finding(
            agent,
            question,
            answers,
            context.llm,
            context.specs.get(agent),
            context.rules,
            failures=failures,
            shown=dict(reading.shown),
            premise=reading.premise,
        )
        wrong, _receipts = check_with_receipts(finding, context.engines, gate)
        failures = tuple(wrong)
        reasons = []
        for failure in wrong:
            reasons.append({"check": failure.check, "reason": failure.reason})
        tries.append({"template": finding.template, "failures": reasons})
        say(f"  try {len(tries)}: {finding.template}")
        for failure in wrong:
            say(f"    {failure.check}: {failure.reason}")
        if not wrong:
            break
    passed = not failures
    row = {
        "read_back": read_back(reading),
        "by": reading.by,
        "kind": reading.kind,
        "numbers_offered": len(answers),
        "tries": tries,
        "passed": passed,
        "seconds": round(time.time() - started, 1),
    }
    if passed:
        row["answer"] = render(finding)
        say(f"  PASSED: {row['answer']}")
    else:
        say("  WITHHELD")
    return row


def whole_graph(question: str, context: GraphContext, say=print) -> dict:
    """Day 78: the whole graph on one question, and what each part did."""
    started = time.time()
    fresh = dataclasses.replace(context)  # the graph never writes to it
    state = build_graph().invoke({"question": question, "clip": None}, context=fresh)
    say(f"\n{question}\n{state['answer']}")
    routing = state.get("routing")
    report = state.get("report")
    withheld = []
    for each in state.get("withheld", ()):
        withheld.append(
            {"agent": each.agent, "checks": each.checks, "reasons": each.reasons}
        )
    tries = []
    for finding in state.get("findings", []):
        tries.append(
            {
                "agent": finding.agent,
                "template": finding.template,
                "unreadable": finding.unreadable,
            }
        )
    return {
        "answer": state["answer"],
        "read_by": state["reading"].by,
        "kind": state["reading"].kind,
        "chosen": list(routing.chosen) if routing is not None else [],
        "how": routing.how if routing is not None else None,
        "routed_by": routing.by if routing is not None else None,
        "checks_run": state.get("checks_run", 0),
        "tries": tries,
        "withheld": withheld,
        "report_by": report.by if report is not None else None,
        "report_note": report.note if report is not None else None,
        "seconds": round(time.time() - started, 1),
    }
