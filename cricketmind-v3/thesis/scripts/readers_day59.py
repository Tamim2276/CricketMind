"""Day 59 (D-024, D-074): which door and which model read questions best, as
pre-registered (thesis/prereg/day59_readers.md).

Run from backend/, with Ollama running and no CricShot training on the GPU:
    uv run python ../thesis/scripts/readers_day59.py ../data/cricket.duckdb

Reads the 90 practice wordings with the rules, then OllamaLLM and LangChainLLM with
qwen2.5:7b, then the adapter that stays with qwen3:8b. Every reading is kept in
data/day59_runs/ as soon as it is made, so a power cut costs one question at most.
Writes thesis/results/readers_day59.csv and readers_day59.md, one ledger line each.
Reads the database only.
"""

import csv
import dataclasses
import hashlib
import io
import json
import sys
import time

import duckdb
import ollama
from ledger import PROJECT, RESULTS, add_to_ledger, relative, write_safely

from cricketmind_v3.application.reader import read_question
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.kinds import KINDS
from cricketmind_v3.domain.names import name_list
from cricketmind_v3.infrastructure.langchain_llm import LangChainLLM
from cricketmind_v3.infrastructure.ollama_llm import OllamaLLM
from cricketmind_v3.pipeline.grounds import load_grounds
from cricketmind_v3.pipeline.players import load_players
from cricketmind_v3.pipeline.teams import load_teams

PREREG = PROJECT / "thesis" / "prereg" / "day59_readers.md"
PREREG_SHA256 = "af1d59d002f00bf667d627c4098f6bf7b7c0141689da5421485fa840499ecd45"
PRACTICE = PROJECT / "thesis" / "practice" / "day59_readings.jsonl"
PRACTICE_SHA256 = "ff50793238791582acb04b299e3ba8ef9dec58cdcc30399d208733b3227f00d6"
RUNS = PROJECT / "data" / "day59_runs"
SMALLER = "qwen2.5:7b"  # the tie keeps it (section 4)
LARGER = "qwen3:8b"
DOORS = {"OllamaLLM": OllamaLLM, "LangChainLLM": LangChainLLM}


def main(database: str) -> None:
    for path, sha in ((PREREG, PREREG_SHA256), (PRACTICE, PRACTICE_SHA256)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            sys.exit(
                f"{relative(path)} has changed since it was fingerprinted: not running"
            )
    started = time.time()
    db = duckdb.connect(database, read_only=True)
    registers = {
        "players": tuple(load_players(db)),
        "grounds": tuple(load_grounds(db)),
        "teams": tuple(load_teams(db)),
    }
    registers["names"] = cricket_gazetteer(name_list(list(registers["players"])))
    rows = []
    for line in PRACTICE.read_text(encoding="utf-8").splitlines():
        rows.append(json.loads(line))

    scored = {}
    scored["rules"] = read_all("rules", None, rows, registers, started)
    for door in ("OllamaLLM", "LangChainLLM"):
        label = f"{door} {SMALLER}"
        scored[label] = read_all(label, DOORS[door](SMALLER), rows, registers, started)

    # section 4: the higher score stays; a tie keeps OllamaLLM
    door = "OllamaLLM"
    if exact(scored[f"LangChainLLM {SMALLER}"]) > exact(scored[f"OllamaLLM {SMALLER}"]):
        door = "LangChainLLM"
    label = f"{door} {LARGER}"
    scored[label] = read_all(label, DOORS[door](LARGER), rows, registers, started)
    model = SMALLER
    if exact(scored[label]) > exact(scored[f"{door} {SMALLER}"]):
        model = LARGER

    write("readers_day59.csv", table(scored))
    report = summary(scored, door, model, digests())
    write("readers_day59.md", report)
    print(report)
    say("done", started)


def read_all(
    label: str, llm, rows: list[dict], registers: dict, started: float
) -> list:
    """Every wording read once by this reader; what is kept on disk is not read again."""
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / (label.replace(" ", "_").replace(":", "-") + ".jsonl")
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            kept = json.loads(line)
            done[kept["id"]] = kept
    with open(path, "a", encoding="utf-8", newline="\n") as out:
        for row in rows:
            if row["id"] in done:
                continue
            began = time.time()
            reading = read_question(
                row["question"],
                llm,
                registers["players"],
                registers["names"],
                registers["grounds"],
                registers["teams"],
                row["clip"],
            )
            kept = {
                "id": row["id"],
                "kind": reading.kind,
                "subject": filled(reading.subject),
                "by": reading.by,
                "seconds": round(time.time() - began, 2),
            }
            out.write(json.dumps(kept, ensure_ascii=False) + "\n")
            out.flush()
            done[row["id"]] = kept
    say(f"{label}: read", started)
    results = []
    for row in rows:
        results.append((row, done[row["id"]]))
    return results


def filled(subject) -> dict:
    found = {}
    for field in dataclasses.fields(subject):
        value = getattr(subject, field.name)
        if value is not None:
            found[field.name] = value
    return found


def right(row: dict, kept: dict) -> bool:
    # section 3: the kind, and the whole subject, exactly
    return kept["kind"] == row["kind"] and kept["subject"] == row["expected"]


def exact(results: list) -> int:
    count = 0
    for row, kept in results:
        if right(row, kept):
            count += 1
    return count


def table(scored: dict) -> str:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(
        [
            "reader",
            "exactly right",
            "of",
            "share",
            *KINDS,
            "read by rules",
            "mean seconds",
        ]
    )
    for label, results in scored.items():
        by_kind = []
        for kind in KINDS:
            count = 0
            for row, kept in results:
                if row["kind"] == kind and right(row, kept):
                    count += 1
            by_kind.append(count)
        fell_back = 0
        seconds = 0.0
        for _, kept in results:
            if kept["by"] == "rules":
                fell_back += 1
            seconds += kept["seconds"]
        writer.writerow(
            [
                label,
                exact(results),
                len(results),
                round(exact(results) / len(results), 4),
                *by_kind,
                fell_back,
                round(seconds / len(results), 2),
            ]
        )
    return out.getvalue()


def summary(scored: dict, door: str, model: str, found_digests: dict) -> str:
    lines = [
        "# Day 59: which door and which model read questions best (D-024, D-074)",
        "",
        "90 practice wordings (30 questions, 6 of each kind, 3 ways each), each read once.",
        "Exactly right: the kind and the whole subject (pre-registration, section 3).",
        "",
        "| reader | exactly right | share |",
        "|---|---|---|",
    ]
    for label, results in scored.items():
        lines.append(
            f"| {label} | {exact(results)} of 90 | {exact(results) / 90:.1%} |"
        )
    lines += [
        "",
        f"- **The door that stays:** {door} (a tie keeps OllamaLLM).",
        (
            f"- **The model that stays:** {model} (a tie keeps qwen2.5:7b), Ollama digest "
            f"{found_digests.get(model, 'not found')}."
        ),
        "- A gap of 9 wordings or fewer is within luck (section 5): the rule chose, but the",
        "  paper claims no difference that small.",
        "",
        "## What went wrong most often",
        "",
    ]
    for label, results in scored.items():
        wrong = {}
        for row, kept in results:
            for field in mistakes(row, kept):
                wrong[field] = wrong.get(field, 0) + 1
        ranked = []
        for field, count in wrong.items():
            ranked.append((-count, field))
        ranked.sort()
        parts = []
        for count, field in ranked[:6]:
            parts.append(f"{field} {-count}")
        lines.append(f"- {label}: {', '.join(parts) if parts else 'nothing'}")
    lines.append("")
    return "\n".join(lines)


def mistakes(row: dict, kept: dict) -> list[str]:
    """The boxes that differ from the expected reading."""
    found = []
    if kept["kind"] != row["kind"]:
        found.append("kind")
    fields = sorted(set(row["expected"]) | set(kept["subject"]))
    for field in fields:
        if row["expected"].get(field) != kept["subject"].get(field):
            found.append(field)
    return found


def digests() -> dict:
    found = {}
    for entry in ollama.Client().list().models:
        found[entry.model] = entry.digest[:12]
    return found


def write(name: str, text: str) -> None:
    path = RESULTS / name
    write_safely(path, text)
    add_to_ledger(
        {
            "experiment": "D-074 the readers",
            "data": "practice questions written for Day 59; the real registers",
            "script": "thesis/scripts/readers_day59.py",
            "prereg_sha256": PREREG_SHA256,
        },
        path,
    )
    print(f"wrote {relative(path)}")


def say(message: str, started: float) -> None:
    print(f"[{time.time() - started:7.0f}s] {message}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
