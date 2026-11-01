"""Day 59 (D-080): do the reader's fixes help, on questions they never saw? As
pre-registered (thesis/prereg/day59b_fixes.md).

Run from backend/, with Ollama running and no CricShot training on the GPU:
    uv run python ../thesis/scripts/readers_day59b.py ../data/cricket.duckdb before
then, once the fixes are committed:
    uv run python ../thesis/scripts/readers_day59b.py ../data/cricket.duckdb after

Each stage reads the 90 new wordings with the rules and with LangChainLLM and
qwen2.5:7b. Every reading, with its notes, is kept in data/day59b_runs/ as soon as it is
made, so a power cut costs one question at most. "after" also compares the two stages
by the rule. Writes thesis/results/readers_day59b_<stage>.md (and the comparison), one
ledger line each. Reads the database only.
"""

import hashlib
import json
import sys
import time

import duckdb
from ledger import PROJECT, RESULTS, add_to_ledger, relative, write_safely
from readers_day59 import filled, mistakes, right

from cricketmind_v3.application.reader import read_question
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.kinds import KINDS
from cricketmind_v3.domain.names import name_list
from cricketmind_v3.infrastructure.langchain_llm import LangChainLLM
from cricketmind_v3.pipeline.grounds import load_grounds
from cricketmind_v3.pipeline.players import load_players
from cricketmind_v3.pipeline.teams import load_teams

PREREG = PROJECT / "thesis" / "prereg" / "day59b_fixes.md"
PREREG_SHA256 = "d05c277c5a7c3ca30bf00d34e6ef962d40570af8a897706efd36c973001d94c1"
PRACTICE = PROJECT / "thesis" / "practice" / "day59b_readings.jsonl"
PRACTICE_SHA256 = "956994847959b1d51646b36b8196e8b61f9e910022e2448a22867156be5cafec"
RUNS = PROJECT / "data" / "day59b_runs"
MODEL = "qwen2.5:7b"  # D-074
OFF_MENU = "C6"  # the group that asks for a number no engine counts


def main(database: str, stage: str) -> None:
    if stage not in ("before", "after"):
        sys.exit("the stage is before or after")
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

    scored = {
        f"rules ({stage})": read_all(f"{stage}_rules", None, rows, registers, started),
        f"LangChainLLM {MODEL} ({stage})": read_all(
            f"{stage}_model", LangChainLLM(MODEL), rows, registers, started
        ),
    }
    report = summary(scored)
    write(f"readers_day59b_{stage}.md", report, stage)
    print(report)
    if stage == "after":
        before = kept_readings(RUNS / "before_model.jsonl", rows)
        after = kept_readings(RUNS / "after_model.jsonl", rows)
        verdict = compare(before, after)
        write("readers_day59b_verdict.md", verdict, "verdict")
        print(verdict)
    say("done", started)


def read_all(
    label: str, llm, rows: list[dict], registers: dict, started: float
) -> list:
    """Every wording read once; what is kept on disk is not read again."""
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"{label}.jsonl"
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
                "measure": getattr(reading, "measure", None),
                "by": reading.by,
                "notes": list(reading.notes),
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


def kept_readings(path, rows: list[dict]) -> list:
    done = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        kept = json.loads(line)
        done[kept["id"]] = kept
    results = []
    for row in rows:
        results.append((row, done[row["id"]]))
    return results


def tally(results: list) -> dict:
    """What a reader got right, counted once: in all, by kind, without the off-menu
    group; how many of the cant_check wordings it declined; how often it fell back."""
    found = {"right": 0, "on_menu": 0, "declined": 0, "fell_back": 0}
    for kind in KINDS:
        found[kind] = 0
    for row, kept in results:
        if right(row, kept):
            found["right"] += 1
            found[row["kind"]] += 1
            if not row["id"].startswith(OFF_MENU):
                found["on_menu"] += 1
        if row["kind"] == "cant_check" and kept["kind"] == "cant_check":
            found["declined"] += 1
        if kept["by"] == "rules":
            found["fell_back"] += 1
    return found


def summary(scored: dict) -> str:
    header = "| reader | exactly right |"
    rule = "|---|---|"
    for kind in KINDS:
        header += f" {kind} |"
        rule += "---|"
    lines = [
        "# Day 59: the reader's fixes on 90 new wordings (D-080)",
        "",
        header + " declined, of 18 | read by rules |",
        rule + "---|---|",
    ]
    for label, results in scored.items():
        found = tally(results)
        line = f"| {label} | {found['right']} of 90 |"
        for kind in KINDS:
            line += f" {found[kind]} |"
        lines.append(line + f" {found['declined']} | {found['fell_back']} |")
    lines += ["", "## What went wrong most often", ""]
    for label, results in scored.items():
        wrong = {}
        for row, kept in results:
            for field in mistakes(row, kept):
                wrong[field] = wrong.get(field, 0) + 1
        ranked = []
        for field, many in wrong.items():
            ranked.append((-many, field))
        ranked.sort()
        parts = []
        for many, field in ranked[:6]:
            parts.append(f"{field} {-many}")
        if not parts:
            parts.append("nothing")
        lines.append(f"- {label}: {', '.join(parts)}")
    lines += [
        "",
        "## Why a model's reply was not used (the reader fell back to the rules)",
        "",
    ]
    for label, results in scored.items():
        for row, kept in results:
            if kept["by"] == "rules" and kept["notes"]:
                lines.append(f"- {label}, {row['id']}: {kept['notes'][0][:300]}")
    lines.append("")
    return "\n".join(lines)


def compare(before: list, after: list) -> str:
    """The pre-registered rule (section 4)."""
    old = tally(before)
    new = tally(after)
    measured = 0
    asked = 0
    for row, kept in after:
        if row["measure"] is not None and row["kind"] in ("stats", "prediction"):
            asked += 1
            if kept.get("measure") == row["measure"]:
                measured += 1
    stays = "the reader before (the fixes are taken out)"
    if new["right"] > old["right"]:
        stays = "the fixed reader"
    luck = "more than luck"
    if abs(new["right"] - old["right"]) <= 9:
        luck = "within luck: no difference is claimed"
    return "\n".join(
        [
            "# Day 59: the rule (pre-registration, section 4)",
            "",
            f"- Exactly right, of 90: before {old['right']}, after {new['right']}; {luck}.",
            (
                f"- Without the off-menu group (87 wordings): before {old['on_menu']}, "
                f"after {new['on_menu']}."
            ),
            f"- Declined, of the 18 cant_check: before {old['declined']}, after {new['declined']}.",
            f"- **Stays:** {stays}.",
            (
                f"- The fixed reader's measure was the expected one in {measured} of {asked} "
                "stats and prediction wordings (reported, deciding nothing)."
            ),
            "",
        ]
    )


def write(name: str, text: str, stage: str) -> None:
    path = RESULTS / name
    write_safely(path, text)
    add_to_ledger(
        {
            "experiment": f"D-080 the reader's fixes ({stage})",
            "data": "practice questions written for Day 59 (second set); the real registers",
            "script": "thesis/scripts/readers_day59b.py",
            "prereg_sha256": PREREG_SHA256,
        },
        path,
    )
    print(f"wrote {relative(path)}")


def say(message: str, started: float) -> None:
    print(f"[{time.time() - started:7.0f}s] {message}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
