"""E-004: run the fabrication harness on synthetic findings and record the result.

Run from backend/:  uv run python ../thesis/scripts/fabrication_synthetic.py

Writes thesis/results/fabrication_synthetic.csv and adds one line to
thesis/results/ledger.jsonl. SYNTHETIC: every number in the harness is invented or
comes from the v2 synthetic corpus; the result shows what the gates catch, nothing
about real players.
"""

import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from cricketmind_v3.application.harness import (
    GATES,
    classes_caught,
    controls_passed,
    run,
)

PROJECT = Path(__file__).resolve().parents[2]  # cricketmind-v3/
RESULTS = PROJECT / "thesis" / "results"
CSV_PATH = RESULTS / "fabrication_synthetic.csv"
LEDGER = RESULTS / "ledger.jsonl"


def main() -> None:
    results = run()
    write_csv(results)
    summary = {}
    for gate, _ in GATES:
        caught = classes_caught(results, gate)
        summary[gate] = {
            "lie classes caught": f"{len(caught)}/8",
            "caught": caught,
            "correct findings passed": f"{controls_passed(results, gate)}/2",
        }
    add_to_ledger(summary)
    for gate, line in summary.items():
        print(
            f"{gate:10}  {line['lie classes caught']} lie classes caught, "
            f"{line['correct findings passed']} correct findings passed"
        )
    print(
        f"wrote {CSV_PATH.relative_to(PROJECT)} and one line in "
        f"{LEDGER.relative_to(PROJECT)}"
    )


def write_csv(results) -> None:
    header = ["case", "lie", "data"]
    for gate, _ in GATES:
        header.append(gate)
        header.append(gate + ": checks that fired")
    rows = [header]
    for result in results:
        row = [result.case.name, result.case.lie, "synthetic"]
        for gate, _ in GATES:
            fired = result.fired[gate]
            if result.case.expected == ():
                row.append("passed" if fired == () else "wrongly stopped")
            else:
                row.append("caught" if fired != () else "missed")
            row.append(" ".join(fired))
        rows.append(row)

    # write to a temporary file first, so a power cut never leaves half a file
    RESULTS.mkdir(parents=True, exist_ok=True)
    temporary = CSV_PATH.with_suffix(".csv.tmp")
    with open(temporary, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    os.replace(temporary, CSV_PATH)


def add_to_ledger(summary: dict) -> None:
    entry = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": "E-004 fabrication containment",
        "data": "synthetic",
        "script": "thesis/scripts/fabrication_synthetic.py",
        "code_commit": git("rev-parse", "--short", "HEAD"),
        "uncommitted_changes": git("status", "--porcelain", "--", ".") != "",
        "output": str(CSV_PATH.relative_to(PROJECT)).replace("\\", "/"),
        "output_sha256": hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),
        "summary": summary,
    }
    with open(LEDGER, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def git(*args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=PROJECT, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


if __name__ == "__main__":
    main()
