"""E-004: run the fabrication harness on synthetic findings and record the result.

Run from backend/:  uv run python ../thesis/scripts/fabrication_synthetic.py

Writes thesis/results/fabrication_synthetic.csv and adds one line to
thesis/results/ledger.jsonl. SYNTHETIC: every number in the harness is invented or
comes from the v2 synthetic corpus; the result shows what the gates catch, nothing
about real players.
"""

import csv
import io

from ledger import RESULTS, add_to_ledger, relative, write_safely

from cricketmind_v3.application.harness import (
    GATES,
    classes_caught,
    controls_passed,
    run,
)

CSV_PATH = RESULTS / "fabrication_synthetic.csv"


def main() -> None:
    results = run()
    write_safely(CSV_PATH, csv_text(results))
    summary = {}
    for gate, _ in GATES:
        caught = classes_caught(results, gate)
        summary[gate] = {
            "lie classes caught": f"{len(caught)}/8",
            "caught": caught,
            "correct findings passed": f"{controls_passed(results, gate)}/2",
        }
    add_to_ledger(
        {
            "experiment": "E-004 fabrication containment",
            "data": "synthetic",
            "script": "thesis/scripts/fabrication_synthetic.py",
            "summary": summary,
        },
        CSV_PATH,
    )
    for gate, line in summary.items():
        print(
            f"{gate:10}  {line['lie classes caught']} lie classes caught, "
            f"{line['correct findings passed']} correct findings passed"
        )
    print(f"wrote {relative(CSV_PATH)} and one line in the ledger")


def csv_text(results) -> str:
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
    text = io.StringIO()
    csv.writer(text, lineterminator="\r\n").writerows(rows)
    return text.getvalue()


if __name__ == "__main__":
    main()
