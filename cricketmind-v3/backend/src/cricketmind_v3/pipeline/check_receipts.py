"""Re-run every receipt in a file on the saved data, and confirm each number (Day 48).

Run from backend/:
    uv run python -m cricketmind_v3.pipeline.check_receipts ../data/cricket.duckdb receipts.jsonl

One receipt per line. The database is opened read-only. Exit code 1 if any receipt is
not confirmed.
"""

import sys
from pathlib import Path

import duckdb

from cricketmind_v3.application.receipt_checker import check_receipt
from cricketmind_v3.domain.receipts import Receipt, ReceiptError, from_line, to_line
from cricketmind_v3.infrastructure.duckdb_engines import engines_on


def write_receipts(path: Path, receipts: list[Receipt]) -> None:
    """A receipts file: one receipt per line (the final exam keeps one, Day 177)."""
    lines = []
    for receipt in receipts:
        lines.append(to_line(receipt))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_file(db, path: Path, say=print) -> tuple[int, int]:
    """(confirmed, not confirmed), saying one line for every receipt."""
    engines = engines_on(db)
    confirmed = 0
    failed = 0
    lines = path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            receipt = from_line(line)
        except ReceiptError as problem:
            failed += 1
            say(f"line {number}: NOT CONFIRMED: {problem}")
            continue
        problems = check_receipt(receipt, engines)
        if problems:
            failed += 1
            say(f"line {number}: NOT CONFIRMED: " + "; ".join(problems))
        else:
            confirmed += 1
            say(f"line {number}: confirmed {receipt.value} ({receipt.engine})")
    say(f"{confirmed} confirmed, {failed} not")
    return confirmed, failed


def main(arguments: list[str]) -> int:
    if len(arguments) != 2:
        print(
            "usage: python -m cricketmind_v3.pipeline.check_receipts <database> <file>"
        )
        return 2
    database, receipts = arguments
    with duckdb.connect(database, read_only=True) as db:
        _confirmed, failed = check_file(db, Path(receipts))
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
