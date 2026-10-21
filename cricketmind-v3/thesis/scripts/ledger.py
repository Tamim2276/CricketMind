"""Writing a result file safely, and recording it in results/ledger.jsonl.

Every result the paper uses gets one ledger line: when it was made, by which script,
from which commit (and whether there were uncommitted changes), and the file's SHA-256.
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]  # cricketmind-v3/
RESULTS = PROJECT / "thesis" / "results"
LEDGER = RESULTS / "ledger.jsonl"


def write_safely(path: Path, text: str) -> None:
    """Write to a temporary file first, so a power cut never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(temporary, path)


def add_to_ledger(entry: dict, output: Path, ledger: Path = LEDGER) -> None:
    """One line: the entry, plus when, which code, and the output's fingerprint."""
    line = {"run_at": datetime.now().isoformat(timespec="seconds")}
    line.update(entry)
    line["code_commit"] = git("rev-parse", "--short", "HEAD")
    line["uncommitted_changes"] = git("status", "--porcelain", "--", ".") != ""
    line["output"] = relative(output)
    line["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(line) + "\n")


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT).as_posix()
    except ValueError:
        return path.as_posix()  # outside the project (a test run)


def git(*args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=PROJECT, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()
