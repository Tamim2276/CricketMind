"""Join the friends' exam files and lock the result (Day 15, D-070, D-087).

Put every friend's CSV in thesis/exam/raw/, unopened. Then, from backend/:
    uv run python -m cricketmind_v3.pipeline.lock_exam

It prints counts, row numbers and the SHA-256, and never a word of a question, so nobody
reads the exam to lock it. If anything is wrong it names the file and the row, and writes
nothing. It refuses to run twice: a locked exam is never written again.
"""

import csv
import hashlib
import html
import io
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXAM = ROOT / "thesis" / "exam"
FINAL = "final_exam.csv"
GUIDE = ROOT / "thesis" / "drafts" / "day15_friend_guide.html"

COLUMNS = ("group", "kind", "question", "expected_handling", "measure", "filters")
KINDS = ("stats", "prediction", "video", "rules", "cant_check")
TRICKY = ("false_premise", "trick")
ROWS_IN_PART = {"A": 80, "B": 20, "C": 20}  # the plan's Day 15: 120 rows
WORDINGS = 4  # each question of Part A, written four ways


class ExamLockedError(ValueError):
    """The exam is not locked yet: no real answers before it is (D-070)."""


@dataclass(frozen=True, kw_only=True)
class Row:
    file: str  # "friend1.csv"
    number: int  # its row in the sheet, counting the column names as row 1
    cells: dict[str, str]


def exam_is_locked(folder: Path = EXAM) -> bool:
    return (folder / FINAL).exists()


def require_locked_exam(folder: Path = EXAM) -> None:
    """Stops a script that would give a real answer before the exam is locked (D-070)."""
    if not exam_is_locked(folder):
        raise ExamLockedError(
            f"{folder / FINAL} does not exist: lock the exam first (Day 67 needs it)"
        )


def _decoded(path: Path) -> str:
    # Google Sheets writes UTF-8; Excel may write its own code page
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def read_file(path: Path) -> tuple[list[Row], list[str]]:
    """The rows of one friend's file, and what is wrong with its columns."""
    lines = _decoded(path).splitlines(keepends=True)
    if not lines:
        return [], [f"{path.name}: the file is empty"]
    delimiter = ","
    if ";" in lines[0] and "," not in lines[0]:
        delimiter = ";"  # Excel in some countries
    reader = csv.reader(lines, delimiter=delimiter)
    header = []
    for name in next(reader):
        header.append(name.strip().lower())
    missing = []
    for column in COLUMNS:
        if column not in header:
            missing.append(column)
    if missing:
        return [], [f"{path.name}: no column called {', '.join(missing)}"]

    rows = []
    for number, values in enumerate(reader, start=2):
        cells = {}
        for column, value in zip(header, values, strict=False):
            cells[column] = value.strip()
        filled = False
        for column in COLUMNS:
            if cells.get(column, ""):
                filled = True
        if filled:
            rows.append(Row(file=path.name, number=number, cells=cells))
    return rows, []


def _part(row: Row) -> str:
    return row.cells.get("group", "")[:1].upper()


def _where(row: Row) -> str:
    return f"{row.file}, row {row.number}"


def _good_group(row: Row) -> bool:
    return _part(row) in ROWS_IN_PART and row.cells.get("group", "")[1:].isdigit()


def row_problems(row: Row) -> list[str]:
    """What is wrong with one row. Names columns, never what is written in them."""
    problems = []
    kind = row.cells.get("kind", "").lower()
    if not _good_group(row):
        problems.append(f"{_where(row)}: the group must be like A1, B1 or C1")
    if not row.cells.get("question", ""):
        problems.append(f"{_where(row)}: no question")
    if not row.cells.get("expected_handling", ""):
        problems.append(f"{_where(row)}: no expected_handling")
    if _part(row) == "A" and kind not in KINDS:
        problems.append(f"{_where(row)}: Part A needs a kind from the five")
    if _part(row) == "B" and kind and kind not in KINDS:
        problems.append(f"{_where(row)}: not one of the five kinds")
    if _part(row) == "C" and kind not in TRICKY:
        problems.append(f"{_where(row)}: Part C is false_premise or trick")
    return problems


def group_problems(rows: list[Row]) -> list[str]:
    """Part A: each group holds four wordings of one kind of question."""
    groups = {}
    for row in rows:
        if _part(row) == "A" and _good_group(row):  # any other label is not said
            key = (row.file, row.cells.get("group", "").upper())
            if key not in groups:
                groups[key] = []
            groups[key].append(row)
    problems = []
    for (file, group), members in groups.items():
        if len(members) != WORDINGS:
            problems.append(
                f"{file}: group {group} has {len(members)} wordings, not {WORDINGS}"
            )
        kinds = set()
        for member in members:
            kinds.add(member.cells.get("kind", "").lower())
        if len(kinds) > 1:
            problems.append(f"{file}: group {group} mixes kinds of question")
    return problems


def _plain(text: str) -> str:
    return " ".join(html.unescape(text).lower().split())


def copied_problems(rows: list[Row], guide: Path = GUIDE) -> list[str]:
    """The same question twice, or one lifted from the guide's own examples."""
    guide_text = ""
    if guide.exists():
        guide_text = _plain(guide.read_text(encoding="utf-8"))
    problems = []
    seen = {}
    for row in rows:
        question = _plain(row.cells.get("question", ""))
        if not question:
            continue
        if question in seen:
            problems.append(f"{_where(row)}: the same question as {seen[question]}")
        else:
            seen[question] = _where(row)
        if question in guide_text:
            problems.append(f"{_where(row)}: copied from the guide's examples")
    return problems


def count_problems(rows: list[Row]) -> list[str]:
    counts = _part_counts(rows)
    problems = []
    for part, wanted in ROWS_IN_PART.items():
        if counts[part] != wanted:
            problems.append(f"Part {part} has {counts[part]} rows, not {wanted}")
    return problems


def _part_counts(rows: list[Row]) -> dict[str, int]:
    counts = dict.fromkeys(ROWS_IN_PART, 0)
    for row in rows:
        if _part(row) in counts:
            counts[_part(row)] += 1
    return counts


def _kind_counts(rows: list[Row]) -> dict[str, int]:
    # only the known kinds are ever said: a cell can hold anything someone typed
    counts = {}
    for row in rows:
        kind = row.cells.get("kind", "").lower()
        if not kind:
            kind = "(left empty)"
        elif kind not in KINDS and kind not in TRICKY:
            kind = "(not a kind)"
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def joined_exam(rows: list[Row]) -> bytes:
    """One file: each friend's groups kept apart, the same bytes on every machine."""
    writers = {}
    for row in rows:
        if row.file not in writers:
            writers[row.file] = f"w{len(writers) + 1}"
    text = io.StringIO(newline="")
    table = csv.writer(text, lineterminator="\n")
    table.writerow(("id", "writer", *COLUMNS))
    for number, row in enumerate(rows, start=1):
        who = writers[row.file]
        cells = [str(number), who, f"{who}-{row.cells['group'].upper()}"]
        cells.append(row.cells.get("kind", "").lower())
        for column in COLUMNS[2:]:
            cells.append(row.cells.get(column, ""))
        table.writerow(cells)
    return text.getvalue().encode("utf-8")


def lock(folder: Path = EXAM, guide: Path = GUIDE, say=print) -> int:
    """0 if the exam is now locked, 1 if not. Says only counts, row numbers and the SHA."""
    final = folder / FINAL
    if final.exists():
        say(f"{final} already exists: a locked exam is never written again")
        return 1
    files = sorted((folder / "raw").glob("*.csv"))
    if not files:
        say(f"no CSV files in {folder / 'raw'}")
        return 1

    rows = []
    problems = []
    for path in files:
        found, wrong = read_file(path)
        rows.extend(found)
        problems.extend(wrong)
    for row in rows:
        problems.extend(row_problems(row))
    problems.extend(group_problems(rows))
    problems.extend(copied_problems(rows, guide))
    problems.extend(count_problems(rows))

    say(f"{len(files)} files, {len(rows)} rows")
    for part, count in _part_counts(rows).items():
        say(f"  Part {part}: {count}")
    for kind, count in sorted(_kind_counts(rows).items()):
        say(f"  {kind}: {count}")
    if problems:
        say(f"NOT LOCKED: {len(problems)} problems. Nothing was written.")
        for problem in problems:
            say(f"  {problem}")
        return 1

    content = joined_exam(rows)
    with open(final, "xb") as out:  # "x": never over an existing file
        out.write(content)
    say(f"LOCKED: {final}")
    say(f"SHA-256: {hashlib.sha256(content).hexdigest()}")
    say("Put that SHA-256 in E-013 and commit. Do not open the file until Day 178.")
    return 0


if __name__ == "__main__":
    sys.exit(lock())
