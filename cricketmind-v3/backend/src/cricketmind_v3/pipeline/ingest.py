"""Load every match in the snapshot into DuckDB, and carry on after a stop (Day 31).

Run from backend/:
    uv run python -m cricketmind_v3.pipeline.ingest ../data/raw/all_json.zip ../data/cricket.duckdb

Stop it at any time, or lose power: run the same command again and it carries on.
"""

import csv
import hashlib
import json
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb

from cricketmind_v3.pipeline.parse_match import ParsedMatch, ParseError, parse_match
from cricketmind_v3.pipeline.schema import TABLES, row_values

# which snapshot this database holds, and which matches are already in it
BOOKKEEPING = (
    """
    CREATE TABLE IF NOT EXISTS snapshot (
        zip_sha256 VARCHAR NOT NULL,
        zip_bytes BIGINT NOT NULL,
        matches_in_zip INTEGER NOT NULL,
        first_run_at TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS loaded (
        match_id VARCHAR PRIMARY KEY,
        deliveries INTEGER NOT NULL,
        loaded_at TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS problems (
        match_id VARCHAR PRIMARY KEY,
        error VARCHAR NOT NULL
    )
    """,
)

NULL = "\\N"  # how a missing value is written in the CSV files


class IngestError(RuntimeError):
    """The database and the zip do not belong together."""


@dataclass(frozen=True, kw_only=True)
class Report:
    """What one run did."""

    loaded_now: int  # matches added by this run
    rows_now: int  # balls added by this run
    problems_now: int  # matches that could not be read
    already_done: int  # matches skipped because an earlier run had them
    matches_in_zip: int


def ingest(
    zip_path: Path,
    db_path: Path,
    batch_size: int = 500,
    stop_after: int | None = None,
    say=print,
) -> Report:
    """Load every match not yet loaded, one batch at a time; each batch all or nothing."""
    sha256, size = fingerprint(zip_path)
    archive = zipfile.ZipFile(zip_path)
    names = match_files(archive)
    db = duckdb.connect(str(db_path))
    try:
        create_tables(db)
        check_snapshot(db, sha256, size, len(names))
        done = finished(db)
        todo = []
        for name in names:
            if name.removesuffix(".json") not in done:
                todo.append(name)
        already = len(names) - len(todo)
        if stop_after is not None:
            todo = todo[:stop_after]  # for testing a stop part-way

        loaded = rows = problems = 0
        for start in range(0, len(todo), batch_size):
            batch = todo[start : start + batch_size]
            added, balls, bad = load_batch(db, archive, batch)
            loaded += added
            rows += balls
            problems += bad
            say(
                f"{already + loaded + problems} of {len(names)} matches done ({rows} balls this run)"
            )
        return Report(
            loaded_now=loaded,
            rows_now=rows,
            problems_now=problems,
            already_done=already,
            matches_in_zip=len(names),
        )
    finally:
        db.close()


def fingerprint(path: Path) -> tuple[str, int]:
    """The file's SHA-256 and size, read in pieces so a big file needs little memory."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            piece = handle.read(1024 * 1024)
            if not piece:
                break
            digest.update(piece)
            size += len(piece)
    return digest.hexdigest(), size


def match_files(archive: zipfile.ZipFile) -> list[str]:
    """The match files in the zip, in a fixed order (README and LICENSE left out)."""
    names = []
    for name in archive.namelist():
        if name.endswith(".json"):
            names.append(name)
    return sorted(names)


def create_tables(db) -> None:
    for _, create, _ in TABLES:
        db.execute(create)
    for create in BOOKKEEPING:
        db.execute(create)


def check_snapshot(db, sha256: str, size: int, matches: int) -> None:
    """A database holds one snapshot only: mixing two would mix two versions of the data."""
    known = db.execute("SELECT zip_sha256 FROM snapshot").fetchall()
    if not known:
        db.execute(
            "INSERT INTO snapshot VALUES (?, ?, ?, ?)",
            [sha256, size, matches, datetime.now()],
        )
    elif known[0][0] != sha256:
        raise IngestError(
            f"this database holds snapshot {known[0][0][:12]}, not {sha256[:12]}; "
            "use a new database file for a new snapshot"
        )


def finished(db) -> set[str]:
    """Matches an earlier run loaded, or found it could not read."""
    done = set()
    for (match_id,) in db.execute("SELECT match_id FROM loaded").fetchall():
        done.add(match_id)
    for (match_id,) in db.execute("SELECT match_id FROM problems").fetchall():
        done.add(match_id)
    return done


def load_batch(db, archive: zipfile.ZipFile, names: list[str]) -> tuple[int, int, int]:
    """Read and store one batch: either all of it is saved, or none of it."""
    parsed = []
    problems = []
    for name in names:
        match_id = name.removesuffix(".json")
        try:
            data = json.loads(archive.read(name))
            parsed.append(parse_match(match_id, data))
        except (ParseError, json.JSONDecodeError, UnicodeDecodeError) as error:
            problems.append([match_id, str(error)])

    with tempfile.TemporaryDirectory() as folder:
        files = write_csv_files(Path(folder), parsed)
        db.execute("BEGIN TRANSACTION")
        try:
            for table, path in files:
                where = path.as_posix().replace("'", "''")
                # say there is no header: left to guess, DuckDB took a first row
                # of names for one and dropped it (found by a test on Day 31)
                db.execute(
                    f"COPY {table} FROM '{where}' "
                    f"(FORMAT csv, HEADER false, NULL '{NULL}')"
                )
            record_loaded(db, parsed)
            if problems:
                db.executemany("INSERT INTO problems VALUES (?, ?)", problems)
            db.execute("COMMIT")
        except Exception:
            db.execute("ROLLBACK")  # nothing of this batch stays
            raise

    balls = 0
    for match in parsed:
        balls += len(match.deliveries)
    return len(parsed), balls, len(problems)


def record_loaded(db, parsed: list[ParsedMatch]) -> None:
    now = datetime.now()
    rows = []
    for match in parsed:
        rows.append([match.match.match_id, len(match.deliveries), now])
    if rows:
        db.executemany("INSERT INTO loaded VALUES (?, ?, ?)", rows)


def write_csv_files(folder: Path, parsed: list[ParsedMatch]) -> list[tuple[str, Path]]:
    """One CSV file per table; DuckDB reads a CSV far faster than row-by-row inserts."""
    files = []
    for table, _, _ in TABLES:
        path = folder / f"{table}.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            for match in parsed:
                for row in rows_for(match, table):
                    writer.writerow(_cells(row))
        files.append((table, path))
    return files


def rows_for(match: ParsedMatch, table: str) -> tuple:
    if table == "deliveries":
        return match.deliveries
    if table == "matches":
        return (match.match,)
    if table == "innings":
        return match.innings
    return match.people


def _cells(row) -> list:
    cells = []
    for value in row_values(row):
        if value is None:
            cells.append(NULL)
        elif value is True:
            cells.append("true")
        elif value is False:
            cells.append("false")
        else:
            cells.append(value)
    return cells


def main(arguments: list[str]) -> None:
    if len(arguments) != 2:
        print(
            "usage: python -m cricketmind_v3.pipeline.ingest <all_json.zip> <database>"
        )
        sys.exit(2)
    report = ingest(Path(arguments[0]), Path(arguments[1]))
    print(
        f"done: {report.loaded_now} matches added ({report.rows_now} balls), "
        f"{report.problems_now} could not be read, {report.already_done} were already in"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
