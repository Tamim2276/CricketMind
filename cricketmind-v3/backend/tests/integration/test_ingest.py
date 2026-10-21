"""Day 31: the resumable load, on a small zip made from the three fixtures."""

import zipfile
from pathlib import Path

import duckdb
import pytest

from cricketmind_v3.pipeline import ingest as ingest_module
from cricketmind_v3.pipeline.ingest import IngestError, ingest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cricsheet"
IDS = ("1490709", "1535463", "1550709")
BALLS = 235 + 252 + 247  # counted from the files on Day 30


def make_zip(folder: Path, broken: bool = False, name: str = "all_json.zip") -> Path:
    path = folder / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("README.txt", "not a match")  # must be ignored
        for match_id in IDS:
            archive.write(FIXTURES / f"{match_id}.json", f"{match_id}.json")
        if broken:
            archive.writestr("9999999.json", "{ this is not json")
    return path


def count(db_path: Path, table: str) -> int:
    with duckdb.connect(str(db_path)) as db:
        return db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def quiet(message: str) -> None:
    pass


def test_everything_is_loaded(tmp_path):
    report = ingest(make_zip(tmp_path), tmp_path / "cricket.duckdb", say=quiet)
    assert (report.loaded_now, report.rows_now, report.problems_now) == (3, BALLS, 0)
    db = tmp_path / "cricket.duckdb"
    assert count(db, "deliveries") == BALLS
    assert count(db, "matches") == 3
    assert count(db, "innings") == 6
    assert count(db, "people") == 27 + 29 + 27


def test_the_values_survive_the_trip(tmp_path):
    ingest(make_zip(tmp_path), tmp_path / "cricket.duckdb", say=quiet)
    with duckdb.connect(str(tmp_path / "cricket.duckdb")) as db:
        wide = db.execute(
            "SELECT wides, runs_total, phase, player_out, super_over FROM deliveries "
            "WHERE match_id = '1550709' AND innings = 1 AND over = 9 AND ball_index = 4"
        ).fetchone()
        first_innings = db.execute(
            "SELECT SUM(runs_total) FROM deliveries "
            "WHERE match_id = '1550709' AND innings = 1"
        ).fetchone()[0]
        umpire = db.execute(
            "SELECT team FROM people WHERE match_id = '1550709' AND name = 'Faisal Afridi'"
        ).fetchone()[0]
    assert wide == (1, 1, "middle", None, False)  # NULL and false came back right
    assert first_innings == 215
    assert umpire is None


def test_running_twice_adds_nothing(tmp_path):
    archive = make_zip(tmp_path)
    ingest(archive, tmp_path / "cricket.duckdb", say=quiet)
    again = ingest(archive, tmp_path / "cricket.duckdb", say=quiet)
    assert (again.loaded_now, again.already_done) == (0, 3)
    assert count(tmp_path / "cricket.duckdb", "deliveries") == BALLS


def test_stopping_and_restarting(tmp_path):
    archive = make_zip(tmp_path)
    first = ingest(archive, tmp_path / "cricket.duckdb", stop_after=1, say=quiet)
    assert first.loaded_now == 1
    second = ingest(archive, tmp_path / "cricket.duckdb", say=quiet)
    assert (second.loaded_now, second.already_done) == (2, 1)
    assert count(tmp_path / "cricket.duckdb", "deliveries") == BALLS  # each ball once


def test_a_crash_inside_a_batch_leaves_nothing_half_saved(tmp_path, monkeypatch):
    archive = make_zip(tmp_path)

    def power_cut(db, parsed):
        raise OSError("the power went out")  # after the balls were copied in

    monkeypatch.setattr(ingest_module, "record_loaded", power_cut)
    with pytest.raises(OSError):
        ingest(archive, tmp_path / "cricket.duckdb", say=quiet)
    assert count(tmp_path / "cricket.duckdb", "deliveries") == 0  # rolled back

    monkeypatch.undo()
    report = ingest(archive, tmp_path / "cricket.duckdb", say=quiet)
    assert report.loaded_now == 3
    assert count(tmp_path / "cricket.duckdb", "deliveries") == BALLS


def test_a_broken_file_is_written_down_not_fatal(tmp_path):
    report = ingest(make_zip(tmp_path, broken=True), tmp_path / "c.duckdb", say=quiet)
    assert (report.loaded_now, report.problems_now) == (3, 1)
    with duckdb.connect(str(tmp_path / "c.duckdb")) as db:
        match_id, error = db.execute("SELECT * FROM problems").fetchone()
    assert match_id == "9999999"
    assert "Expecting" in error  # the JSON reader's own message


def test_a_different_snapshot_is_refused(tmp_path):
    ingest(make_zip(tmp_path), tmp_path / "cricket.duckdb", say=quiet)
    other = make_zip(tmp_path, broken=True, name="newer.zip")
    with pytest.raises(IngestError):
        ingest(other, tmp_path / "cricket.duckdb", say=quiet)


def test_the_snapshot_is_recorded(tmp_path):
    archive = make_zip(tmp_path)
    ingest(archive, tmp_path / "cricket.duckdb", say=quiet)
    sha256, size = ingest_module.fingerprint(archive)
    with duckdb.connect(str(tmp_path / "cricket.duckdb")) as db:
        row = db.execute(
            "SELECT zip_sha256, zip_bytes, matches_in_zip FROM snapshot"
        ).fetchone()
    assert row == (sha256, size, 3)


def test_the_first_row_of_every_table_is_kept(tmp_path):
    # found on Day 31: left to guess, DuckDB read a first row of names as a header
    ingest(make_zip(tmp_path), tmp_path / "cricket.duckdb", say=quiet)
    with duckdb.connect(str(tmp_path / "cricket.duckdb")) as db:
        capsey = db.execute(
            "SELECT person_id FROM people WHERE match_id = '1490709' AND name = 'A Capsey'"
        ).fetchall()
    assert len(capsey) == 1
