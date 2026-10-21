"""Phase 3 in one command: build the database from the Cricsheet snapshot, check it,
and record what it holds.

Run from backend/:
    uv run python ../thesis/scripts/build_data.py

It uses data/raw/all_json.zip, data/raw/player_meta.csv and, if you made one,
data/hand_labels.csv, and builds data/cricket.duckdb. Safe to stop and run again: the load
carries on where it stopped, and every other step is rebuilt. It writes three reports to
thesis/results/ (data_checks.md, data_coverage.md, data_summary.md), each with a line
in ledger.jsonl.
"""

import argparse
from pathlib import Path

import duckdb
from ledger import PROJECT, RESULTS, add_to_ledger, relative, write_safely

from cricketmind_v3.pipeline.attributes import (
    build_attributes,
    coverage,
    coverage_report,
    read_hand_labels,
    read_player_meta,
)
from cricketmind_v3.pipeline.derived import build_derived
from cricketmind_v3.pipeline.ingest import ingest
from cricketmind_v3.pipeline.players import build_players
from cricketmind_v3.pipeline.quality import report, run
from cricketmind_v3.pipeline.summary import summary_report

DATA = PROJECT / "data"


def main() -> None:
    options = arguments()
    results = Path(options.results)

    print("1/5 loading the matches (stop any time; run again to carry on)")
    ingest(Path(options.zip), Path(options.database))

    with duckdb.connect(options.database) as db:
        snapshot = db.execute("SELECT zip_sha256 FROM snapshot").fetchone()[0][:12]
        data = f"real: Cricsheet snapshot {snapshot}"

        print("2/5 checking the data against itself")
        checks = report(run(db))
        save(results / "data_checks.md", "Data checks", checks, data, options)

        print("3/5 players and their names")
        print(f"    {build_players(db)} players")

        print("4/5 batting hands and bowler types")
        meta_path = Path(options.player_meta)
        if meta_path.exists():
            hand = []
            if Path(options.hand_labels).exists():
                hand = read_hand_labels(Path(options.hand_labels))
            build_attributes(db, read_player_meta(meta_path), hand)
            text = coverage_report(coverage(db))
            save(results / "data_coverage.md", "Coverage", text, data, options)
        else:
            print(f"    skipped: {relative(meta_path)} is not there yet")

        print("5/5 venues, innings, workloads and reviews")
        for table, rows in build_derived(db).items():
            print(f"    {table}: {rows:,} rows")

    with duckdb.connect(options.database, read_only=True) as db:
        size = Path(options.database).stat().st_size
        save(
            results / "data_summary.md",
            "Summary",
            summary_report(db, size),
            data,
            options,
        )
    print("done")


def save(path: Path, title: str, text: str, data: str, options) -> None:
    write_safely(path, f"# {title} (Phase 3)\n\n{text}\n")
    add_to_ledger(
        {
            "experiment": f"Phase 3 data: {title.lower()}",
            "data": data,
            "script": "thesis/scripts/build_data.py",
        },
        path,
        ledger=Path(options.results) / "ledger.jsonl",
    )
    print(f"    wrote {relative(path)}")


def arguments():
    parser = argparse.ArgumentParser(description="Build the cricket database.")
    parser.add_argument("--zip", default=str(DATA / "raw" / "all_json.zip"))
    parser.add_argument("--database", default=str(DATA / "cricket.duckdb"))
    parser.add_argument("--player-meta", default=str(DATA / "raw" / "player_meta.csv"))
    parser.add_argument("--hand-labels", default=str(DATA / "hand_labels.csv"))
    parser.add_argument("--results", default=str(RESULTS))
    return parser.parse_args()


if __name__ == "__main__":
    main()
