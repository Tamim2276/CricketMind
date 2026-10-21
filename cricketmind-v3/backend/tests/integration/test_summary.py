"""Day 37: the summary of what the database holds."""

import duckdb

from cricketmind_v3.pipeline.derived import build_derived
from cricketmind_v3.pipeline.ingest import ingest
from cricketmind_v3.pipeline.players import build_players
from cricketmind_v3.pipeline.summary import (
    matches_by_format,
    summary_report,
    table_rows,
)
from tests.integration.test_ingest import BALLS, make_zip, quiet


def test_rows_per_table_and_formats(tmp_path):
    path = tmp_path / "cricket.duckdb"
    ingest(make_zip(tmp_path), path, say=quiet)
    with duckdb.connect(str(path)) as db:
        build_players(db)
        build_derived(db)
        rows = dict(table_rows(db))
        formats = matches_by_format(db)
        text = summary_report(db, 1234)
    assert rows["deliveries"] == BALLS
    assert rows["players"] == 66
    assert "attributes" not in rows  # not built until the player file is in
    assert formats[0][:4] == ("T20", "male", 2, 499)
    assert "| deliveries | 734 |" in text
    assert "Database file: 1,234 bytes" in text
