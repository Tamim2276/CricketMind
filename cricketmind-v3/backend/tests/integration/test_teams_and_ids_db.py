"""Day 58, part 2c, on a real database: the team register, and a player found by his
register id when his name is not his alone."""

import duckdb
import pytest

from cricketmind_v3.application.ports import FilterError
from cricketmind_v3.infrastructure.duckdb_repo import find_person
from cricketmind_v3.pipeline.ingest import ingest
from cricketmind_v3.pipeline.teams import load_teams
from tests.integration.test_ingest import make_zip, quiet


def test_every_team_that_took_the_field(tmp_path):
    path = tmp_path / "cricket.duckdb"
    ingest(make_zip(tmp_path), path, say=quiet)
    with duckdb.connect(str(path)) as db:
        teams = load_teams(db)
    names = []
    for team in teams:
        names.append(team.name)
    assert names == [
        "Australia",
        "England",
        "Pakistan",
        "Rajasthan Royals",
        "Sri Lanka",
        "Sunrisers Hyderabad",
    ]
    assert (teams[0].matches, teams[0].first_year, teams[0].last_year) == (
        1,
        2026,
        2026,
    )


def test_a_shared_name_is_refused_and_an_id_is_found():
    db = duckdb.connect()
    db.execute(
        "CREATE TABLE players (person_id VARCHAR, name VARCHAR, matches INTEGER)"
    )
    db.execute("INSERT INTO players VALUES ('a1a1a1a1', 'Imran Khan', 125)")
    db.execute("INSERT INTO players VALUES ('b2b2b2b2', 'Imran Khan', 60)")
    db.execute("INSERT INTO players VALUES ('ba607b88', 'V Kohli', 837)")
    with pytest.raises(FilterError, match="2 players are called Imran Khan"):
        find_person(db, "Imran Khan")
    assert find_person(db, "a1a1a1a1") == "a1a1a1a1"  # the one N3 meant (Day 58)
    assert find_person(db, "V Kohli") == "ba607b88"
    assert find_person(db, "Nobody") is None
