"""Day 32: the data checks, on the fixtures, then on deliberately spoiled copies."""

import duckdb
import pytest

from cricketmind_v3.pipeline.ingest import ingest
from cricketmind_v3.pipeline.quality import report, run
from tests.integration.test_ingest import make_zip, quiet


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "cricket.duckdb"
    ingest(make_zip(tmp_path), path, say=quiet)
    connection = duckdb.connect(str(path))
    yield connection
    connection.close()


def counts(db) -> dict[str, int]:
    found = {}
    for outcome in run(db):
        found[outcome.check.name] = outcome.count
    return found


def test_the_fixtures_keep_every_rule(db):
    for name, count in counts(db).items():
        assert count == 0, name


def test_a_wrong_ball_total_is_caught_three_ways(db):
    # one ball of Pakistan's innings told wrongly: 5 runs more than it was
    db.execute(
        "UPDATE deliveries SET runs_total = runs_total + 5 "
        "WHERE match_id = '1550709' AND innings = 1 AND over = 0 AND ball_index = 1"
    )
    found = counts(db)
    assert found["ball totals"] == 1
    assert found["targets"] == 1  # 220 + 1 is not the target of 216
    assert found["winning margins"] == 1  # 220 - 133 is not 82


def test_extras_that_do_not_add_up(db):
    db.execute(
        "UPDATE deliveries SET wides = 2 "
        "WHERE match_id = '1550709' AND innings = 1 AND over = 9 AND ball_index = 4"
    )
    assert counts(db)["extras add up"] == 1


def test_a_lost_ball_is_caught(db):
    db.execute(
        "DELETE FROM deliveries "
        "WHERE match_id = '1535463' AND innings = 2 AND over = 0 AND ball_index = 1"
    )
    assert counts(db)["ball counts"] == 1


def test_a_name_without_an_id_is_caught(db):
    db.execute("DELETE FROM people WHERE match_id = '1550709' AND name = 'E Malinga'")
    assert counts(db)["names without an id"] == 1


def test_a_shortened_powerplay_is_counted_not_called_a_problem(db):
    db.execute(
        "UPDATE innings SET powerplay_to = 2.6 WHERE match_id = '1550709' AND innings = 1"
    )
    outcomes = run(db)
    text = report(outcomes)
    assert counts(db)["powerplays unlike the rule"] == 1
    assert "PROBLEM" not in text  # real, but not an error in the data


def test_the_report_names_a_problem(db):
    db.execute(
        "UPDATE deliveries SET batter = '' WHERE match_id = '1490709' AND over = 3"
    )
    assert "| empty names |" in report(run(db))
    assert "PROBLEM" in report(run(db))


def test_a_rain_target_for_fewer_overs_is_not_a_problem(db):
    # found on Day 32: a chase reset to fewer overs, with "no result" and no rain rule named
    db.execute("UPDATE matches SET overs = 50 WHERE match_id = '1550709'")
    db.execute(
        "UPDATE innings SET target_runs = 150, target_overs = 33 "
        "WHERE match_id = '1550709' AND innings = 2"
    )
    assert counts(db)["targets"] == 0
