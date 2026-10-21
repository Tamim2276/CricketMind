"""Day 41: the M3 engine on a tiny DuckDB file made from the three fixture matches.

Counted by hand from the files: E Malinga bowled 25 balls in the IPL match of
2026-05-27 (balls 6 to 105 of its 252) and 26 in the T20I of 2026-10-11 (balls 30 to
124 of its 247), wides and no-balls included. The last match in the data is that T20I.
"""

from datetime import date

import duckdb
import pytest

from cricketmind_v3.application.engines.m3_workload import workload_ratio_flag
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.domain.workload import balls_by_day
from cricketmind_v3.infrastructure.duckdb_workload import DuckDBWorkloadRepository
from cricketmind_v3.pipeline.ingest import ingest
from cricketmind_v3.pipeline.players import build_players
from tests.integration.test_ingest import make_zip, quiet


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "cricket.duckdb"
    ingest(make_zip(tmp_path), path, say=quiet)
    connection = duckdb.connect(str(path))
    build_players(connection)
    yield connection
    connection.close()


@pytest.fixture
def repo(db):
    return DuckDBWorkloadRepository(db)


def malinga_on(day: str) -> CricketSubject:
    return CricketSubject(bowler="E Malinga", date=day)


def test_his_spells_in_both_matches(repo):
    first, second = repo.spells("E Malinga")
    assert (first.first_day, first.days, first.match_balls) == (
        date(2026, 5, 27),
        1,
        252,
    )
    assert len(first.positions) == 25
    assert (first.positions[0], first.positions[-1]) == (6, 105)
    assert (second.first_day, second.match_balls, len(second.positions)) == (
        date(2026, 10, 11),
        247,
        26,
    )


def test_a_spike_on_the_day_of_the_t20i(repo):
    # the week and the four weeks both hold the T20I's 26 balls; May is long before
    assert workload_ratio_flag(repo, malinga_on("2026-10-11")) == Fact(value=4.0, n=26)


def test_no_answer_after_the_data_ends(repo):
    assert repo.data_until() == date(2026, 10, 11)
    assert workload_ratio_flag(repo, malinga_on("2026-10-12")) is None


def test_no_answer_for_a_newcomer_or_a_quiet_month(repo):
    assert workload_ratio_flag(repo, malinga_on("2026-06-01")) is None  # first match
    assert workload_ratio_flag(repo, malinga_on("2026-09-01")) is None  # nothing bowled


def test_a_two_day_match_spreads_its_balls(db, repo):
    # as if the T20I had run over two days: only his last ball, number 124 of 247,
    # comes after the halfway mark
    db.execute("UPDATE matches SET days = 2 WHERE match_id = '1550709'")
    by_day = balls_by_day(repo.spells("E Malinga"))
    assert by_day[date(2026, 10, 11)] == 25
    assert by_day[date(2026, 10, 12)] == 1


def test_a_name_two_players_share_gives_no_answer(db, repo):
    db.execute("INSERT INTO players VALUES ('ffffffff', 'E Malinga', 1)")
    assert workload_ratio_flag(repo, malinga_on("2026-10-11")) is None


def test_nobody_of_that_name(repo):
    assert repo.spells("DG Bradman") == []
    nobody = CricketSubject(bowler="DG Bradman", date="2026-10-11")
    assert workload_ratio_flag(repo, nobody) is None
