"""Day 38: the M1 engine on a tiny DuckDB file made from the three fixture matches.

Every expected number was counted by hand from 1550709.json (Pakistan v Sri Lanka):
Sahibzada Farhan faced 44 balls for 73 runs, out caught off MD Shanaka in over 14. He
also had a wide (not a ball faced) and a no-ball (a ball faced), both from PHKD Mendis.
"""

import duckdb
import pytest

from cricketmind_v3.application.engines.m1_rates import dismissal_rate, strike_rate
from cricketmind_v3.application.ports import FilterError
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from cricketmind_v3.pipeline.attributes import build_attributes, read_player_meta
from cricketmind_v3.pipeline.ingest import ingest
from cricketmind_v3.pipeline.players import build_players
from tests.integration.test_attributes_join import SAMPLE
from tests.integration.test_ingest import make_zip, quiet

FARHAN = "Sahibzada Farhan"


def build(folder, change=None):
    """The fixtures loaded, with players and attributes; change() runs before players."""
    path = folder / "cricket.duckdb"
    ingest(make_zip(folder), path, say=quiet)
    db = duckdb.connect(str(path))
    if change is not None:
        change(db)
    build_players(db)
    build_attributes(db, read_player_meta(SAMPLE), [])
    return db


@pytest.fixture
def db(tmp_path):
    connection = build(tmp_path)
    yield connection
    connection.close()


@pytest.fixture
def repo(db):
    return DuckDBDeliveryRepository(db)


# The plan's test: the numbers counted by hand


def test_a_whole_innings(repo):
    subject = CricketSubject(player=FARHAN)
    # 44 balls: the no-ball is in, the wide is out (45 would give 162.2, 43 gives 169.8)
    assert strike_rate(repo, subject) == Fact(value=100.0 * 73 / 44, n=44)
    assert dismissal_rate(repo, subject) == Fact(value=1 / 44, n=44)


def test_the_powerplay_only(repo):
    subject = CricketSubject(player=FARHAN, phase="powerplay")
    assert strike_rate(repo, subject) == Fact(value=120.0, n=20)  # 24 off 20
    assert dismissal_rate(repo, subject) == Fact(value=0.0, n=20)  # out in the middle


def test_against_pace_and_spin(repo):
    # pace: B Fernando 17 off 10, E Malinga 11 off 7
    pace = CricketSubject(player=FARHAN, bowler_type="pace")
    assert strike_rate(repo, pace) == Fact(value=100.0 * 28 / 17, n=17)
    # spin: M Theekshana 25 off 10, PHKD Mendis 2 off 2 (the no-ball in, the wide out)
    spin = CricketSubject(player=FARHAN, bowler_type="spin")
    assert strike_rate(repo, spin) == Fact(value=100.0 * 27 / 12, n=12)
    # N Thushara and MD Shanaka have no type in the sample: 15 balls in neither


def test_against_one_bowler(repo):
    theekshana = CricketSubject(player=FARHAN, bowler="M Theekshana")
    assert strike_rate(repo, theekshana) == Fact(value=250.0, n=10)
    shanaka = CricketSubject(player=FARHAN, bowler="MD Shanaka")
    assert dismissal_rate(repo, shanaka) == Fact(value=1 / 8, n=8)


# People, not names


def test_one_player_two_teams(repo):
    # MD Shanaka: 5 off 3 for Rajasthan Royals, 9 off 8 for Sri Lanka, out in both
    subject = CricketSubject(player="MD Shanaka")
    assert strike_rate(repo, subject) == Fact(value=100.0 * 14 / 11, n=11)
    assert dismissal_rate(repo, subject) == Fact(value=2 / 11, n=11)


def test_one_player_two_names(tmp_path):
    def rename_in_the_ipl_match(db):
        for column in ("batter", "non_striker", "bowler", "player_out"):
            db.execute(
                f"UPDATE deliveries SET {column} = 'Dasun Shanaka' "
                f"WHERE match_id = '1535463' AND {column} = 'MD Shanaka'"
            )
        db.execute(
            "UPDATE people SET name = 'Dasun Shanaka' "
            "WHERE match_id = '1535463' AND name = 'MD Shanaka'"
        )

    db = build(tmp_path, change=rename_in_the_ipl_match)
    repo = DuckDBDeliveryRepository(db)
    # one register id, so both matches still count (by name it would be 3 balls); on a
    # tie the players table keeps the name that sorts first
    balls = repo.deliveries(CricketSubject(player="Dasun Shanaka"))
    assert len(balls) == 11
    # in match order, each ball keeps the name used in its match: May first, then October
    assert balls[0].batter == "Dasun Shanaka"
    assert balls[-1].batter == "MD Shanaka"
    db.close()


def test_a_name_two_players_share_is_refused(db, repo):
    db.execute("INSERT INTO players VALUES ('ffffffff', 'Fakhar Zaman', 1)")
    subject = CricketSubject(player="Fakhar Zaman")
    with pytest.raises(FilterError, match="2 players are called Fakhar Zaman"):
        repo.deliveries(subject)
    assert strike_rate(repo, subject) is None  # never a guess


def test_nobody_of_that_name(repo):
    assert repo.deliveries(CricketSubject(player="DG Bradman")) == []
    assert strike_rate(repo, CricketSubject(player="DG Bradman")) is None


# What it must not do


def test_a_filter_it_cannot_apply_is_never_left_out(repo):
    subject = CricketSubject(player=FARHAN, shot="Cover Drive")
    with pytest.raises(FilterError, match="shot"):
        repo.deliveries(subject)
    assert strike_rate(repo, subject) is None  # not his strike rate on every shot


def test_a_super_over_is_not_counted(db, repo):
    # a copy of one of his balls, as if bowled in a super over
    db.execute(
        "INSERT INTO deliveries SELECT * REPLACE (3 AS innings, true AS super_over) "
        "FROM deliveries WHERE match_id = '1550709' AND batter = ? LIMIT 1",
        [FARHAN],
    )
    assert strike_rate(repo, CricketSubject(player=FARHAN)).n == 44


def test_the_same_question_gives_the_same_balls(repo):
    subject = CricketSubject(player="MD Shanaka")
    assert repo.deliveries(subject) == repo.deliveries(subject)
