"""Day 50: how long the batter had been in at each ball, in DuckDB.

Hand-made innings first; then Sahibzada Farhan's 44 balls in 1550709.json, counted by
hand from the file: his first 20 balls were in the powerplay, his 44th was his last
(caught), and a wide came between his 38th and 39th.
"""

import duckdb
import pytest

from cricketmind_v3.domain.balls import tally
from cricketmind_v3.domain.measures import outs_and_faced
from cricketmind_v3.domain.stages import by_batter, stage_of
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_repo import (
    DuckDBDeliveryRepository,
    stage_case,
)
from cricketmind_v3.pipeline.derived import BALL_STAGES
from cricketmind_v3.pipeline.schema import CREATE_DELIVERIES
from tests.integration.test_duckdb_repo import build

FARHAN = "Sahibzada Farhan"

COLUMNS = (
    "match_id, match_type, gender, team_type, date, venue, innings, batting_team, "
    "super_over, over, ball_index, batter, bowler, non_striker, runs_batter, "
    "runs_extras, runs_total, wides, noballs, byes, legbyes, penalty, wickets, "
    "player_out, wicket_kind, review_umpires_call"
)


def hand_made(
    db, innings, over, index, batter, wides=0, noballs=0, out=None, kind=None
):
    super_over = innings == 3
    db.execute(
        f"INSERT INTO deliveries ({COLUMNS}) VALUES "
        "('m1', 'T20', 'male', 'club', '2024-01-01', 'Ground', ?, 'A', ?, ?, ?, ?, "
        "'Bowler', 'Partner', 0, ?, ?, ?, ?, 0, 0, 0, ?, ?, ?, false)",
        [
            innings,
            super_over,
            over,
            index,
            batter,
            wides + noballs,
            wides + noballs,
            wides,
            noballs,
            1 if out else 0,
            out,
            kind,
        ],
    )


def test_hand_made_innings():
    db = duckdb.connect()
    db.execute(CREATE_DELIVERIES)
    # innings 1, over 0
    hand_made(db, 1, 0, 1, "A")  # A has faced 0
    hand_made(db, 1, 0, 2, "A", wides=1)  # 1: a wide is not a ball faced...
    hand_made(db, 1, 0, 3, "A", noballs=1)  # 1: ...so this is still his second
    hand_made(db, 1, 0, 4, "A")  # 2: a no-ball is a ball faced
    hand_made(db, 1, 0, 5, "B")  # B has faced 0: his own count
    hand_made(db, 1, 0, 6, "B")  # 1
    hand_made(db, 1, 0, 7, "A")  # 3
    # over 1: A is out on his fifth ball faced; C comes in, and is stumped off a wide
    hand_made(db, 1, 1, 1, "A", out="A", kind="caught")  # 4
    hand_made(db, 1, 1, 2, "C")  # 0
    hand_made(db, 1, 1, 3, "C", wides=1, out="C", kind="stumped")  # 1
    # innings 2: A starts again; innings 3 is a super over, left out
    hand_made(db, 2, 0, 1, "A")  # 0
    hand_made(db, 3, 0, 1, "A")
    db.execute(BALL_STAGES)
    rows = db.execute(
        "SELECT s.innings, s.over, s.ball_index, d.batter, s.faced_before "
        "FROM ball_stages s JOIN deliveries d USING (match_id, innings, over, ball_index) "
        "ORDER BY s.innings, s.over, s.ball_index"
    ).fetchall()
    found = []
    for innings, over, index, batter, before in rows:
        found.append((innings, over, index, batter, before))
    assert found == [
        (1, 0, 1, "A", 0),
        (1, 0, 2, "A", 1),
        (1, 0, 3, "A", 1),
        (1, 0, 4, "A", 2),
        (1, 0, 5, "B", 0),
        (1, 0, 6, "B", 1),
        (1, 0, 7, "A", 3),
        (1, 1, 1, "A", 4),
        (1, 1, 2, "C", 0),
        (1, 1, 3, "C", 1),
        (2, 0, 1, "A", 0),
    ]
    db.close()


def test_the_sql_stage_is_the_python_stage():
    db = duckdb.connect()
    rows = db.execute(
        f"SELECT x, {stage_case('x')} FROM range(0, 40) t(x) ORDER BY x"
    ).fetchall()
    for faced_before, stage in rows:
        assert stage == stage_of(faced_before + 1), faced_before
    db.close()


# On the fixture matches


@pytest.fixture
def repo(tmp_path):
    db = build(tmp_path)
    yield DuckDBDeliveryRepository(db)
    db.close()


def stages(records) -> list[tuple[int, int, int]]:
    found = []
    for record in records:
        found.append((record.stage, record.outs, record.faced))
    return found


def test_his_innings_by_stage(repo):
    # balls 1-5, 6-15, 16-30, then 31-44 (the wide is not one), out on the last
    whole = repo.records(CricketSubject(player=FARHAN))
    assert stages(whole) == [(1, 0, 5), (2, 0, 10), (3, 0, 15), (4, 1, 14)]


def test_a_filter_does_not_start_him_again(repo):
    # the powerplay: his balls 1 to 20
    powerplay = repo.records(CricketSubject(player=FARHAN, phase="powerplay"))
    assert stages(powerplay) == [(1, 0, 5), (2, 0, 10), (3, 0, 5)]
    # the middle overs begin at his ball 21: stage 3, not stage 1
    middle = repo.records(CricketSubject(player=FARHAN, phase="middle"))
    assert stages(middle) == [(3, 0, 10), (4, 1, 14)]


def counts_match(repo) -> int:
    # every batter in the fixtures, whole innings and powerplay: the stage records
    # add up to what balls.py counts from his balls
    names = repo.db.execute("SELECT DISTINCT batter FROM deliveries").fetchall()
    checked = 0
    for (name,) in names:
        for phase in (None, "powerplay"):
            subject = CricketSubject(player=name, phase=phase)
            counted = outs_and_faced(tally(repo.deliveries(subject)))
            if counted == (0, 0):
                continue
            assert by_batter(repo.records(subject)) == [counted], name
            checked += 1
    return checked


def test_duckdb_counts_as_python_does(repo):
    assert counts_match(repo) > 60


def test_outs_that_are_not_his(repo):
    # Farhan's first ball becomes Fakhar Zaman's run out at the other end, and the
    # batter on strike at the start of over 1 retires hurt: neither is the striker out
    repo.db.execute(
        "UPDATE deliveries SET player_out = 'Fakhar Zaman', wicket_kind = 'run out', "
        "wickets = 1 WHERE match_id = '1550709' AND innings = 1 AND over = 0 "
        "AND ball_index = 1"
    )
    repo.db.execute(
        "UPDATE deliveries SET player_out = batter, wicket_kind = 'retired hurt', "
        "wickets = 1 WHERE match_id = '1550709' AND innings = 1 AND over = 1 "
        "AND ball_index = 1"
    )
    whole = repo.records(CricketSubject(player=FARHAN))
    assert stages(whole) == [(1, 0, 5), (2, 0, 10), (3, 0, 15), (4, 1, 14)]
    assert counts_match(repo) > 60


def test_the_others_are_everyone_but_him(repo):
    like_him = CricketSubject(player=FARHAN, match_format="T20", gender="male")
    others = repo.others_records(like_him)
    farhan_id = repo.db.execute(
        "SELECT person_id FROM players WHERE name = ?", [FARHAN]
    ).fetchone()[0]
    batters = set()
    for record in others:
        assert record.batter != farhan_id
        batters.add(record.batter)
    assert len(batters) == 32  # the other men's T20 batters
    outs = 0
    faced = 0
    for record in others:
        outs += record.outs
        faced += record.faced
    assert (outs, faced) == (27, 435)
