"""Day 38: the M1 engine on a tiny DuckDB file made from the three fixture matches.

Every expected number was counted by hand from 1550709.json (Pakistan v Sri Lanka):
Sahibzada Farhan faced 44 balls for 73 runs, out caught off MD Shanaka in over 14. He
also had a wide (not a ball faced) and a no-ball (a ball faced), both from PHKD Mendis.
"""

import duckdb
import pytest

from cricketmind_v3.application.engines.m1_rates import (
    dismissal_rate,
    sr_vs_average,
    strike_rate,
)
from cricketmind_v3.application.ports import FilterError
from cricketmind_v3.domain import measures
from cricketmind_v3.domain.balls import (
    SOMEONE_ELSE,
    THE_BATTER,
    by_format_and_gender,
    tally,
)
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from cricketmind_v3.pipeline.attributes import build_attributes, read_player_meta
from cricketmind_v3.pipeline.derived import build_derived
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
    build_derived(db)  # one name per ground, for the venue filter (Day 43)
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


# Day 39: everyone else, counted by DuckDB. All three matches, counted by hand from the
# files: 705 balls faced, 1,034 runs off the bat, 35 batters out. Fakhar Zaman: 22 off
# 17, out once. The Lord's match is women's (226 balls faced, 278 runs): the two men's
# matches hold 479 and 756 (Day 42).


def test_everyone_else_comes_counted_with_the_names_given_way(repo):
    others = repo.others(CricketSubject(player="Fakhar Zaman"))
    assert measures.strike_rate(others) == Fact(value=100.0 * 1012 / 688, n=688)
    assert measures.dismissal_rate(others) == Fact(value=34 / 688, n=688)
    assert len(others) < 50  # alike balls counted together, not 700 rows
    for ball in others:
        assert ball.batter == THE_BATTER


def test_a_non_strikers_run_out_is_not_the_batters(db, repo):
    # the first ball of 1490709, AE Jones on strike: make it DN Wyatt's run out
    db.execute(
        "UPDATE deliveries SET player_out = non_striker, wicket_kind = 'run out', "
        "wickets = 1 WHERE match_id = '1490709' AND innings = 1 AND over = 0 "
        "AND ball_index = 1"
    )
    others = repo.others(CricketSubject(player="Fakhar Zaman"))
    assert measures.dismissal_rate(others) == Fact(value=34 / 688, n=688)
    run_outs = 0
    for ball, times in others.items():
        if ball.player_out == SOMEONE_ELSE:
            run_outs += times
    assert run_outs == 1


def test_a_player_against_everyone_else(repo):
    # men only (Day 42): 479 - 17 = 462 balls, 756 - 22 = 734 runs
    found = sr_vs_average(repo, CricketSubject(player="Fakhar Zaman"))
    assert found.value == 100.0 * 22 / 17 - 100.0 * 734 / 462  # 129.4 - 158.9
    assert found.n == 17
    low, high = found.interval
    assert low < 0 < high  # 17 balls: not enough to call him slower


def test_no_average_when_the_player_is_a_big_part_of_it(repo):
    # Farhan's 44 balls against 435 of the other men's: 10 times, not 20
    assert sr_vs_average(repo, CricketSubject(player=FARHAN)) is None


def test_everyone_else_follows_the_filters(repo):
    # his powerplay balls and everyone else's make all the powerplay balls
    subject = CricketSubject(player="Fakhar Zaman", phase="powerplay")
    his = measures.strike_rate(tally(repo.deliveries(subject))).n
    theirs = measures.strike_rate(repo.others(subject)).n
    all_of_them = repo.db.execute(
        "SELECT COUNT(*) FROM deliveries WHERE phase = 'powerplay' AND wides = 0"
    ).fetchone()[0]
    assert his + theirs == all_of_them
    assert repo.others(CricketSubject(player="DG Bradman")) == {}  # nobody of that name


def test_everyone_else_is_split_by_format_and_gender(db, repo):
    # call the IPL match an ODI. The IPL match: 238 balls faced, 417 runs; the women's
    # match at Lord's: 226 and 278. Everyone else's men's T20 balls are then
    # Rawalpindi's alone: 241 - 17 = 224 balls, 339 - 22 = 317 runs
    db.execute("UPDATE deliveries SET match_type = 'ODI' WHERE match_id = '1535463'")
    split = by_format_and_gender(repo.others(CricketSubject(player="Fakhar Zaman")))
    assert measures.strike_rate(split[("ODI", "male")]) == Fact(
        value=100.0 * 417 / 238, n=238
    )
    assert measures.strike_rate(split[("T20", "male")]) == Fact(
        value=100.0 * 317 / 224, n=224
    )
    assert measures.strike_rate(split[("T20", "female")]) == Fact(
        value=100.0 * 278 / 226, n=226
    )
    # Hassan Nawaz (19 off 9 at Rawalpindi) is set against the other men's T20 balls
    # only: 241 - 9 = 232 balls, 339 - 19 = 320 runs
    found = sr_vs_average(repo, CricketSubject(player="Hassan Nawaz"))
    assert found.value == 100.0 * 19 / 9 - 100.0 * 320 / 232
