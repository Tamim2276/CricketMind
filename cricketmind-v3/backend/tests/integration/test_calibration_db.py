"""Day 53: the test year's moments, from DuckDB.

Counted by hand (Days 38, 50): Sahibzada Farhan faced 44 balls on 2026-10-11, the last
one caught; his one wide is not a ball faced. The men's club match (2026-05-27) holds 238
balls faced, the women's (2026-07-05) 226, the men's international 241.
"""

import pytest

from cricketmind_v3.domain.balls import batter_out, faced
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from tests.integration.test_duckdb_repo import build

FARHAN = CricketSubject(player="Sahibzada Farhan")


@pytest.fixture
def repo(tmp_path):
    db = build(tmp_path)
    yield DuckDBDeliveryRepository(db)
    db.close()


def test_his_balls_as_moments(repo):
    found = repo.moments(FARHAN)
    assert len(found) == 44  # the wide is left out: not faced, not out
    before = []
    for moment in found:
        before.append(moment.faced_before)
        assert moment.faced
        assert moment.order == "top"
    assert before == list(range(44))
    outs = []
    for moment in found:
        outs.append(moment.out)
    assert outs == [False] * 43 + [True]
    assert found[-1].bowler == repo.person("MD Shanaka")


def test_a_wide_he_is_out_on_is_a_moment(repo):
    # make his wide a stumping: it is then a ball he was out on, though not one he faced
    repo.db.execute(
        "UPDATE deliveries SET player_out = batter, wicket_kind = 'stumped', wickets = 1 "
        "WHERE match_id = '1550709' AND batter = 'Sahibzada Farhan' AND wides > 0"
    )
    found = repo.moments(FARHAN)
    assert len(found) == 45
    wide = []
    for moment in found:
        if not moment.faced:
            wide.append(moment)
    assert len(wide) == 1
    assert wide[0].out
    assert wide[0].faced_before == 38  # he had faced 38 balls: it came before his 39th


def test_the_year_is_split_by_date(repo):
    men = CricketSubject(match_format="T20", gender="male")
    women = CricketSubject(match_format="T20", gender="female")
    earlier = repo.moments(
        CricketSubject(match_format="T20", gender="male", before="2026-10-11")
    )
    later = repo.moments(
        CricketSubject(match_format="T20", gender="male", since="2026-10-11")
    )
    assert len(earlier) == 238  # the club match only
    assert len(later) == 241  # the international only
    assert len(repo.moments(men)) == 238 + 241
    assert len(repo.moments(women)) == 226


def test_moments_count_as_python_does(repo):
    every = CricketSubject()
    counted = 0
    for ball in repo.deliveries(every):
        if faced(ball) or batter_out(ball):
            counted += 1
    assert len(repo.moments(every)) == counted
