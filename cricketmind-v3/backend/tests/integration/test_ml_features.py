"""Day 54: the ML baseline's inputs, on the tiny DuckDB.

Counted by hand from the fixture files: before Farhan's last ball (caught, over 14),
Pakistan were 143 for 1 after 88 legal balls. In the IPL match of 2026-05-27, Shanaka
faced 3 balls and was out once; he did not bowl. Here the "test year" starts on
2026-10-01, so the May and July matches are the train ones.
"""

import numpy as np
import pytest

from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from cricketmind_v3.infrastructure.ml_features import column_names, features, matrix
from tests.integration.test_duckdb_repo import build

CUTOFF = "2026-10-01"


@pytest.fixture
def db(tmp_path):
    connection = build(tmp_path)
    yield connection
    connection.close()


def rows_of(found: dict, column: str, value) -> list[int]:
    places = []
    for i, cell in enumerate(found[column]):
        if cell == value:
            places.append(i)
    return places


def test_one_row_for_every_moment_of_day_53(db):
    found = features(db, CUTOFF)
    moments = DuckDBDeliveryRepository(db).moments(CricketSubject())
    assert len(found["out"]) == len(moments)  # faced, or out on: 705 balls
    outs = 0
    for moment in moments:
        outs += moment.out
    assert int(found["out"].sum()) == outs


def test_the_match_before_the_ball(db):
    found = features(db, CUTOFF)
    farhan = DuckDBDeliveryRepository(db).person("Sahibzada Farhan")
    his = rows_of(found, "batter", farhan)
    assert len(his) == 44
    first = his[0]
    assert (found["wickets_down"][first], found["runs_so_far"][first]) == (0, 0)
    last = his[-1]
    assert found["out"][last] == 1
    assert found["wickets_down"][last] == 1  # one down, by hand
    assert found["runs_so_far"][last] == 143
    assert found["legal_so_far"][last] == 88
    assert found["faced_before"][last] == 43
    assert found["batting_order"][last] == "top"
    assert found["overs"][last] == 20


def test_records_come_only_from_earlier_train_matches(db):
    found = features(db, CUTOFF)
    repo = DuckDBDeliveryRepository(db)
    shanaka = repo.person("MD Shanaka")
    his = rows_of(found, "batter", shanaka)
    early = []
    late = []
    for i in his:
        if found["test"][i]:
            late.append(i)
        else:
            early.append(i)
    assert len(early) == 3  # the IPL match: nothing before it
    for i in early:
        assert (found["batter_outs"][i], found["batter_balls"][i]) == (0, 0)
    assert len(late) == 8  # the T20I: his IPL record, 1 out in 3 balls
    for i in late:
        assert (found["batter_outs"][i], found["batter_balls"][i]) == (1, 3)
    # as a bowler in the T20I: he did not bowl in May, so nothing before
    bowled = rows_of(found, "bowler", shanaka)
    assert bowled
    for i in bowled:
        assert (found["bowler_outs"][i], found["bowler_balls"][i]) == (0, 0)


def test_a_test_record_never_holds_a_test_match(db):
    # with every match in the test year, nobody has a record at all
    found = features(db, "2026-01-01")
    for column in ("batter_outs", "batter_balls", "bowler_outs", "bowler_balls"):
        assert int(found[column].sum()) == 0, column


def test_nobody_has_a_record_in_the_first_match(db):
    # the IPL match of 2026-05-27 comes first: no batter and no bowler has played before
    found = features(db, CUTOFF)
    first = rows_of(found, "match_id", "1535463")
    assert first
    for i in first:
        for column in ("batter_outs", "batter_balls", "bowler_outs", "bowler_balls"):
            assert found[column][i] == 0, column


def test_the_model_reads_numbers(db):
    found = features(db, CUTOFF)
    priors = {}
    for kind in ("men_t20", "women_t20"):
        for order in ("top", "middle", "tail"):
            priors[(kind, order)] = (1.0, 19.0)  # 5%, worth 20 balls
    numbers = matrix(found, priors, {"men_t20": (1.0, 19.0), "women_t20": (1.0, 19.0)})
    names = column_names()
    assert numbers.shape == (len(found["out"]), len(names))
    assert not np.isnan(numbers).any()
    at = {}
    for i, name in enumerate(names):
        at[name] = i
    farhan = DuckDBDeliveryRepository(db).person("Sahibzada Farhan")
    last = rows_of(found, "batter", farhan)[-1]
    row = numbers[last]
    # Pakistan batted first: no chase; 120 balls less the 88 bowled leaves 32
    assert row[at["chasing"]] == 0
    assert row[at["runs_needed"]] == 0
    assert row[at["balls_left"]] == 32
    assert row[at["limited_overs"]] == 1
    assert row[at["kind=men_t20"]] == 1
    assert row[at["batting_order=top"]] == 1
    assert abs(row[at["batter_rate"]] - 0.05) < 1e-6  # no record before: the prior
    # Sri Lanka's chase of 216: the first ball needs all of it, with 120 balls left
    chase = []
    for i in range(len(found["out"])):
        if found["match_id"][i] == "1550709" and found["innings"][i] == 2:
            chase.append(i)
    later = chase[-1]
    needed = 216 - found["runs_so_far"][later]
    assert numbers[later][at["runs_needed"]] == needed  # what is left of the target
    first = numbers[chase[0]]
    assert (
        first[at["chasing"]],
        first[at["runs_needed"]],
        first[at["balls_left"]],
    ) == (
        1,
        216,
        120,
    )
