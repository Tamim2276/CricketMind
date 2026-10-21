"""The deliveries table in a real DuckDB database (in memory), and the four SQL words."""

import dataclasses
import re

import duckdb
import pytest

from cricketmind_v3.pipeline.schema import (
    CREATE_DELIVERIES,
    Delivery,
    insert_sql,
    make_delivery,
    row_values,
)
from tests.unit.test_schema import T20I, WIDE, at


def ball(batter: str, runs: int) -> dict:
    return {
        "batter": batter,
        "bowler": "E Malinga",
        "non_striker": "Sahibzada Farhan",
        "runs": {"batter": runs, "extras": 0, "total": runs},
    }


@pytest.fixture
def db():
    connection = duckdb.connect()  # in memory: gone when the test ends
    connection.execute(CREATE_DELIVERIES)  # CREATE
    yield connection
    connection.close()


def test_the_sql_columns_match_the_row_exactly():
    # one list of columns in Python, one in SQL: they must never drift apart
    sql_columns = []
    for line in CREATE_DELIVERIES.splitlines():
        found = re.match(r"\s+(\w+) (VARCHAR|INTEGER|BOOLEAN|DATE)", line)
        if found:
            sql_columns.append(found.group(1))
    row_columns = []
    for field in dataclasses.fields(Delivery):
        row_columns.append(field.name)
    assert sql_columns == row_columns


def test_a_row_goes_in_and_comes_back_the_same(db):
    row = make_delivery(T20I, at(over=9, ball_index=4), WIDE)
    db.execute(insert_sql(), row_values(row))  # INSERT
    back = db.execute("SELECT * FROM deliveries").fetchall()  # SELECT
    assert len(back) == 1
    values = list(back[0])
    values[5] = str(values[5])  # DuckDB gives the date back as a date
    assert values == row_values(row)


def test_select_with_a_condition(db):
    for index, (batter, runs) in enumerate([("Maaz Sadaqat", 6), ("Babar Azam", 1)]):
        row = make_delivery(T20I, at(over=10, ball_index=index + 1), ball(batter, runs))
        db.execute(insert_sql(), row_values(row))
    found = db.execute(
        "SELECT batter, runs_batter FROM deliveries WHERE runs_batter = 6"
    ).fetchall()
    assert found == [("Maaz Sadaqat", 6)]


def test_group_by_adds_up_each_batters_runs(db):
    balls = [("Maaz Sadaqat", 6), ("Maaz Sadaqat", 4), ("Babar Azam", 1)]
    for index, (batter, runs) in enumerate(balls):
        row = make_delivery(T20I, at(over=10, ball_index=index + 1), ball(batter, runs))
        db.execute(insert_sql(), row_values(row))
    totals = db.execute(
        "SELECT batter, SUM(runs_batter), COUNT(*) FROM deliveries "
        "GROUP BY batter ORDER BY batter"
    ).fetchall()
    assert totals == [("Babar Azam", 1, 1), ("Maaz Sadaqat", 10, 2)]


def test_the_same_ball_cannot_be_stored_twice(db):
    row = make_delivery(T20I, at(over=9, ball_index=4), WIDE)
    db.execute(insert_sql(), row_values(row))
    with pytest.raises(duckdb.ConstraintException):
        db.execute(insert_sql(), row_values(row))  # the primary key refuses it
