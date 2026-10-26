"""Day 50: the four stages of an innings, and the parts of the batting order."""

from cricketmind_v3.domain.stages import (
    Record,
    by_batter,
    in_order,
    in_stage,
    order_of,
    stage_of,
    usual_order,
)


def test_where_each_stage_starts():
    found = []
    for ball in (1, 5, 6, 15, 16, 30, 31, 200):
        found.append(stage_of(ball))
    assert found == [1, 1, 2, 2, 3, 3, 4, 4]


def test_a_batters_stages_add_up_to_his_innings():
    records = [
        Record(batter="a", stage=1, order="top", outs=0, faced=5),
        Record(batter="a", stage=2, order="top", outs=1, faced=7),
        Record(batter="b", stage=1, order="tail", outs=1, faced=3),
    ]
    assert by_batter(records) == [(1, 12), (1, 3)]
    assert in_stage(records, 1) == [(0, 5), (1, 3)]
    assert in_stage(records, 2) == [(1, 7)]
    assert in_stage(records, 4) == []


def test_the_parts_of_the_order():
    found = []
    for place in (1, 3, 4, 7, 8, 11, 12):
        found.append(order_of(place))
    assert found == ["top", "top", "middle", "middle", "tail", "tail", "tail"]


def test_where_he_usually_bats_is_where_most_of_his_balls_were():
    kohli = [
        Record(batter="k", stage=1, order="top", outs=0, faced=40),
        Record(batter="k", stage=2, order="top", outs=1, faced=60),
        Record(batter="k", stage=1, order="middle", outs=1, faced=30),
    ]
    assert usual_order(kohli) == "top"  # 100 balls at the top, 30 in the middle
    level = [
        Record(batter="x", stage=1, order="middle", outs=0, faced=10),
        Record(batter="x", stage=1, order="top", outs=0, faced=10),
    ]
    assert usual_order(level) == "top"  # level: the top first
    assert usual_order([]) is None


def test_only_the_balls_batting_in_one_part_of_the_order():
    records = [
        Record(batter="a", stage=1, order="top", outs=0, faced=5),
        Record(batter="a", stage=1, order="tail", outs=1, faced=2),
        Record(batter="b", stage=1, order="tail", outs=1, faced=3),
    ]
    tail = in_order(records, "tail")
    assert by_batter(tail) == [(1, 2), (1, 3)]
