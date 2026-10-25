"""Day 50: the four stages of an innings, by the balls faced so far."""

from cricketmind_v3.domain.stages import Record, by_batter, in_stage, stage_of


def test_where_each_stage_starts():
    found = []
    for ball in (1, 5, 6, 15, 16, 30, 31, 200):
        found.append(stage_of(ball))
    assert found == [1, 1, 2, 2, 3, 3, 4, 4]


def test_a_batters_stages_add_up_to_his_innings():
    records = [
        Record(batter="a", stage=1, outs=0, faced=5),
        Record(batter="a", stage=2, outs=1, faced=7),
        Record(batter="b", stage=1, outs=1, faced=3),
    ]
    assert by_batter(records) == [(1, 12), (1, 3)]
    assert in_stage(records, 1) == [(0, 5), (1, 3)]
    assert in_stage(records, 2) == [(1, 7)]
    assert in_stage(records, 4) == []
