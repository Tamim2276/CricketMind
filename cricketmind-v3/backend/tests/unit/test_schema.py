import dataclasses

import pytest

from cricketmind_v3.pipeline.schema import (
    Delivery,
    Match,
    Place,
    SchemaError,
    make_delivery,
    phase,
)

# Pakistan v Sri Lanka, Rawalpindi, 2026-10-11 (Cricsheet 1550709), as read on Day 28
T20I = Match(
    match_id="1550709",
    match_type="T20",
    gender="male",
    team_type="international",
    competition="Sri Lanka tour of Pakistan",
    date="2026-10-11",
    venue="Rawalpindi Cricket Stadium",
    balls_per_over=6,
)

# the wide at 9.4, typed in from the file
WIDE = {
    "actual_delivery": "9.4",
    "batter": "Maaz Sadaqat",
    "bowler": "E Malinga",
    "extras": {"wides": 1},
    "non_striker": "Sahibzada Farhan",
    "runs": {"batter": 0, "extras": 1, "total": 1},
}


def at(over: int, ball_index: int, innings: int = 1, super_over: bool = False) -> Place:
    return Place(
        innings=innings,
        batting_team="Pakistan",
        super_over=super_over,
        over=over,
        ball_index=ball_index,
    )


# The plan's test: a hand-made delivery becomes the right row


def test_the_wide_at_9_4_becomes_the_right_row():
    assert make_delivery(T20I, at(over=9, ball_index=4), WIDE) == Delivery(
        match_id="1550709",
        match_type="T20",
        gender="male",
        team_type="international",
        competition="Sri Lanka tour of Pakistan",
        date="2026-10-11",
        venue="Rawalpindi Cricket Stadium",
        innings=1,
        batting_team="Pakistan",
        super_over=False,
        over=9,
        ball_index=4,
        phase="middle",
        batter="Maaz Sadaqat",
        bowler="E Malinga",
        non_striker="Sahibzada Farhan",
        runs_batter=0,
        runs_extras=1,
        runs_total=1,
        wides=1,
        noballs=0,
        byes=0,
        legbyes=0,
        penalty=0,
        wickets=0,
        player_out=None,
        wicket_kind=None,
        review_by=None,
        review_batter=None,
        review_decision=None,
        review_umpires_call=False,
    )


def test_the_wide_bowled_again_is_a_different_row():
    again = {
        "actual_delivery": "9.4",  # the same number as the wide
        "batter": "Maaz Sadaqat",
        "bowler": "E Malinga",
        "non_striker": "Sahibzada Farhan",
        "runs": {"batter": 4, "extras": 0, "total": 4},
    }
    first = make_delivery(T20I, at(over=9, ball_index=4), WIDE)
    second = make_delivery(T20I, at(over=9, ball_index=5), again)
    assert first != second
    assert (second.ball_index, second.wides, second.runs_batter) == (5, 0, 4)


def test_a_no_ball_the_batter_scored_off():
    noball = {
        "actual_delivery": "12.1",
        "batter": "Sahibzada Farhan",
        "bowler": "PHKD Mendis",
        "extras": {"noballs": 1},
        "non_striker": "Maaz Sadaqat",
        "runs": {"batter": 1, "extras": 1, "total": 2},
    }
    row = make_delivery(T20I, at(over=12, ball_index=1), noball)
    assert (row.noballs, row.runs_batter, row.runs_total) == (1, 1, 2)


def test_a_wicket():
    caught = {
        "actual_delivery": "6.5",
        "batter": "Fakhar Zaman",
        "bowler": "M Theekshana",
        "non_striker": "Sahibzada Farhan",
        "runs": {"batter": 0, "extras": 0, "total": 0},
        "wickets": [
            {
                "player_out": "Fakhar Zaman",
                "fielders": [{"name": "PHKD Mendis"}],
                "kind": "caught",
            }
        ],
    }
    row = make_delivery(T20I, at(over=6, ball_index=5), caught)
    assert (row.wickets, row.player_out, row.wicket_kind) == (
        1,
        "Fakhar Zaman",
        "caught",
    )


def test_a_review():
    reviewed = {
        "actual_delivery": "13.6",
        "batter": "Maaz Sadaqat",
        "bowler": "N Thushara",
        "non_striker": "Sahibzada Farhan",
        "review": {
            "by": "Sri Lanka",
            "umpire": "Faisal Afridi",
            "batter": "Sahibzada Farhan",
            "decision": "struck down",
            "type": "wicket",
        },
        "runs": {"batter": 0, "extras": 0, "total": 0},
    }
    row = make_delivery(T20I, at(over=13, ball_index=6), reviewed)
    assert row.review_by == "Sri Lanka"
    assert row.review_batter == "Sahibzada Farhan"  # not the striker: it can differ
    assert row.review_decision == "struck down"
    assert row.review_umpires_call is False


def test_a_super_over_has_no_phase():
    row = make_delivery(
        T20I, at(over=0, ball_index=1, innings=3, super_over=True), WIDE
    )
    assert row.phase is None
    assert row.super_over is True


# The phase rule (overs counted from 0)


def test_t20_phases():
    assert phase("T20", 0) == "powerplay"
    assert phase("T20", 5) == "powerplay"
    assert phase("T20", 6) == "middle"
    assert phase("T20", 14) == "middle"
    assert phase("T20", 15) == "death"
    assert phase("T20", 19) == "death"


def test_odi_phases():
    assert phase("ODI", 9) == "powerplay"
    assert phase("ODI", 10) == "middle"
    assert phase("ODI", 39) == "middle"
    assert phase("ODI", 40) == "death"


def test_related_formats_share_the_rules():
    assert phase("IT20", 5) == phase("T20", 5)
    assert phase("ODM", 40) == phase("ODI", 40)


def test_long_formats_have_no_phases():
    assert phase("Test", 3) is None
    assert phase("MDM", 3) is None


def test_the_hundred_has_no_t20_phases():
    # 389 matches in the snapshot are labelled T20 but have five-ball overs
    assert phase("T20", 3, balls_per_over=5) is None


def test_unknown_formats_and_negative_overs_are_refused():
    with pytest.raises(SchemaError):
        phase("Six-a-side", 1)
    with pytest.raises(SchemaError):
        phase("T20", -1)


def test_a_row_cannot_change():
    row = make_delivery(T20I, at(over=9, ball_index=4), WIDE)
    with pytest.raises(dataclasses.FrozenInstanceError):
        row.runs_total = 6
