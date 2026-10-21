"""Day 30: whole matches into rows, on three real Cricsheet fixtures."""

from pathlib import Path

import pytest

from cricketmind_v3.pipeline.parse_match import ParseError, parse_match, read_match

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "cricsheet"

T20I = read_match(FIXTURES / "1550709.json")  # Pakistan v Sri Lanka, format 1.3.0
IPL = read_match(FIXTURES / "1535463.json")  # Rajasthan v Sunrisers, format 1.2.0
WOMEN = read_match(FIXTURES / "1490709.json")  # England v Australia, format 1.2.0


def innings_balls(parsed, innings: int) -> list:
    found = []
    for row in parsed.deliveries:
        if row.innings == innings:
            found.append(row)
    return found


def total(rows, column: str) -> int:
    added = 0
    for row in rows:
        added += getattr(row, column)
    return added


# One row per ball as bowled, wides and no-balls included


def test_one_row_per_ball():
    assert len(innings_balls(T20I, 1)) == 125
    assert len(innings_balls(T20I, 2)) == 122


def test_wides_and_no_balls_are_rows_but_not_legal_balls():
    first = innings_balls(T20I, 1)
    assert total(first, "wides") == 4
    assert total(first, "noballs") == 1
    legal = 0
    for row in first:
        if row.wides == 0 and row.noballs == 0:
            legal += 1
    assert legal == 120  # twenty overs of six


def test_a_wide_bowled_again_gets_the_next_position():
    over_nine = []
    for row in innings_balls(T20I, 1):
        if row.over == 9:
            over_nine.append(row)
    positions = []
    for row in over_nine:
        positions.append(row.ball_index)
    assert positions == [1, 2, 3, 4, 5, 6, 7]  # the over had seven balls
    assert over_nine[3].wides == 1  # the wide, numbered 9.4 in the file
    assert over_nine[4].runs_batter == 4  # the ball bowled again, also 9.4


def test_wickets():
    assert total(innings_balls(T20I, 1), "wickets") == 3
    assert total(innings_balls(T20I, 2), "wickets") == 7


def test_phases_change_after_over_five():
    first = innings_balls(T20I, 1)
    last_of_over_five = None
    first_of_over_six = None
    for row in first:
        if row.over == 5:
            last_of_over_five = row
        if row.over == 6 and first_of_over_six is None:
            first_of_over_six = row
    assert last_of_over_five.phase == "powerplay"
    assert first_of_over_six.phase == "middle"


# The rows agree with the result printed in the file


def test_the_runs_add_up_to_won_by_82_runs():
    first = total(innings_balls(T20I, 1), "runs_total")
    second = total(innings_balls(T20I, 2), "runs_total")
    assert (first, second) == (215, 133)
    assert first - second == T20I.match.won_by_runs == 82


def test_the_target_is_the_first_innings_plus_one():
    first = total(innings_balls(T20I, 1), "runs_total")
    assert T20I.innings[1].target_runs == first + 1 == 216


def test_the_match_row():
    match = T20I.match
    assert (match.winner, match.won_by_runs, match.won_by_wickets) == (
        "Pakistan",
        82,
        None,
    )
    assert (match.toss_winner, match.toss_decision) == ("Sri Lanka", "field")
    assert (match.season, match.date, match.days) == ("2026/27", "2026-10-11", 1)
    assert (match.city, match.data_version, match.method) == (
        "Rawalpindi",
        "1.3.0",
        None,
    )


def test_the_innings_rows():
    first, second = T20I.innings
    assert (first.team, first.target_runs, first.powerplay_to) == (
        "Pakistan",
        None,
        5.6,
    )
    assert (second.team, second.target_runs) == ("Sri Lanka", 216)
    assert not first.super_over and not first.forfeited


def test_the_people_and_their_ids():
    assert len(T20I.people) == 27  # 22 players and 5 officials
    with_team = 0
    for person in T20I.people:
        if person.team is not None:
            with_team += 1
        if person.name == "Abrar Ahmed":
            assert person.person_id == "abb7c76c"
    assert with_team == 22


# The older format, 1.2.0, and a women's match


def test_the_ipl_match_in_the_older_format():
    first = total(innings_balls(IPL, 1), "runs_total")
    second = total(innings_balls(IPL, 2), "runs_total")
    assert len(IPL.deliveries) == 252
    assert first - second == IPL.match.won_by_runs == 47
    with_team = 0
    for person in IPL.people:
        if person.team is not None:
            with_team += 1
    assert with_team == 24  # twelve a side, with the impact player


def test_the_womens_match_won_by_wickets():
    assert WOMEN.match.gender == "female"
    assert (WOMEN.match.winner, WOMEN.match.won_by_wickets) == ("Australia", 7)
    assert total(innings_balls(WOMEN, 2), "wickets") == 3
    assert (
        WOMEN.innings[0].powerplay_to == 5.7
    )  # an extra gave over five a seventh ball


# Cases the fixtures do not have


def tiny_match(innings: list) -> dict:
    return {
        "meta": {"data_version": "1.3.0"},
        "info": {
            "match_type": "T20",
            "gender": "male",
            "team_type": "club",
            "season": 2026,  # a number this time
            "dates": ["2026-01-01"],
            "venue": "Somewhere",
            "teams": ["A", "B"],
            "outcome": {"result": "tie", "eliminator": "A"},
        },
        "innings": innings,
    }


ONE_BALL = {
    "batter": "x",
    "bowler": "y",
    "non_striker": "z",
    "runs": {"batter": 1, "extras": 0, "total": 1},
}


def test_a_super_over_is_kept_apart_with_no_phase():
    parsed = parse_match(
        "t1",
        tiny_match(
            [
                {"team": "A", "overs": [{"over": 0, "deliveries": [ONE_BALL]}]},
                {
                    "team": "A",
                    "super_over": True,
                    "overs": [{"over": 0, "deliveries": [ONE_BALL]}],
                },
            ]
        ),
    )
    normal, extra = parsed.deliveries
    assert (normal.phase, normal.super_over) == ("powerplay", False)
    assert (extra.phase, extra.super_over, extra.innings) == (None, True, 2)
    assert (parsed.match.result, parsed.match.eliminator) == ("tie", "A")
    assert parsed.match.season == "2026"


def test_a_forfeited_innings_has_a_row_but_no_balls():
    parsed = parse_match("t2", tiny_match([{"team": "B", "forfeited": True}]))
    assert parsed.deliveries == ()
    assert parsed.innings[0].forfeited is True


def test_a_broken_match_names_itself():
    broken = tiny_match([{"team": "A", "overs": [{"over": 0, "deliveries": [{}]}]}])
    with pytest.raises(ParseError) as error:
        parse_match("9999999", broken)
    assert str(error.value).startswith("9999999: KeyError")


def test_an_odi_keeps_its_opening_powerplay():
    # found on Day 32: an ODI lists three "mandatory" powerplays; the first one counts
    odi = tiny_match(
        [
            {
                "team": "A",
                "powerplays": [
                    {"from": 0.1, "to": 9.6, "type": "mandatory"},
                    {"from": 10.1, "to": 39.6, "type": "mandatory"},
                    {"from": 40.1, "to": 49.6, "type": "mandatory"},
                ],
                "overs": [{"over": 0, "deliveries": [ONE_BALL]}],
            }
        ]
    )
    assert parse_match("t3", odi).innings[0].powerplay_to == 9.6
