"""One ball, one row: the deliveries table, and the phase rule."""

import dataclasses
from dataclasses import dataclass


class SchemaError(ValueError):
    """A match or a ball the table cannot hold."""


# formats that share overs and phases (Cricsheet's match_type)
T20_TYPES = ("T20", "IT20")
ODI_TYPES = ("ODI", "ODM")  # ODM: one-day matches that are not ODIs
LONG_TYPES = ("Test", "MDM")  # MDM: first-class; no fixed overs, so no phases


def phase(match_type: str, over: int, balls_per_over: int = 6) -> str | None:
    """The phase of an over, counted from 0 as Cricsheet does; None if there is none.

    T20: overs 0-5 powerplay, 6-14 middle, 15 on death. ODI: 0-9, 10-39, 40 on.
    """
    if over < 0:
        raise SchemaError(f"over {over}: overs are counted from 0")
    if match_type in LONG_TYPES:
        return None
    if match_type in T20_TYPES:
        last_powerplay, first_death = 5, 15
    elif match_type in ODI_TYPES:
        last_powerplay, first_death = 9, 40
    else:
        raise SchemaError(f"unknown match type '{match_type}'")
    if balls_per_over != 6:
        return None  # The Hundred: five-ball overs, its own phases
    if over <= last_powerplay:
        return "powerplay"
    if over < first_death:
        return "middle"
    return "death"


@dataclass(frozen=True, kw_only=True)
class Match:
    """What every ball of a match shares."""

    match_id: str  # "1550709", the file name
    match_type: str  # "T20"
    gender: str  # "male" or "female"
    team_type: str  # "international" or "club"
    competition: str | None  # "Indian Premier League"; None if not named
    date: str  # "2026-10-11", the first day
    venue: str  # "Rawalpindi Cricket Stadium"
    balls_per_over: int  # 6; 5 in The Hundred


@dataclass(frozen=True, kw_only=True)
class Place:
    """Where a ball sits in its match."""

    innings: int  # 1, 2 (up to 4 in a Test)
    batting_team: str  # "Pakistan"
    super_over: bool  # a tie-breaker, kept apart from the match
    over: int  # from 0, as Cricsheet counts
    ball_index: int  # its position in the over's list, from 1


@dataclass(frozen=True, kw_only=True)
class Delivery:
    """One row of the deliveries table: one ball, as it was bowled."""

    # the match, on every row, so a filter never needs a second table
    match_id: str
    match_type: str
    gender: str
    team_type: str
    competition: str | None
    date: str
    venue: str
    # where
    innings: int
    batting_team: str
    super_over: bool
    over: int
    ball_index: int  # a re-bowled wide gets the next position, not the same number
    phase: str | None  # powerplay, middle, death; None in long formats and The Hundred
    # who
    batter: str
    bowler: str
    non_striker: str
    # what happened
    runs_batter: int  # off the bat
    runs_extras: int
    runs_total: int
    wides: int
    noballs: int
    byes: int
    legbyes: int
    penalty: int
    wickets: int  # how many fell on this ball: almost always 0 or 1
    player_out: str | None  # the first wicket's; may be the non-striker
    wicket_kind: str | None  # "caught", "bowled", "lbw", "run out", ...
    review_by: str | None  # the team that asked for the review
    review_batter: str | None
    review_decision: str | None  # "upheld" or "struck down"
    review_umpires_call: bool


def make_delivery(match: Match, place: Place, ball: dict) -> Delivery:
    """One Cricsheet ball, with where it sits, as one row."""
    extras = ball.get("extras", {})
    wickets = ball.get("wickets", [])
    review = ball.get("review", {})

    player_out = None
    wicket_kind = None
    if wickets:
        player_out = wickets[0]["player_out"]
        wicket_kind = wickets[0]["kind"]

    if place.super_over:
        ball_phase = None  # a super over is not part of the match's phases
    else:
        ball_phase = phase(match.match_type, place.over, match.balls_per_over)

    return Delivery(
        match_id=match.match_id,
        match_type=match.match_type,
        gender=match.gender,
        team_type=match.team_type,
        competition=match.competition,
        date=match.date,
        venue=match.venue,
        innings=place.innings,
        batting_team=place.batting_team,
        super_over=place.super_over,
        over=place.over,
        ball_index=place.ball_index,
        phase=ball_phase,
        batter=ball["batter"],
        bowler=ball["bowler"],
        non_striker=ball["non_striker"],
        runs_batter=ball["runs"]["batter"],
        runs_extras=ball["runs"]["extras"],
        runs_total=ball["runs"]["total"],
        wides=extras.get("wides", 0),
        noballs=extras.get("noballs", 0),
        byes=extras.get("byes", 0),
        legbyes=extras.get("legbyes", 0),
        penalty=extras.get("penalty", 0),
        wickets=len(wickets),
        player_out=player_out,
        wicket_kind=wicket_kind,
        review_by=review.get("by"),
        review_batter=review.get("batter"),
        review_decision=review.get("decision"),
        review_umpires_call=review.get("umpires_call", False),
    )


# the same table in SQL, column for column (a test keeps the two in step)
CREATE_DELIVERIES = """
CREATE TABLE IF NOT EXISTS deliveries (
    match_id VARCHAR NOT NULL,
    match_type VARCHAR NOT NULL,
    gender VARCHAR NOT NULL,
    team_type VARCHAR NOT NULL,
    competition VARCHAR,
    date DATE NOT NULL,
    venue VARCHAR NOT NULL,
    innings INTEGER NOT NULL,
    batting_team VARCHAR NOT NULL,
    super_over BOOLEAN NOT NULL,
    over INTEGER NOT NULL,
    ball_index INTEGER NOT NULL,
    phase VARCHAR,
    batter VARCHAR NOT NULL,
    bowler VARCHAR NOT NULL,
    non_striker VARCHAR NOT NULL,
    runs_batter INTEGER NOT NULL,
    runs_extras INTEGER NOT NULL,
    runs_total INTEGER NOT NULL,
    wides INTEGER NOT NULL,
    noballs INTEGER NOT NULL,
    byes INTEGER NOT NULL,
    legbyes INTEGER NOT NULL,
    penalty INTEGER NOT NULL,
    wickets INTEGER NOT NULL,
    player_out VARCHAR,
    wicket_kind VARCHAR,
    review_by VARCHAR,
    review_batter VARCHAR,
    review_decision VARCHAR,
    review_umpires_call BOOLEAN NOT NULL,
    PRIMARY KEY (match_id, innings, over, ball_index)
)
"""


# Three small tables beside the balls (Day 30): the result, each innings, the people


@dataclass(frozen=True, kw_only=True)
class MatchRow:
    """One row of the matches table: the facts and the result of a match."""

    match_id: str
    data_version: str  # "1.2.0" or "1.3.0"
    match_type: str
    gender: str
    team_type: str
    competition: str | None
    season: str  # "2026" or "2026/27"
    date: str  # the first day
    days: int  # how many dates it was played on
    venue: str
    city: str | None
    team1: str
    team2: str
    toss_winner: str | None
    toss_decision: str | None  # "bat" or "field"
    winner: str | None
    won_by_runs: int | None
    won_by_wickets: int | None
    won_by_innings: bool  # "won by an innings and 20 runs"
    result: str | None  # "tie", "no result" or "draw"
    method: str | None  # "D/L" for a rain rule
    eliminator: str | None  # who won a super over or a bowl-out
    overs: int | None  # per innings; None in long formats
    balls_per_over: int


@dataclass(frozen=True, kw_only=True)
class InningsRow:
    """One row of the innings table."""

    match_id: str
    innings: int
    team: str
    super_over: bool
    declared: bool
    forfeited: bool  # no ball bowled
    target_runs: int | None  # the runs needed to win, set before a chase
    target_overs: float | None
    powerplay_to: (
        float | None
    )  # where the mandatory powerplay ended: 5.6, or 5.7 with an extra
    penalty_pre: int  # penalty runs awarded outside the balls
    penalty_post: int
    miscounted_overs: int  # overs Cricsheet marks as having the wrong number of balls


@dataclass(frozen=True, kw_only=True)
class PersonRow:
    """One row of the people table: a name as used in one match, and its id."""

    match_id: str
    name: str  # "V Kohli"
    person_id: str  # "ba607b88", the Cricsheet register id
    team: str | None  # None for an umpire or a referee


CREATE_MATCHES = """
CREATE TABLE IF NOT EXISTS matches (
    match_id VARCHAR PRIMARY KEY,
    data_version VARCHAR NOT NULL,
    match_type VARCHAR NOT NULL,
    gender VARCHAR NOT NULL,
    team_type VARCHAR NOT NULL,
    competition VARCHAR,
    season VARCHAR NOT NULL,
    date DATE NOT NULL,
    days INTEGER NOT NULL,
    venue VARCHAR NOT NULL,
    city VARCHAR,
    team1 VARCHAR NOT NULL,
    team2 VARCHAR NOT NULL,
    toss_winner VARCHAR,
    toss_decision VARCHAR,
    winner VARCHAR,
    won_by_runs INTEGER,
    won_by_wickets INTEGER,
    won_by_innings BOOLEAN NOT NULL,
    result VARCHAR,
    method VARCHAR,
    eliminator VARCHAR,
    overs INTEGER,
    balls_per_over INTEGER NOT NULL
)
"""

CREATE_INNINGS = """
CREATE TABLE IF NOT EXISTS innings (
    match_id VARCHAR NOT NULL,
    innings INTEGER NOT NULL,
    team VARCHAR NOT NULL,
    super_over BOOLEAN NOT NULL,
    declared BOOLEAN NOT NULL,
    forfeited BOOLEAN NOT NULL,
    target_runs INTEGER,
    target_overs DOUBLE,
    powerplay_to DOUBLE,
    penalty_pre INTEGER NOT NULL,
    penalty_post INTEGER NOT NULL,
    miscounted_overs INTEGER NOT NULL,
    PRIMARY KEY (match_id, innings)
)
"""

CREATE_PEOPLE = """
CREATE TABLE IF NOT EXISTS people (
    match_id VARCHAR NOT NULL,
    name VARCHAR NOT NULL,
    person_id VARCHAR NOT NULL,
    team VARCHAR,
    PRIMARY KEY (match_id, name)
)
"""

# every table the pipeline fills from the match files: name, SQL, row class
TABLES = (
    ("deliveries", CREATE_DELIVERIES, Delivery),
    ("matches", CREATE_MATCHES, MatchRow),
    ("innings", CREATE_INNINGS, InningsRow),
    ("people", CREATE_PEOPLE, PersonRow),
)


def insert_sql(table: str, row_class: type) -> str:
    """INSERT INTO table VALUES (?, ?, ...), one ? per column."""
    marks = []
    for _ in dataclasses.fields(row_class):
        marks.append("?")
    return f"INSERT INTO {table} VALUES (" + ", ".join(marks) + ")"


def row_values(row) -> list:
    """Any row's values, in column order."""
    values = []
    for field in dataclasses.fields(row):
        values.append(getattr(row, field.name))
    return values
