"""One Cricsheet match -> its rows, with plain loops (Day 30)."""

import json
from dataclasses import dataclass
from pathlib import Path

from cricketmind_v3.pipeline.schema import (
    Delivery,
    InningsRow,
    Match,
    MatchRow,
    PersonRow,
    Place,
    make_delivery,
)


class ParseError(ValueError):
    """A match file the reader cannot turn into rows."""


@dataclass(frozen=True, kw_only=True)
class ParsedMatch:
    """Every row one match file gives, for every table."""

    match: MatchRow
    innings: tuple[InningsRow, ...]
    people: tuple[PersonRow, ...]
    deliveries: tuple[Delivery, ...]


def read_match(path: Path) -> ParsedMatch:
    """Read one match file; its name is its id ("1550709.json")."""
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    return parse_match(path.stem, data)


def parse_match(match_id: str, data: dict) -> ParsedMatch:
    """Turn one match, already read from JSON, into rows."""
    try:
        return _parse(match_id, data)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ParseError(f"{match_id}: {type(error).__name__}: {error}") from error


def _parse(match_id: str, data: dict) -> ParsedMatch:
    info = data["info"]
    shared = match_facts(match_id, info)

    innings_rows = []
    deliveries = []
    for number, innings in enumerate(data["innings"], start=1):
        innings_rows.append(innings_row(match_id, number, innings))
        for over in innings.get("overs", []):  # a forfeited innings has none
            # counting positions: a wide bowled again gets the next one
            for position, ball in enumerate(over["deliveries"], start=1):
                place = Place(
                    innings=number,
                    batting_team=innings["team"],
                    super_over=innings.get("super_over", False),
                    over=over["over"],
                    ball_index=position,
                )
                deliveries.append(make_delivery(shared, place, ball))

    return ParsedMatch(
        match=match_row(match_id, data),
        innings=tuple(innings_rows),
        people=people_rows(match_id, info),
        deliveries=tuple(deliveries),
    )


def match_facts(match_id: str, info: dict) -> Match:
    """What every ball of the match shares."""
    return Match(
        match_id=match_id,
        match_type=info["match_type"],
        gender=info["gender"],
        team_type=info["team_type"],
        competition=info.get("event", {}).get("name"),
        date=info["dates"][0],
        venue=info["venue"],
        balls_per_over=info.get("balls_per_over", 6),
    )


def match_row(match_id: str, data: dict) -> MatchRow:
    info = data["info"]
    toss = info.get("toss", {})
    outcome = info.get("outcome", {})
    won_by = outcome.get("by", {})
    return MatchRow(
        match_id=match_id,
        data_version=data["meta"]["data_version"],
        match_type=info["match_type"],
        gender=info["gender"],
        team_type=info["team_type"],
        competition=info.get("event", {}).get("name"),
        season=str(info["season"]),  # sometimes a number, sometimes "2026/27"
        date=info["dates"][0],
        days=len(info["dates"]),
        venue=info["venue"],
        city=info.get("city"),
        team1=info["teams"][0],
        team2=info["teams"][1],
        toss_winner=toss.get("winner"),
        toss_decision=toss.get("decision"),
        winner=outcome.get("winner"),
        won_by_runs=won_by.get("runs"),
        won_by_wickets=won_by.get("wickets"),
        won_by_innings="innings" in won_by,
        result=outcome.get("result"),
        method=outcome.get("method"),
        eliminator=outcome.get("eliminator") or outcome.get("bowl_out"),
        overs=info.get("overs"),
        balls_per_over=info.get("balls_per_over", 6),
    )


def innings_row(match_id: str, number: int, innings: dict) -> InningsRow:
    target = innings.get("target", {})
    penalty = innings.get("penalty_runs", {})
    # the opening powerplay: an ODI lists three "mandatory" ones (overs 1-10, 11-40,
    # 41-50), so take the one that starts in the first over (found on Day 32)
    powerplay_to = None
    for powerplay in innings.get("powerplays", []):
        if powerplay.get("type") == "mandatory" and powerplay["from"] < 1:
            powerplay_to = powerplay["to"]
    return InningsRow(
        match_id=match_id,
        innings=number,
        team=innings["team"],
        super_over=innings.get("super_over", False),
        declared=innings.get("declared", False),
        forfeited=innings.get("forfeited", False),
        target_runs=target.get("runs"),
        target_overs=target.get("overs"),
        powerplay_to=powerplay_to,
        penalty_pre=penalty.get("pre", 0),
        penalty_post=penalty.get("post", 0),
        miscounted_overs=len(innings.get("miscounted_overs", {})),
    )


def people_rows(match_id: str, info: dict) -> tuple[PersonRow, ...]:
    """Everyone in the match's register, with their team if they played."""
    teams = {}
    for team, names in info.get("players", {}).items():
        for name in names:
            teams[name] = team
    rows = []
    for name, person_id in info.get("registry", {}).get("people", {}).items():
        rows.append(
            PersonRow(
                match_id=match_id,
                name=name,
                person_id=person_id,
                team=teams.get(name),
            )
        )
    return tuple(rows)
