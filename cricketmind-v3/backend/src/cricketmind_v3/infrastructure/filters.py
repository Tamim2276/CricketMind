"""The filters every engine shares: which matches a number may come from (Day 43).

Each query names the matches table "m" and the match_venues table "v".
"""

import datetime

from cricketmind_v3.application.ports import FilterError
from cricketmind_v3.domain.formats import format_named

MATCH_FILTERS = ("match_format", "gender", "season", "since", "before", "venue")


def match_condition(field: str, value) -> tuple[str, list]:
    """One match filter as a piece of WHERE, with its values (every value a ?)."""
    if field == "match_format":
        found = format_named(value)
        if found is None:
            raise FilterError(f"no format called {value}")
        marks = ", ".join(["?"] * len(found.match_types))
        sql = f"m.match_type IN ({marks})"
        values = list(found.match_types)
        if found.team_type is not None:
            sql += " AND m.team_type = ?"
            values.append(found.team_type)
        return "(" + sql + ")", values
    if field == "gender":
        return "m.gender = ?", [value]
    if field == "season":
        return "m.season = ?", [value]
    if field == "since":
        return "m.date >= ?", [first_day(value)]
    if field == "before":
        return "m.date < ?", [first_day(value)]  # the day itself is not before it
    if field == "venue":
        return "v.venue_key = ?", [value]
    raise FilterError(f"cannot pick matches by {field}")


def first_day(since: str) -> datetime.date:
    """ "2020" is 1 January 2020; "2020-06-01" is that day."""
    try:
        if len(since) == 4:
            return datetime.date(int(since), 1, 1)
        return datetime.date.fromisoformat(since)
    except ValueError:
        raise FilterError(f"'{since}' is not a year or a date") from None
