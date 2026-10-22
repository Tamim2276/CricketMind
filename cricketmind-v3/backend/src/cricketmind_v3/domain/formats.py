"""What a format's name means in the data (Day 43, D-047).

Cricsheet files a T20 international and an IPL match alike as "T20", and keeps "IT20"
for 320 other international T20s (regional qualifiers). A question's "T20I" is the
first kind only; its "T20" is every twenty-over match.
"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Format:
    match_types: tuple[str, ...]  # Cricsheet's match types
    team_type: str | None  # "international", or None for any


FORMATS = {
    "Test": Format(match_types=("Test",), team_type=None),
    "ODI": Format(match_types=("ODI",), team_type=None),
    "T20I": Format(match_types=("T20",), team_type="international"),
    "T20": Format(match_types=("T20", "IT20"), team_type=None),
    # Cricsheet's own codes, for the rest
    "IT20": Format(match_types=("IT20",), team_type=None),
    "ODM": Format(match_types=("ODM",), team_type=None),  # one-day, not ODIs
    "MDM": Format(match_types=("MDM",), team_type=None),  # multi-day, not Tests
}


def format_named(name: str) -> Format | None:
    """The format a name stands for; None for a name the data does not know."""
    return FORMATS.get(name)
