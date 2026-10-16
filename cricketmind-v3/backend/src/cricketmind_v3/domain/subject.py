from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Subject:
    """Who and what a number is about. An empty field means "any"."""

    player: str | None = None  # "V Kohli"
    shot: str | None = None  # "Cover Drive"
    phase: str | None = None  # "powerplay"
    bowler_type: str | None = None  # "pace"
