from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Subject:
    """Who and what a number is about. Each sport adds its own fields (D-020)."""


@dataclass(frozen=True, kw_only=True)
class CricketSubject(Subject):
    """A cricket subject. An empty field means "any"."""

    player: str | None = None  # "V Kohli"
    shot: str | None = None  # "Cover Drive"
    other_shot: str | None = None  # "Pull", the other side of a comparison
    phase: str | None = None  # "powerplay"
    bowler_type: str | None = None  # "pace"
    bowler: str | None = None  # "JJ Bumrah"
    other_bowler: str | None = None  # "Rashid Khan", the other side of a comparison
    venue: str | None = None  # "Wankhede Stadium"
    match_format: str | None = None  # "T20"
    team: str | None = None  # "India"
    dismissal_type: str | None = None  # "lbw"
    date: str | None = None  # "2024-04-12"
    k: int | None = None  # 6, as in "out within the next 6 balls"
