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
    match_format: str | None = None  # "T20I", "T20", "ODI", "Test" (D-047)
    gender: str | None = None  # "male" or "female", as Cricsheet writes it (Day 42)
    team: str | None = None  # "India"
    dismissal_type: str | None = None  # "lbw"
    date: str | None = None  # "2024-04-12"
    k: int | None = None  # 6, as in "out within the next 6 balls"
    # filters added on Day 43
    opposition: str | None = None  # "Australia": the side played against
    season: str | None = None  # "2024" or "2024/25", as Cricsheet writes it
    since: str | None = None  # "2020" or "2020-06-01": from that day on
    innings: int | None = None  # 1: batted first
    batting_hand: str | None = None  # "left" or "right"
    side: str | None = None  # "batting" or "fielding": who asked for a review
    faced_so_far: int | None = None  # 3: he has faced 3 balls of this innings (Day 50)
