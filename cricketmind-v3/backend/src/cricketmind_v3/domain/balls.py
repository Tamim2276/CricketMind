"""One ball as the engines count it, and the rules for counting it (Day 38)."""

from dataclasses import dataclass

# a batter who retires hurt, or retires and may come back, is not out (the Laws);
# "retired out" is a dismissal
NOT_DISMISSALS = ("retired hurt", "retired not out")

# the names on balls of many batters counted together (Day 39): whoever faced the ball,
# and anyone else who was out on it
THE_BATTER = "(the batter)"
SOMEONE_ELSE = "(the non-striker)"


@dataclass(frozen=True, kw_only=True)
class Ball:
    """Only what the engines count; the rest of the row stays in the database."""

    match_type: str  # "T20", "Test": like is set against like (Day 39)
    gender: str  # "male" or "female": never mixed (Day 42)
    batter: str  # "V Kohli", as named in that match
    runs_batter: int  # off the bat
    wides: int  # runs from a wide; 0 when the ball was not a wide
    player_out: str | None  # may be the non-striker
    wicket_kind: str | None  # "caught", "run out", "retired hurt", ...


def faced(ball: Ball) -> bool:
    """A wide is not a ball faced; a no-ball is, because the batter could hit it."""
    return ball.wides == 0


def batter_out(ball: Ball) -> bool:
    """The striker was dismissed on this ball. A non-striker's run out is not theirs."""
    if ball.player_out != ball.batter:
        return False
    return ball.wicket_kind not in NOT_DISMISSALS


def tally(balls: list[Ball]) -> dict[Ball, int]:
    """Each different ball once, with how many times it happened (Day 39).

    Kohli's 42,988 balls are only 48 different balls: a dot ball is a dot ball.
    """
    counts = {}
    for ball in balls:
        counts[ball] = counts.get(ball, 0) + 1
    return counts


def by_format_and_gender(
    counts: dict[Ball, int],
) -> dict[tuple[str, str], dict[Ball, int]]:
    """The tally split by kind of cricket: ("T20", "male") -> its balls, and so on."""
    split = {}
    for ball, times in counts.items():
        kind = (ball.match_type, ball.gender)
        if kind not in split:
            split[kind] = {}
        split[kind][ball] = times
    return split
