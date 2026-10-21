"""One ball as the engines count it, and the rules for counting it (Day 38)."""

from dataclasses import dataclass

# a batter who retires hurt, or retires and may come back, is not out (the Laws);
# "retired out" is a dismissal
NOT_DISMISSALS = ("retired hurt", "retired not out")


@dataclass(frozen=True, kw_only=True)
class Ball:
    """Only what the engines count; the rest of the row stays in the database."""

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
