from dataclasses import dataclass, field

from cricketmind_v3.domain.receipts import Receipt


class FactError(ValueError):
    """An engine answer that contradicts itself."""


@dataclass(frozen=True, kw_only=True)
class Replays:
    """A prediction's simulation, kept small for the pictures (Day 49, D-054)."""

    # after each ball, from now (ball 0) to ball k: the chance he is still batting,
    # as (middle, low, high)
    survival: tuple[tuple[float, float, float], ...]
    # 100 overs that stand for all the replays: the ball he was out on, or 0
    overs: tuple[int, ...]


@dataclass(frozen=True, kw_only=True)
class Fact:
    """What an engine says about one subject: the number, its sample, its range."""

    value: float
    n: int | None = None  # balls, innings or reviews behind a count
    interval: tuple[float, float] | None = None  # 95% range, when the engine gives one
    # where the number came from (Day 47); not part of the number, so two facts with
    # the same number are equal whatever their receipts
    receipt: Receipt | None = field(default=None, compare=False)
    # a prediction's replays, for the pictures (Day 49); not part of the number either
    replays: Replays | None = field(default=None, compare=False)

    def __post_init__(self):
        if self.interval is not None:
            low, high = self.interval
            if not low <= self.value <= high:
                raise FactError(
                    f"value {self.value} is outside its own range {low} to {high}"
                )
