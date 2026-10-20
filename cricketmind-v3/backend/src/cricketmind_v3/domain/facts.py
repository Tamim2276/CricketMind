from dataclasses import dataclass


class FactError(ValueError):
    """An engine answer that contradicts itself."""


@dataclass(frozen=True, kw_only=True)
class Fact:
    """What an engine says about one subject: the number, its sample, its range."""

    value: float
    n: int | None = None  # balls, innings or reviews behind a count
    interval: tuple[float, float] | None = None  # 95% range, when the engine gives one

    def __post_init__(self):
        if self.interval is not None:
            low, high = self.interval
            if not low <= self.value <= high:
                raise FactError(
                    f"value {self.value} is outside its own range {low} to {high}"
                )
