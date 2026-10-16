from dataclasses import dataclass


@dataclass(frozen=True)
class Ball:
    batter: str
    runs: int
    is_wide: bool = False


def strike_rate(balls: list[Ball]) -> float | None:
    runs = 0
    faced = 0
    for ball in balls:
        if ball.is_wide:
            continue  # a wide is not a ball faced
        faced += 1
        runs += ball.runs
    if faced == 0:
        return None
    return round(100.0 * runs / faced, 1)
