"""Shrinkage (Day 50): a small record is pulled toward what is normal for batters like him.

Other batters' records under the same filters say two things: the normal out-rate, and
how much batters really differ. The less they differ, the harder he is pulled; the more
balls he has faced, the less he moves. 1 out in 2 balls is not a 50% risk.
"""

from dataclasses import dataclass

FEW_BATTERS = 30  # fewer others than this: too few to tell what normal is


@dataclass(frozen=True, kw_only=True)
class Prior:
    """What we believe before his own balls: a Beta curve of believable out-rates,
    worth so many pretend outs and pretend balls survived."""

    outs: float
    survived: float

    def normal(self) -> float:
        """The out-rate it leans to: 7.6 outs in 138.6 balls is 5.5%."""
        return self.outs / (self.outs + self.survived)

    def worth(self) -> float:
        """How many of his own balls it counts as."""
        return self.outs + self.survived


JEFFREYS = Prior(outs=0.5, survived=0.5)  # no view at all, worth one ball (Day 49)


def population_prior(records: list[tuple[int, int]]) -> Prior | None:
    """The prior from other batters' (times out, balls faced); None if it cannot be told.

    Their rates spread out partly because batters differ, partly by luck: a batter
    with 10 balls can look far from normal by luck alone. Take luck's share away and
    what is left is how much they really differ (the method of moments; Efron and
    Morris did it for batting averages).
    """
    batters = 0
    balls = 0
    outs = 0
    squares = 0
    for out, faced in records:
        if faced < 1:
            continue
        batters += 1
        balls += faced
        outs += out
        squares += faced * faced
    if batters < FEW_BATTERS or outs == 0 or outs == balls:
        return None
    normal = outs / balls
    luck = normal * (1 - normal)  # how much one ball varies at the normal rate

    # the spread we see: each batter's distance from normal, weighted by his balls
    seen = 0.0
    for out, faced in records:
        if faced >= 1:
            seen += faced * (out / faced - normal) ** 2
    # luck alone would give (batters - 1) * luck; what is left is the real spread
    room = balls - squares / balls - (batters - 1)
    if room <= 0:
        return None  # every batter faced one ball: real and luck cannot be told apart
    spread = (seen - (batters - 1) * luck) / room
    if spread <= 0:
        return None  # no real difference seen: we cannot tell how hard to pull
    # a Beta curve's spread is normal * (1 - normal) / (worth + 1)
    worth = luck / spread - 1
    if worth <= 0:
        return None
    return Prior(outs=normal * worth, survived=(1 - normal) * worth)


def shrunk_rate(prior: Prior, outs: int, faced: int) -> float:
    """His out-rate pulled toward normal: his outs and the prior's, over both their balls."""
    return (prior.outs + outs) / (prior.worth() + faced)
