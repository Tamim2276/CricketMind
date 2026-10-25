"""How long the batter has been in (Day 50, on trial: D-056).

His innings in four stages, by the balls he has faced so far: 1-5, 6-15, 16-30, after
30. Each stage has its own risk, pulled toward normal for that stage.
"""

from dataclasses import dataclass

STARTS = (1, 6, 16, 31)  # the first ball faced of each stage


@dataclass(frozen=True, kw_only=True)
class Record:
    """A batter's times out and balls faced in one stage of his innings, under a
    question's filters."""

    batter: str  # his register id
    stage: int  # 1: balls 1-5, 2: balls 6-15, 3: balls 16-30, 4: after 30
    outs: int
    faced: int


def stage_of(ball: int) -> int:
    """The stage his ball faced number `ball` is in: his 6th ball is in stage 2."""
    found = 1
    for number, first in enumerate(STARTS, start=1):
        if ball >= first:
            found = number
    return found


def by_batter(records: list[Record]) -> list[tuple[int, int]]:
    """Each batter's (outs, faced) over all his stages: one risk for the whole innings."""
    totals = {}
    for record in records:
        outs, faced = totals.get(record.batter, (0, 0))
        totals[record.batter] = (outs + record.outs, faced + record.faced)
    return list(totals.values())


def in_stage(records: list[Record], stage: int) -> list[tuple[int, int]]:
    """Each batter's (outs, faced) in one stage."""
    found = []
    for record in records:
        if record.stage == stage:
            found.append((record.outs, record.faced))
    return found
