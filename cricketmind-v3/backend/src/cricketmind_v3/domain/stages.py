"""How long the batter has been in, and where he bats (Day 50).

His innings in four stages, by the balls he has faced so far: 1-5, 6-15, 16-30, after
30. Each stage has its own risk, pulled toward normal for that stage. Tried on Day 53 and
not kept for the engine (D-066); kept here for the comparisons of Day 157.
Normal is for batters who bat where he bats: a tailender is not pulled toward the top
order (L-045).
"""

from dataclasses import dataclass

STARTS = (1, 6, 16, 31)  # the first ball faced of each stage

# the parts of the batting order, by the first place in each: 1-3, 4-7, then 8 on
ORDERS = (("top", 1), ("middle", 4), ("tail", 8))


@dataclass(frozen=True, kw_only=True)
class Record:
    """A batter's times out and balls faced in one stage of his innings, batting in one
    part of the order, under a question's filters."""

    batter: str  # his register id
    stage: int  # 1: balls 1-5, 2: balls 6-15, 3: balls 16-30, 4: after 30
    order: str  # "top", "middle" or "tail": where he batted in those innings
    outs: int
    faced: int


def stage_of(ball: int) -> int:
    """The stage his ball faced number `ball` is in: his 6th ball is in stage 2."""
    found = 1
    for number, first in enumerate(STARTS, start=1):
        if ball >= first:
            found = number
    return found


def order_of(place: int) -> str:
    """The part of the order a place is in: 3 is top, 4 is middle, 11 is tail."""
    found = ORDERS[0][0]
    for name, first in ORDERS:
        if place >= first:
            found = name
    return found


def usual_order(records: list[Record]) -> str | None:
    """Where he bats: the part of the order he faced the most balls in (the top first,
    if two are level). None if he faced no balls."""
    faced = {}
    for record in records:
        faced[record.order] = faced.get(record.order, 0) + record.faced
    found = None
    most = 0
    for name, _ in ORDERS:
        if faced.get(name, 0) > most:
            found = name
            most = faced[name]
    return found


def in_order(records: list[Record], order: str) -> list[Record]:
    """Only the records of balls faced batting in this part of the order."""
    found = []
    for record in records:
        if record.order == order:
            found.append(record)
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
