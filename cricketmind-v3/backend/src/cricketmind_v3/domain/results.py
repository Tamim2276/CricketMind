"""M5: who won, and by how much (Day 45)."""

from dataclasses import dataclass

from cricketmind_v3.domain.facts import Fact


@dataclass(frozen=True, kw_only=True)
class Result:
    """How one match ended."""

    teams: tuple[str, str]  # ("Pakistan", "Sri Lanka")
    winner: str | None  # None after a tie, a draw or no result
    by_runs: int | None  # 82; with by_innings, "an innings and 82 runs"
    by_wickets: int | None
    by_innings: bool


def win_share(results: list[Result], team: str) -> Fact | None:
    """Of the team's matches that someone won, the share it won; n is those matches.

    A tie (even one settled by a super over), a draw or no result is left out.
    """
    decided = 0
    won = 0
    for result in results:
        if team not in result.teams or result.winner is None:
            continue
        decided += 1
        if result.winner == team:
            won += 1
    if decided == 0:
        return None
    return Fact(value=won / decided, n=decided)


def won_by_runs(results: list[Result], team: str) -> Fact | None:
    """The margin, when the team won this one match by runs; otherwise no answer.

    An innings win is left out: "by 20 runs" would hide "an innings and 20 runs".
    """
    result = _the_one_match(results)
    if result is None or result.winner != team:
        return None
    if result.by_runs is None or result.by_innings:
        return None
    return Fact(value=float(result.by_runs), n=1)


def won_by_wickets(results: list[Result], team: str) -> Fact | None:
    """The margin, when the team won this one match by wickets; otherwise no answer."""
    result = _the_one_match(results)
    if result is None or result.winner != team or result.by_wickets is None:
        return None
    return Fact(value=float(result.by_wickets), n=1)


def _the_one_match(results: list[Result]) -> Result | None:
    # no match on that day, or two: no answer rather than a guess
    if len(results) != 1:
        return None
    return results[0]
