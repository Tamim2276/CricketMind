"""N8 M5: who won, and by how much (Day 45)."""

from cricketmind_v3.application.ports import FilterError, ResultRepository
from cricketmind_v3.domain import results
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.registry import extra_keys, lookup, missing_keys
from cricketmind_v3.domain.subject import Subject


def team_win_share(repo: ResultRepository, subject: Subject) -> Fact | None:
    """The engine's answer, or None when it has none for this subject."""
    found = _results(repo, "team_win_share", subject)
    if found is None:
        return None
    return results.win_share(found, subject.team)


def match_won_by_runs(repo: ResultRepository, subject: Subject) -> Fact | None:
    """The subject's team is the winner it claims: if it lost, there is no answer."""
    found = _results(repo, "match_won_by_runs", subject)
    if found is None:
        return None
    return results.won_by_runs(found, subject.team)


def match_won_by_wickets(repo: ResultRepository, subject: Subject) -> Fact | None:
    found = _results(repo, "match_won_by_wickets", subject)
    if found is None:
        return None
    return results.won_by_wickets(found, subject.team)


def _results(repo: ResultRepository, metric: str, subject: Subject):
    claim_type = lookup(metric)
    if missing_keys(claim_type, subject) or extra_keys(claim_type, subject):
        return None
    try:
        return repo.results(subject)
    except FilterError:
        return None
