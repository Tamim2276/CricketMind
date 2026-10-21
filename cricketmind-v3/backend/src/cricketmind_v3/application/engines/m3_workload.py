"""N8 M3: a bowler's workload this week against his usual week (Day 41)."""

from datetime import date

from cricketmind_v3.application.ports import FilterError, WorkloadRepository
from cricketmind_v3.domain import workload
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.registry import extra_keys, lookup, missing_keys
from cricketmind_v3.domain.subject import Subject


def workload_ratio_flag(repo: WorkloadRepository, subject: Subject) -> Fact | None:
    """The engine's answer, or None when it has none for this subject."""
    claim_type = lookup("workload_ratio_flag")
    if missing_keys(claim_type, subject) or extra_keys(claim_type, subject):
        return None
    try:
        on = date.fromisoformat(subject.date)
    except ValueError:
        return None  # not a date
    if on > repo.data_until():
        return None  # the data stops before then: the week would look empty
    try:
        spells = repo.spells(subject.bowler)
    except FilterError:
        return None  # several players share the name
    return workload.workload_ratio(spells, on)
