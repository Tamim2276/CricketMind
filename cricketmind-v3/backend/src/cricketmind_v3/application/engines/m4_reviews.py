"""N8 M4: how often a team's reviews change the umpire's decision (Day 42)."""

from cricketmind_v3.application.ports import ReviewRepository
from cricketmind_v3.domain import reviews
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.registry import extra_keys, lookup, missing_keys
from cricketmind_v3.domain.subject import Subject


def review_overturn_rate(repo: ReviewRepository, subject: Subject) -> Fact | None:
    """The engine's answer, or None when it has none for this subject."""
    claim_type = lookup("review_overturn_rate")
    if missing_keys(claim_type, subject) or extra_keys(claim_type, subject):
        return None  # by team only: a dismissal type cannot be answered (D-045)
    return reviews.overturn_rate(repo.decisions(subject.team, subject.gender))
