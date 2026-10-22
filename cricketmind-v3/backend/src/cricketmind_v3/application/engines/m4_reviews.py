"""N8 M4: how often a team's reviews change the umpire's decision (Days 42-43)."""

from cricketmind_v3.application.ports import FilterError, ReviewRepository
from cricketmind_v3.domain import reviews
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.registry import extra_keys, lookup, missing_keys
from cricketmind_v3.domain.subject import Subject


def review_overturn_rate(repo: ReviewRepository, subject: Subject) -> Fact | None:
    """The engine's answer, or None when it has none for this subject."""
    claim_type = lookup("review_overturn_rate")
    if missing_keys(claim_type, subject) or extra_keys(claim_type, subject):
        return None  # by team only: a dismissal type cannot be answered (D-045)
    try:
        return reviews.overturn_rate(repo.decisions(subject))
    except FilterError:
        return None  # a side, format or date the data does not know
