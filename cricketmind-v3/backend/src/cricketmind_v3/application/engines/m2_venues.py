"""N8 M2: what a ground is like in one format (Day 40)."""

import dataclasses

from cricketmind_v3.application.ports import VenueRepository
from cricketmind_v3.domain import venues
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.registry import is_empty, lookup, missing_keys
from cricketmind_v3.domain.subject import Subject


def venue_first_innings_avg(repo: VenueRepository, subject: Subject) -> Fact | None:
    """The engine's answer, or None when it has none for this subject."""
    if not _asks_only_its_menu_row(subject, "venue_first_innings_avg"):
        return None
    scores = repo.innings_at(subject.venue, subject.match_format)
    return venues.first_innings_average(scores)


def venue_chasing_win_share(repo: VenueRepository, subject: Subject) -> Fact | None:
    if not _asks_only_its_menu_row(subject, "venue_chasing_win_share"):
        return None
    scores = repo.innings_at(subject.venue, subject.match_format)
    return venues.chasing_win_share(scores)


def venue_wicket_share(repo: VenueRepository, subject: Subject) -> Fact | None:
    if not _asks_only_its_menu_row(subject, "venue_wicket_share"):
        return None
    wickets = repo.wickets_at(subject.venue, subject.match_format)
    return venues.wicket_share(wickets, subject.bowler_type)


def _asks_only_its_menu_row(subject: Subject, metric: str) -> bool:
    # every field the menu row needs, and nothing else: "India's first innings at
    # Mirpur" is a different question, and gets no answer rather than the ground's
    claim_type = lookup(metric)
    if missing_keys(claim_type, subject):
        return False
    for field in dataclasses.fields(subject):
        if field.name in claim_type.required_keys:
            continue
        if not is_empty(getattr(subject, field.name)):
            return False
    return True
