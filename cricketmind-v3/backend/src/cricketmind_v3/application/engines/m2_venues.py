"""N8 M2: what a ground is like in one format (Days 40, 43)."""

import dataclasses

from cricketmind_v3.application.ports import FilterError, VenueRepository
from cricketmind_v3.domain import venues
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.registry import extra_keys, lookup, missing_keys
from cricketmind_v3.domain.subject import Subject


def venue_first_innings_avg(repo: VenueRepository, subject: Subject) -> Fact | None:
    """The engine's answer, or None when it has none for this subject."""
    if not _asks_only_its_menu_row(subject, "venue_first_innings_avg"):
        return None
    try:
        return venues.first_innings_average(repo.innings_at(subject))
    except FilterError:
        return None  # a format or a date the data does not know


def venue_chasing_win_share(repo: VenueRepository, subject: Subject) -> Fact | None:
    if not _asks_only_its_menu_row(subject, "venue_chasing_win_share"):
        return None
    try:
        return venues.chasing_win_share(repo.innings_at(subject))
    except FilterError:
        return None


def venue_wicket_share(repo: VenueRepository, subject: Subject) -> Fact | None:
    if not _asks_only_its_menu_row(subject, "venue_wicket_share"):
        return None
    # pace or spin is which share to give, not which matches to count
    matches = dataclasses.replace(subject, bowler_type=None)
    try:
        return venues.wicket_share(repo.wickets_at(matches), subject.bowler_type)
    except FilterError:
        return None


def _asks_only_its_menu_row(subject: Subject, metric: str) -> bool:
    # every field the menu row needs, and nothing it cannot filter by: "India's first
    # innings at Mirpur" is a different question, and gets no answer
    claim_type = lookup(metric)
    return not missing_keys(claim_type, subject) and not extra_keys(claim_type, subject)
