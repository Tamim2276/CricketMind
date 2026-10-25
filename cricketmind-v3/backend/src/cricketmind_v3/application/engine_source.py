"""The real FactSource: the Faith Check asks the engines themselves (Day 46), and keeps
a receipt of every answer (Day 47)."""

import dataclasses
from dataclasses import dataclass

from cricketmind_v3.application.engines import (
    m1_rates,
    m2_venues,
    m3_workload,
    m4_reviews,
    m5_results,
    n9_t1,
)
from cricketmind_v3.application.ports import (
    DeliveryRepository,
    ResultRepository,
    ReviewRepository,
    VenueRepository,
    WorkloadRepository,
)
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.receipts import Receipt, query_text
from cricketmind_v3.domain.registry import lookup
from cricketmind_v3.domain.subject import Subject

# each engine's version, named by its receipts: raise it when a counting rule changes,
# so an old receipt says it was made by other rules (D-052)
ENGINE_VERSIONS = {
    "N8.M1": "1",
    "N8.M2": "1",
    "N8.M3": "1",
    "N8.M4": "1",
    "N8.M5": "1",
    "N9.T1": "1",
}


@dataclass(frozen=True, kw_only=True)
class Repositories:
    """Where each engine reads its data."""

    deliveries: DeliveryRepository
    venues: VenueRepository
    workload: WorkloadRepository
    reviews: ReviewRepository
    results: ResultRepository


class EngineFactSource:
    """A menu metric -> the engine function that computes it, with the data it reads."""

    def __init__(self, repos: Repositories, data: str = "unnamed data") -> None:
        self.data = data  # the data snapshot's fingerprint, named by every receipt
        self.engines = {
            # N8 M1: balls, any filters
            "strike_rate": (m1_rates.strike_rate, repos.deliveries),
            "dismissal_rate": (m1_rates.dismissal_rate, repos.deliveries),
            "sr_difference": (m1_rates.sr_difference, repos.deliveries),
            "sr_vs_average": (m1_rates.sr_vs_average, repos.deliveries),
            "runs": (m1_rates.runs, repos.deliveries),
            "batting_average": (m1_rates.batting_average, repos.deliveries),
            "boundary_share": (m1_rates.boundary_share, repos.deliveries),
            "dot_ball_share": (m1_rates.dot_ball_share, repos.deliveries),
            "wickets": (m1_rates.wickets, repos.deliveries),
            "economy": (m1_rates.economy, repos.deliveries),
            "bowling_strike_rate": (m1_rates.bowling_strike_rate, repos.deliveries),
            # N8 M2: a ground
            "venue_first_innings_avg": (
                m2_venues.venue_first_innings_avg,
                repos.venues,
            ),
            "venue_chasing_win_share": (
                m2_venues.venue_chasing_win_share,
                repos.venues,
            ),
            "venue_wicket_share": (m2_venues.venue_wicket_share, repos.venues),
            # N8 M3: a bowler's workload
            "workload_ratio_flag": (m3_workload.workload_ratio_flag, repos.workload),
            # N8 M4: reviews
            "review_overturn_rate": (m4_reviews.review_overturn_rate, repos.reviews),
            # N8 M5: results
            "team_win_share": (m5_results.team_win_share, repos.results),
            "match_won_by_runs": (m5_results.match_won_by_runs, repos.results),
            "match_won_by_wickets": (m5_results.match_won_by_wickets, repos.results),
            # N9 T1: the chance of being out soon (Day 49)
            "p_dismissal_within_k": (n9_t1.p_dismissal_within_k, repos.deliveries),
        }

    def fact(
        self, metric: str, subject: Subject, clip_id: str | None = None
    ) -> Fact | None:
        """The engine's own answer, or None when there is no engine or no answer."""
        if clip_id is not None:
            return None  # a clip's numbers are measured by N6, not counted here
        found = self.engines.get(metric)
        if found is None:
            return None  # not on the menu, or its engine is not built yet
        engine, repository = found
        answer = engine(repository, subject)
        if answer is None:
            return None
        source = lookup(metric).source
        receipt = Receipt(
            data=self.data,
            engine=source,
            version=ENGINE_VERSIONS[source],
            query=query_text(metric, subject, clip_id),
            value=answer.value,
            n=answer.n,
            interval=answer.interval,
        )
        return dataclasses.replace(answer, receipt=receipt)

    def version_of(self, engine: str) -> str | None:
        """The engine's version now, to set against an old receipt's (Day 48)."""
        return ENGINE_VERSIONS.get(engine)

    def sources(self) -> set[str]:
        """The engines behind every metric this source answers: "N8.M1", ..."""
        names = set()
        for metric in self.engines:
            names.add(lookup(metric).source)
        return names
