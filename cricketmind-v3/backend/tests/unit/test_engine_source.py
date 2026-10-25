"""Day 46: the real FactSource, wired to fakes: every N8 row on the menu has its engine."""

from datetime import date

from cricketmind_v3.application.engine_source import EngineFactSource, Repositories
from cricketmind_v3.application.ports import (
    FakeDeliveryRepository,
    FakeResultRepository,
    FakeReviewRepository,
    FakeVenueRepository,
    FakeWorkloadRepository,
)
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.registry import CLAIM_TYPES, check_registry
from cricketmind_v3.domain.render import METRICS
from cricketmind_v3.domain.subject import CricketSubject
from tests.examples import V2_COVER_DRIVE


def source(deliveries=None) -> EngineFactSource:
    if deliveries is None:
        deliveries = FakeDeliveryRepository()
    return EngineFactSource(
        Repositories(
            deliveries=deliveries,
            venues=FakeVenueRepository(),
            workload=FakeWorkloadRepository(date(2026, 10, 14)),
            reviews=FakeReviewRepository(),
            results=FakeResultRepository(),
        )
    )


def test_every_counted_row_on_the_menu_has_an_engine():
    engines = source().engines
    for claim_type in CLAIM_TYPES:
        if claim_type.source.startswith("N8"):
            assert claim_type.name in engines, claim_type.name


def test_the_registry_check_names_only_the_engines_still_to_build():
    problems = check_registry({}, source().sources(), set(METRICS))
    assert problems == [
        "weight_transfer: no engine provides N6",
        "matchup_difference: no engine provides N9.T2",
    ]


def test_a_metric_goes_to_its_engine():
    kohli = CricketSubject(player="V Kohli")
    balls = FakeDeliveryRepository()
    balls.add(kohli, V2_COVER_DRIVE)
    assert source(balls).fact("strike_rate", kohli) == Fact(value=142.0, n=50)


def test_no_engine_no_answer():
    kohli = CricketSubject(player="V Kohli")
    assert source().fact("matchup_difference", kohli) is None  # N9 T2: Days 51-52
    assert source().fact("weight_transfer", kohli, clip_id="vid1_0") is None  # N6
    assert source().fact("sponsor_value", kohli) is None  # not on the menu
