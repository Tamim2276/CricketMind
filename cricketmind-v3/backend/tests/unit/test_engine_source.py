"""Day 46: the real FactSource, wired to fakes: every N8 row on the menu has its engine."""

import threading
import time
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
    assert problems == ["weight_transfer: no engine provides N6"]  # N9 T2: Days 51-52


def test_a_metric_goes_to_its_engine():
    kohli = CricketSubject(player="V Kohli")
    balls = FakeDeliveryRepository()
    balls.add(kohli, V2_COVER_DRIVE)
    assert source(balls).fact("strike_rate", kohli) == Fact(value=142.0, n=50)


def test_no_engine_no_answer():
    kohli = CricketSubject(player="V Kohli")
    assert source().fact("matchup_difference", kohli) is None  # no bowlers, k, format
    assert source().fact("weight_transfer", kohli, clip_id="vid1_0") is None  # N6
    assert source().fact("sponsor_value", kohli) is None  # not on the menu


class NoticesTwoAtOnce(FakeDeliveryRepository):
    """A slow query that notices when another one starts in the middle of it."""

    def __init__(self) -> None:
        super().__init__()
        self.running = 0
        self.overlaps = 0

    def deliveries(self, subject):
        self.running += 1
        if self.running > 1:
            self.overlaps += 1
        time.sleep(0.05)
        self.running -= 1
        return []


def test_two_nodes_never_query_at_the_same_time():
    # Day 61: on one connection, Kohli's query came back with Rohit's balls
    repo = NoticesTwoAtOnce()
    engines = source(repo)
    kohli = CricketSubject(player="V Kohli")
    threads = []
    for _ in range(4):
        threads.append(
            threading.Thread(target=engines.fact, args=("strike_rate", kohli))
        )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert repo.overlaps == 0
