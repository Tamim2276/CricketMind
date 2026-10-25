"""Day 49: the N9 T1 engine, on the fake repository (no database)."""

import dataclasses

from cricketmind_v3.application.engines.n9_t1 import p_dismissal_within_k
from cricketmind_v3.application.ports import FakeDeliveryRepository, FilterError
from cricketmind_v3.domain.subject import CricketSubject
from tests.examples import KOHLI_PULL, V2_PULL
from tests.unit.test_engine_source import source

PAST = dataclasses.replace(KOHLI_PULL, k=None)  # the balls behind the question


class RefusingRepository(FakeDeliveryRepository):
    def deliveries(self, subject):
        raise FilterError("cannot pick balls by shot yet")


def pull_balls() -> FakeDeliveryRepository:
    repo = FakeDeliveryRepository()
    repo.add(PAST, V2_PULL)  # SYNTHETIC: 4 outs in 50 balls
    return repo


def test_the_engine_reads_his_past_and_plays_it_forward():
    repo = pull_balls()
    found = p_dismissal_within_k(repo, KOHLI_PULL)
    assert round(found.value, 4) == 0.4056
    assert found.n == 50
    assert (round(found.interval[0], 4), round(found.interval[1], 4)) == (
        0.1568,
        0.6917,
    )
    assert repo.asked == [PAST]  # k is about the future: not a filter on the past
    assert len(found.replays.overs) == 100
    assert len(found.replays.survival) == 7  # now and after each of the six balls


def test_a_question_needs_a_batter_and_how_many_balls():
    repo = pull_balls()
    assert p_dismissal_within_k(repo, PAST) is None  # no k
    assert p_dismissal_within_k(repo, dataclasses.replace(KOHLI_PULL, k=0)) is None
    no_player = dataclasses.replace(KOHLI_PULL, player=None)
    assert p_dismissal_within_k(repo, no_player) is None
    on_a_day = dataclasses.replace(KOHLI_PULL, date="2024-04-12")
    assert p_dismissal_within_k(repo, on_a_day) is None  # not a filter of this row


def test_a_filter_the_data_cannot_apply_gives_no_answer():
    assert p_dismissal_within_k(RefusingRepository(), KOHLI_PULL) is None


def test_through_the_real_source_it_has_a_receipt_and_keeps_its_replays():
    engines = source(pull_balls())
    found = engines.fact("p_dismissal_within_k", KOHLI_PULL)
    assert found.receipt.engine == "N9.T1"
    assert found.receipt.interval == found.interval
    assert found.replays is not None  # stamping the receipt keeps the pictures' summary
    other = CricketSubject(player="RG Sharma", k=6)
    assert engines.fact("p_dismissal_within_k", other) is None  # no balls: no answer
