"""Days 49-50: the N9 T1 engine, on the fake repository (no database).

SYNTHETIC: Kohli's 4 outs in 50 balls is the v2 example; the other batters are made up.
"""

import dataclasses

from cricketmind_v3.application.engines.n9_t1 import p_dismissal_within_k
from cricketmind_v3.application.ports import FakeDeliveryRepository, FilterError
from cricketmind_v3.domain.out_within import Stage, out_within, out_within_stages
from cricketmind_v3.domain.shrinkage import population_prior
from cricketmind_v3.domain.stages import Record, by_batter, in_stage
from cricketmind_v3.domain.subject import CricketSubject
from tests.examples import KOHLI_PULL, V2_PULL, v2_balls
from tests.unit.test_engine_source import source
from tests.unit.test_out_within import exact_chances

PAST = dataclasses.replace(KOHLI_PULL, k=None)  # the balls behind the question
LIKE_HIM = dataclasses.replace(PAST, gender="male")  # who he is pulled toward


def made_up_batters(order: str, careful: tuple, risky: tuple) -> list[Record]:
    """40 other batters in one part of the order, 20 careful and 20 risky, each with
    (outs, faced) in two stages: balls 1-5 and balls 6-15."""
    records = []
    for i in range(20):
        for kind, stages in (("careful", careful), ("risky", risky)):
            for stage, (outs, faced) in enumerate(stages, start=1):
                records.append(
                    Record(
                        batter=f"{order} {kind} {i}",
                        stage=stage,
                        order=order,
                        outs=outs,
                        faced=faced,
                    )
                )
    return records


# the top order: careful out 10 times in 500 balls (2%), risky 40 in 500 (8%)
TOP = made_up_batters("top", ((4, 100), (6, 400)), ((12, 100), (28, 400)))
# the tail: careful out 30 times in 300 balls (10%), risky 60 in 300 (20%)
TAIL = made_up_batters("tail", ((12, 100), (18, 200)), ((24, 100), (36, 200)))
THEM = TOP + TAIL
# his 4 outs in 50 balls, by stage: 2 in his first 20 balls of stage 1, 2 in 30 later,
# batting in the top order
HIS = [
    Record(batter="V Kohli", stage=1, order="top", outs=2, faced=20),
    Record(batter="V Kohli", stage=2, order="top", outs=2, faced=30),
]


class RefusingRepository(FakeDeliveryRepository):
    def deliveries(self, subject):
        raise FilterError("cannot pick balls by shot yet")


def pull_balls() -> FakeDeliveryRepository:
    repo = FakeDeliveryRepository()
    repo.add(PAST, V2_PULL)
    repo.add_records(PAST, HIS)
    repo.add_others_records(LIKE_HIM, THEM)
    return repo


def test_the_engine_pulls_his_record_toward_normal():
    repo = pull_balls()
    found = p_dismissal_within_k(repo, KOHLI_PULL)
    expected = out_within(4, 50, 6, population_prior(by_batter(TOP)))
    assert found.value == expected.chance
    assert found.interval == (expected.low, expected.high)
    assert found.n == 50
    # his balls, where he bats, then the other batters: his format, and men's like him
    assert repo.asked == [PAST, PAST, LIKE_HIM]
    assert len(found.replays.overs) == 100
    assert len(found.replays.survival) == 7  # now and after each of the six balls
    # on his own (Day 49): 0.4056, from 0.1568 to 0.6917. Pulled toward the others'
    # 5%, his 8% comes down to about 6.4% and the range narrows: the curve is
    # Beta(2.81 + 4, 53.43 + 46), and adding it up gives the same answer
    mid, low, high = exact_chances(4, 50, 6, population_prior(by_batter(TOP)))
    assert abs(found.value - mid) < 0.01
    assert abs(found.interval[0] - low) < 0.01
    assert abs(found.interval[1] - high) < 0.01
    assert found.interval[1] - found.interval[0] < 0.6917 - 0.1568
    assert round(found.value, 4) == 0.3156  # seed 42: pinned


def test_one_out_in_two_balls_is_not_a_50_percent_risk():
    new = CricketSubject(player="V Kohli", k=1, match_format="T20")
    past = dataclasses.replace(new, k=None)
    repo = FakeDeliveryRepository()
    repo.add(past, v2_balls(runs=1, outs=1, balls=2))  # out once in two balls
    repo.add_records(
        past, [Record(batter="V Kohli", stage=1, order="top", outs=1, faced=2)]
    )
    repo.add_others_records(dataclasses.replace(past, gender="male"), THEM)
    found = p_dismissal_within_k(repo, new)
    assert 0.04 < found.value < 0.08  # about 6.5%, near the others' 5%
    assert found.n == 2


def test_a_question_needs_a_batter_how_many_balls_and_the_format():
    repo = pull_balls()
    assert p_dismissal_within_k(repo, PAST) is None  # no k
    assert p_dismissal_within_k(repo, dataclasses.replace(KOHLI_PULL, k=0)) is None
    no_player = dataclasses.replace(KOHLI_PULL, player=None)
    assert p_dismissal_within_k(repo, no_player) is None
    no_format = dataclasses.replace(KOHLI_PULL, match_format=None)
    assert p_dismissal_within_k(repo, no_format) is None
    on_a_day = dataclasses.replace(KOHLI_PULL, date="2024-04-12")
    assert p_dismissal_within_k(repo, on_a_day) is None  # not a filter of this row


def test_no_answer_when_normal_cannot_be_told():
    repo = pull_balls()
    repo.add_others_records(LIKE_HIM, THEM[:40])  # only 10 other batters
    assert p_dismissal_within_k(repo, KOHLI_PULL) is None
    nobody = FakeDeliveryRepository()
    nobody.add(PAST, V2_PULL)  # his balls, but no other batter at all
    assert p_dismissal_within_k(nobody, KOHLI_PULL) is None


# Where he bats (L-045)


def test_a_tailender_is_pulled_toward_the_tail():
    # 12 outs in 107 balls, batting at 8 to 11: normal is the tail's 15%, not the top's 5%
    bumrah = CricketSubject(player="JJ Bumrah", k=6, match_format="T20")
    past = dataclasses.replace(bumrah, k=None)
    repo = FakeDeliveryRepository()
    repo.add(past, v2_balls(runs=60, outs=12, balls=107))
    repo.add_records(
        past,
        [
            Record(batter="JJ Bumrah", stage=1, order="tail", outs=6, faced=60),
            Record(batter="JJ Bumrah", stage=2, order="tail", outs=6, faced=47),
        ],
    )
    repo.add_others_records(dataclasses.replace(past, gender="male"), THEM)
    found = p_dismissal_within_k(repo, bumrah)
    tail = out_within(12, 107, 6, population_prior(by_batter(TAIL)))
    top = out_within(12, 107, 6, population_prior(by_batter(TOP)))
    assert found.value == tail.chance
    assert found.value > top.chance  # the top order would make him look safer


def test_he_is_where_most_of_his_balls_were():
    repo = pull_balls()
    # a few balls in the tail as a nightwatchman: he is still a top-order batter
    repo.add_records(
        PAST, [*HIS, Record(batter="V Kohli", stage=1, order="tail", outs=1, faced=3)]
    )
    found = p_dismissal_within_k(repo, KOHLI_PULL)
    expected = out_within(4, 50, 6, population_prior(by_batter(TOP)))
    assert found.value == expected.chance


def test_no_record_of_where_he_bats_gives_no_answer():
    repo = pull_balls()
    repo.add_records(PAST, [])
    assert p_dismissal_within_k(repo, KOHLI_PULL) is None
    assert repo.asked == [PAST, PAST]  # stopped before asking for the others


def test_a_filter_the_data_cannot_apply_gives_no_answer():
    assert p_dismissal_within_k(RefusingRepository(), KOHLI_PULL) is None


# How long he has been in (on trial, D-056)


def stage_prior(number: int):
    return population_prior(in_stage(TOP, number))


def test_each_of_his_next_balls_has_the_risk_of_its_stage():
    repo = pull_balls()
    # he has faced 3: his next six are balls 4 to 9, two in stage 1 and four in stage 2
    found = p_dismissal_within_k(repo, dataclasses.replace(KOHLI_PULL, faced_so_far=3))
    expected = out_within_stages(
        [
            Stage(first=1, outs=2, faced=20, prior=stage_prior(1)),
            Stage(first=6, outs=2, faced=30, prior=stage_prior(2)),
        ],
        6,
        3,
    )
    assert found.value == expected.chance
    assert found.interval == (expected.low, expected.high)
    assert found.n == 50  # his balls in the two stages
    assert repo.asked == [PAST, PAST, LIKE_HIM]  # his balls, his record, the others


def test_just_come_in():
    repo = pull_balls()
    # balls 1 to 5: his first stage only, so only its 20 balls are behind the answer
    found = p_dismissal_within_k(
        repo, dataclasses.replace(KOHLI_PULL, k=5, faced_so_far=0)
    )
    expected = out_within_stages(
        [Stage(first=1, outs=2, faced=20, prior=stage_prior(1))], 5, 0
    )
    assert found.value == expected.chance
    assert found.n == 20


def test_a_stage_with_no_others_gives_no_answer():
    repo = pull_balls()
    # balls 13 to 18 reach stage 3, where no other batter has a record
    later = dataclasses.replace(KOHLI_PULL, faced_so_far=12)
    assert p_dismissal_within_k(repo, later) is None


def test_balls_faced_so_far_must_be_a_count():
    repo = pull_balls()
    for wrong in (-1, "3", 2.5):
        odd = dataclasses.replace(KOHLI_PULL, faced_so_far=wrong)
        assert p_dismissal_within_k(repo, odd) is None
    assert repo.asked == []  # refused before the data is asked


def test_through_the_real_source_it_has_a_receipt_and_keeps_its_replays():
    engines = source(pull_balls())
    found = engines.fact("p_dismissal_within_k", KOHLI_PULL)
    assert found.receipt.engine == "N9.T1"
    assert found.receipt.version == "2"  # pulled toward normal since Day 50
    assert found.receipt.interval == found.interval
    assert found.replays is not None  # stamping the receipt keeps the pictures' summary
    other = CricketSubject(player="RG Sharma", k=6, match_format="T20")
    assert engines.fact("p_dismissal_within_k", other) is None  # no balls: no answer
