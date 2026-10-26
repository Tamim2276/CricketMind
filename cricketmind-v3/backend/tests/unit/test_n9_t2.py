"""Days 51-52: the N9 T2 engine, on the fake repository (no database).

SYNTHETIC: the batters, bowlers and their records are made up (tests/unit/test_matchups).
"""

import dataclasses

from cricketmind_v3.application.engines.n9_t2 import matchup_difference
from cricketmind_v3.application.ports import FakeDeliveryRepository, FilterError
from cricketmind_v3.domain.matchups import Pair, in_order, matchup
from cricketmind_v3.domain.stages import Record
from cricketmind_v3.domain.subject import CricketSubject
from tests.examples import V2_PULL
from tests.unit.test_engine_source import source
from tests.unit.test_matchups import field

QUESTION = CricketSubject(
    player="V Kohli",
    bowler="JJ Bumrah",
    other_bowler="Mohammed Shami",
    k=6,
    match_format="T20",
)
HIS = CricketSubject(player="V Kohli", match_format="T20")  # his balls, any bowler
EVERYONE = CricketSubject(match_format="T20", gender="male")  # every batter and bowler
NAMES = {"K": "V Kohli", "B": "JJ Bumrah", "S": "Mohammed Shami"}


def named_field() -> list[Pair]:
    # the made-up field, with K, B and S named; and 40 tailenders out half the time
    # against Bumrah, who must not count for a top-order batter
    pairs = []
    for pair in field():
        pairs.append(
            dataclasses.replace(
                pair,
                batter=NAMES.get(pair.batter, pair.batter),
                bowler=NAMES.get(pair.bowler, pair.bowler),
            )
        )
    for i in range(40):
        pairs.append(
            Pair(
                batter=f"tail{i}",
                bowler="JJ Bumrah",
                order="tail",
                outs=500,
                faced=1000,
            )
        )
    return pairs


PAIRS = named_field()


class SharedNames(FakeDeliveryRepository):
    def person(self, name):
        raise FilterError(f"3 players are called {name}: say which one")


def balls(repo=None) -> FakeDeliveryRepository:
    if repo is None:
        repo = FakeDeliveryRepository()
    repo.add(HIS, V2_PULL)
    repo.add_records(
        HIS, [Record(batter="V Kohli", stage=1, order="top", outs=4, faced=50)]
    )
    repo.add_pairs(EVERYONE, PAIRS)
    return repo


def test_the_engine_asks_for_the_field_and_plays_both_sides():
    repo = balls()
    found = matchup_difference(repo, QUESTION)
    expected = matchup(
        in_order(PAIRS, "top"), "V Kohli", "JJ Bumrah", "Mohammed Shami", 6
    )
    assert found.value == expected.difference
    assert found.interval == (expected.low, expected.high)
    assert found.n == expected.n
    # his balls (men's), where he bats, then everyone under the same filters
    assert repo.asked == [HIS, HIS, EVERYONE]
    # both sides' pictures, over for over
    assert found.replays.overs == expected.this.overs
    assert found.replays.versus.overs == expected.that.overs
    assert len(found.replays.versus.survival) == 7


def test_only_batters_who_bat_where_he_bats():
    # the tailenders would make Bumrah look far more dangerous to Kohli
    every_place = matchup(PAIRS, "V Kohli", "JJ Bumrah", "Mohammed Shami", 6)
    found = matchup_difference(balls(), QUESTION)
    assert found.value < every_place.difference


def test_the_menu_decides_the_question():
    repo = balls()
    for missing in ("other_bowler", "k", "match_format"):
        thin = dataclasses.replace(QUESTION, **{missing: None})
        assert matchup_difference(repo, thin) is None
    for extra, value in (
        ("bowler_type", "pace"),
        ("opposition", "Australia"),
        ("faced_so_far", 3),
        ("shot", "Pull"),
    ):
        odd = dataclasses.replace(QUESTION, **{extra: value})
        assert matchup_difference(repo, odd) is None
    assert matchup_difference(repo, dataclasses.replace(QUESTION, k=0)) is None
    same = dataclasses.replace(QUESTION, other_bowler="JJ Bumrah")
    assert matchup_difference(repo, same) is None
    assert repo.asked == []  # refused before the data is asked


def test_the_phase_narrows_it():
    repo = balls()
    death = dataclasses.replace(QUESTION, phase="death")
    matchup_difference(repo, death)  # no death-overs balls here: no answer
    assert repo.asked[0] == dataclasses.replace(HIS, phase="death")


def test_a_name_several_players_share_gives_no_answer():
    assert matchup_difference(balls(SharedNames()), QUESTION) is None


def test_no_balls_or_no_place_in_the_order_gives_no_answer():
    assert matchup_difference(FakeDeliveryRepository(), QUESTION) is None
    repo = balls()
    repo.add_records(HIS, [])
    assert matchup_difference(repo, QUESTION) is None


def test_through_the_real_source_it_has_a_receipt_and_keeps_both_sides():
    found = source(balls()).fact("matchup_difference", QUESTION)
    assert found.receipt.engine == "N9.T2"
    assert found.receipt.version == "1"
    assert found.receipt.interval == found.interval
    assert found.replays.versus is not None
