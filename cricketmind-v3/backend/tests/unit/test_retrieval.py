"""Day 60: N7a, the balls each side of a question is about."""

from cricketmind_v3.application.ports import FakeDeliveryRepository, FilterError
from cricketmind_v3.application.retrieval import (
    NO_BALLS,
    NO_DATABASE,
    NO_PERSON,
    WORDS_ONLY,
    retrieve,
)
from cricketmind_v3.domain.balls import Ball
from cricketmind_v3.domain.evidence import sides
from cricketmind_v3.domain.subject import CricketSubject


def ball(runs: int, out: bool = False) -> Ball:
    return Ball(
        match_type="T20",
        gender="male",
        batter="V Kohli",
        runs_batter=runs,
        wides=0,
        noballs=0,
        player_out="V Kohli" if out else None,
        wicket_kind="caught" if out else None,
    )


KOHLI = CricketSubject(player="V Kohli", match_format="T20")
ROHIT = CricketSubject(player="RG Sharma", match_format="T20")


class NoShotsYet(FakeDeliveryRepository):
    def deliveries(self, subject):
        raise FilterError("cannot pick balls by shot yet")


def test_k_is_not_a_filter_on_the_past():
    asked = CricketSubject(player="V Kohli", k=6)
    assert sides(asked) == [CricketSubject(player="V Kohli")]


def test_each_side_of_a_comparison_is_its_own_subject():
    asked = CricketSubject(player="V Kohli", shot="Cover Drive", other_shot="Pull")
    assert sides(asked) == [
        CricketSubject(player="V Kohli", shot="Cover Drive"),
        CricketSubject(player="V Kohli", shot="Pull"),
    ]
    what_if = CricketSubject(
        player="V Kohli", bowler="JJ Bumrah", other_bowler="MA Starc"
    )
    assert sides(what_if) == [
        CricketSubject(player="V Kohli", bowler="JJ Bumrah"),
        CricketSubject(player="V Kohli", bowler="MA Starc"),
    ]


def test_the_balls_are_filed_under_their_own_subject():
    repo = FakeDeliveryRepository()
    repo.add(KOHLI, [ball(4), ball(0), ball(0), ball(1, out=True)])
    repo.add(ROHIT, [ball(6)])
    (found,) = retrieve(KOHLI, "stats", repo)
    assert found.subject == KOHLI  # never Rohit's
    assert found.count == 4
    assert found.balls[ball(0)] == 2  # the same ball twice, counted once with 2
    assert len(found.balls) == 3
    assert found.none_because is None
    assert repo.asked == [KOHLI]


def test_k_is_left_out_of_what_is_asked():
    repo = FakeDeliveryRepository()
    repo.add(KOHLI, [ball(1)])
    (found,) = retrieve(
        CricketSubject(player="V Kohli", match_format="T20", k=6), "prediction", repo
    )
    assert found.count == 1
    assert repo.asked == [KOHLI]


def test_why_there_are_no_balls():
    repo = FakeDeliveryRepository()
    assert retrieve(KOHLI, "stats", repo)[0].none_because == NO_BALLS
    assert retrieve(KOHLI, "stats", None)[0].none_because == NO_DATABASE
    assert retrieve(KOHLI, "stats", NoShotsYet())[0].none_because == (
        "cannot pick balls by shot yet"
    )


def test_no_balls_are_fetched_without_a_person_or_for_a_rule():
    repo = FakeDeliveryRepository()
    ground = CricketSubject(venue="Wankhede Stadium")
    assert retrieve(ground, "stats", repo)[0].none_because == NO_PERSON
    assert retrieve(KOHLI, "rules", repo)[0].none_because == WORDS_ONLY
    assert repo.asked == []
