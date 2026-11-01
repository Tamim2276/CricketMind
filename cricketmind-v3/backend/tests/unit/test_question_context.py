"""Day 60: N7b, who the people are and where the ground is."""

from cricketmind_v3.domain.grounds import Ground
from cricketmind_v3.domain.names import Player
from cricketmind_v3.domain.question_context import question_context
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.domain.teams import Team

KOHLI = Player(
    person_id="ba607b88",
    name="V Kohli",
    matches=600,
    teams=("India", "Royal Challengers Bengaluru"),
    formats=("Test", "ODI"),
    first_year=2008,
    last_year=2025,
)
BUMRAH = Player(person_id="462411b3", name="JJ Bumrah", teams=("India",))
# two players share one name: the subject then carries the register id (Day 58)
MENDIS_1 = Player(person_id="aaaa1111", name="Mendis", teams=("Sri Lanka",))
MENDIS_2 = Player(person_id="bbbb2222", name="Mendis", teams=("Sri Lanka A",))
PLAYERS = (KOHLI, BUMRAH, MENDIS_1, MENDIS_2)
GROUNDS = (Ground(name="Wankhede Stadium, Mumbai", places=("Mumbai",)),)
TEAMS = (Team(name="Delhi Capitals", first_year=2019, last_year=2025),)


def context(subject: CricketSubject):
    return question_context(subject, PLAYERS, GROUNDS, TEAMS)


def test_who_the_people_are_in_the_registers_words():
    found = context(CricketSubject(player="V Kohli", bowler="JJ Bumrah"))
    assert found.people == (
        "V Kohli; India, Royal Challengers Bengaluru; Test, ODI; 2008-2025",
        "JJ Bumrah; India",
    )
    assert found.not_found == ()


def test_never_a_statistic():
    # 600 matches is in the register, and stays there
    assert "600" not in context(CricketSubject(player="V Kohli")).people[0]


def test_a_shared_name_by_its_register_id():
    assert context(CricketSubject(player="bbbb2222")).people == ("Mendis; Sri Lanka A",)
    found = context(CricketSubject(player="Mendis"))  # never guessed
    assert found.people == ()
    assert found.not_found == ("Mendis",)


def test_the_ground_and_a_renamed_team():
    found = context(
        CricketSubject(venue="Wankhede Stadium, Mumbai", team="Delhi Capitals")
    )
    assert found.ground == "Wankhede Stadium, Mumbai; Mumbai"
    assert found.teams == (
        "Delhi Capitals; 2019-2025; also played as Delhi Daredevils",
    )


def test_what_the_registers_do_not_hold_is_said():
    found = context(
        CricketSubject(player="Nobody", venue="The Moon", opposition="Mars XI")
    )
    assert found.not_found == ("Nobody", "The Moon", "Mars XI")
    assert found.ground is None
    assert context(CricketSubject()).not_found == ()
