"""Day 58, part 2c: which team a question means. Every case is one the real data has."""

from cricketmind_v3.domain.teams import Team, describe_team, find_teams, other_name

TEAMS = [
    Team(name="India", matches=1353),
    Team(name="Hyderabad (India)", matches=37),
    Team(name="West Indies", matches=1070),
    Team(name="Sunrisers Hyderabad", matches=211),
    Team(name="Sunrisers", matches=76),  # an English women's side
    Team(name="Sunrisers Eastern Cape", matches=47),
    Team(name="Royal Challengers Bangalore", matches=258),
    Team(name="Royal Challengers Bengaluru", matches=63),
]


def names(found):
    result = []
    for team in found:
        result.append(team.name)
    return result


def test_the_whole_name_and_words_of_names_together():
    assert names(find_teams("India", TEAMS)) == ["India", "Hyderabad (India)"]
    # a team named exactly "Sunrisers" must not hide Sunrisers Hyderabad
    assert names(find_teams("Sunrisers", TEAMS)) == [
        "Sunrisers Hyderabad",
        "Sunrisers",
        "Sunrisers Eastern Cape",
    ]


def test_a_nickname_checked_by_hand():
    assert names(find_teams("Windies", TEAMS)) == ["West Indies"]
    assert names(find_teams("RCB", TEAMS)) == ["Royal Challengers Bengaluru"]


def test_nothing_found():
    assert find_teams("Hogwarts", TEAMS) == []
    assert find_teams("", TEAMS) == []


def test_a_renamed_team_knows_its_other_name():
    bengaluru = Team(name="Royal Challengers Bengaluru")
    assert other_name(bengaluru) == "Royal Challengers Bangalore"
    assert other_name(Team(name="Delhi Daredevils")) == "Delhi Capitals"
    assert other_name(Team(name="India")) is None


def test_described_without_a_statistic():
    assert describe_team(
        Team(name="Delhi Capitals", first_year=2019, last_year=2026)
    ) == ("Delhi Capitals; 2019-2026")
    assert describe_team(Team(name="India")) == "India"
