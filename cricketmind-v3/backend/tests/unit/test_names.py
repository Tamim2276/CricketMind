from cricketmind_v3.domain.names import Player, find_players, name_list, simplify

KOHLI = Player(person_id="ba607b88", name="V Kohli", matches=400)
ROHIT = Player(person_id="740742ef", name="RG Sharma", matches=450)
ISHANT = Player(person_id="2e171977", name="I Sharma", matches=200)
DHONI = Player(person_id="4a8a2e3b", name="MS Dhoni", matches=500)
ABHISHEK = Player(person_id="aaaa0001", name="Abhishek Sharma", matches=60)
SCIVER = Player(person_id="aaaa0002", name="NR Sciver-Brunt", matches=250)
PLAYERS = [KOHLI, ROHIT, ISHANT, DHONI, ABHISHEK, SCIVER]


def names(found: list[Player]) -> list[str]:
    result = []
    for player in found:
        result.append(player.name)
    return result


# The plan's test


def test_kohli_finds_v_kohli():
    assert names(find_players("Kohli", PLAYERS)) == ["V Kohli"]


# The three kinds of match, closest first


def test_the_whole_name_however_it_is_typed():
    assert names(find_players("v kohli", PLAYERS)) == ["V Kohli"]
    assert names(find_players("V. Kohli", PLAYERS)) == ["V Kohli"]
    assert names(find_players("M.S. Dhoni", PLAYERS)) == ["MS Dhoni"]


def test_a_first_name_finds_the_matching_initial():
    assert names(find_players("Rohit Sharma", PLAYERS)) == ["RG Sharma"]
    assert names(find_players("Virat Kohli", PLAYERS)) == ["V Kohli"]


def test_a_surname_shared_by_several_lists_them_most_played_first():
    assert names(find_players("Sharma", PLAYERS)) == [
        "RG Sharma",
        "I Sharma",
        "Abhishek Sharma",
    ]


def test_the_whole_name_beats_an_initial():
    # "Abhishek Sharma" is a whole name; it must not also bring in other A. Sharmas
    assert names(find_players("Abhishek Sharma", PLAYERS)) == ["Abhishek Sharma"]


def test_an_alias_counts_as_the_whole_name():
    virat = Player(
        person_id="ba607b88", name="V Kohli", aliases=("Virat Kohli",), matches=400
    )
    assert find_players("virat kohli", [virat]) == [virat]


def test_a_hyphenated_surname():
    assert names(find_players("Sciver-Brunt", PLAYERS)) == ["NR Sciver-Brunt"]
    assert names(find_players("Brunt", PLAYERS)) == ["NR Sciver-Brunt"]


def test_an_unknown_name_finds_nobody():
    assert find_players("Bradman", PLAYERS) == []
    assert find_players("  ", PLAYERS) == []


def test_part_of_a_word_is_not_a_surname():
    assert find_players("hli", PLAYERS) == []  # not the end of "kohli"


# Helpers


def test_simplify():
    assert simplify(" M.S.  Dhoni ") == "ms dhoni"
    assert simplify("D'Arcy Short") == "darcy short"
    assert simplify("Sciver-Brunt") == "sciver brunt"


def test_the_name_list_for_c6():
    assert name_list([KOHLI, ABHISHEK]) == {
        "V Kohli": ("Kohli",),
        "Abhishek Sharma": ("Sharma",),
    }


# Found on real data (Day 33): many players are known by their first name


SHAKIB = Player(person_id="aaaa0003", name="Shakib Al Hasan", matches=300)
BABAR = Player(person_id="aaaa0004", name="Babar Azam", matches=280)


def test_a_first_name_alone():
    assert names(find_players("Shakib", PLAYERS + [SHAKIB])) == ["Shakib Al Hasan"]
    assert names(find_players("babar", PLAYERS + [BABAR])) == ["Babar Azam"]


def test_one_word_searches_surnames_and_first_names_most_played_first():
    # "Babar" means Babar Azam to most fans, though Zulfiqar Babar has it as a surname
    zulfiqar = Player(person_id="aaaa0008", name="Zulfiqar Babar", matches=10)
    assert names(find_players("Babar", [zulfiqar, BABAR])) == [
        "Babar Azam",
        "Zulfiqar Babar",
    ]


def test_initials_are_not_a_first_name():
    assert find_players("ms", PLAYERS) == []  # "MS Dhoni" starts with initials


def test_cricsheets_number_for_two_people_of_one_name_is_ignored():
    assert simplify("Imran Khan (1)") == "imran khan"
    one = Player(person_id="aaaa0006", name="Imran Khan", matches=40)
    two = Player(person_id="aaaa0007", name="Imran Khan (1)", matches=10)
    assert names(find_players("Imran Khan", [two, one])) == [
        "Imran Khan",
        "Imran Khan (1)",
    ]
