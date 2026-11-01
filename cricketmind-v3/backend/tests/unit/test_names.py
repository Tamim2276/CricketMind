from cricketmind_v3.domain.names import (
    Clues,
    Player,
    clearly_most_played,
    close_spellings,
    clue_fits,
    describe,
    find_players,
    name_list,
    prefer_by_clues,
    simplify,
)

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


# Day 58: telling players of one name apart

STEVE = Player(
    person_id="30a45b23",
    name="SPD Smith",
    matches=574,
    teams=("Australia", "Sydney Sixers"),
    formats=("ODI", "T20", "T20I", "Test"),
    first_year=2010,
    last_year=2026,
    gender="male",
)
GRAEME = Player(
    person_id="6f0ef3ea",
    name="GC Smith",
    matches=314,
    teams=("South Africa",),
    formats=("ODI", "T20", "T20I", "Test"),
    first_year=2002,
    last_year=2014,
    gender="male",
)
DWAYNE = Player(
    person_id="1c4c1b5f",
    name="DR Smith",
    matches=319,
    teams=("West Indies", "Chennai Super Kings"),
    formats=("ODI", "T20", "T20I", "Test"),
    first_year=2004,
    last_year=2019,
    gender="male",
)
SMITHS = [STEVE, DWAYNE, GRAEME]


def test_a_clue_narrows_the_namesakes():
    assert names(prefer_by_clues(SMITHS, Clues(teams=("Australia",)))) == ["SPD Smith"]
    assert names(prefer_by_clues(SMITHS, Clues(from_year=2016))) == [
        "SPD Smith",
        "DR Smith",
    ]
    both = prefer_by_clues(SMITHS, Clues(teams=("India", "Australia")))
    assert names(both) == ["SPD Smith"]  # his side or the other: either may be his


def test_a_clue_that_fits_nobody_rules_nobody_out():
    assert names(prefer_by_clues(SMITHS, Clues(teams=("Nepal",)))) == names(SMITHS)
    assert names(prefer_by_clues(SMITHS, Clues(gender="female"))) == names(SMITHS)


def test_clues_add_up():
    clues = Clues(match_format="Test", from_year=2016, to_year=2018)
    assert names(prefer_by_clues(SMITHS, clues)) == ["SPD Smith", "DR Smith"]


def test_clearly_the_most_played():
    a_kohli = Player(person_id="aaaa0003", name="A Kohli", matches=41)
    assert clearly_most_played(
        [Player(person_id="x", name="V Kohli", matches=837), a_kohli]
    )
    assert not clearly_most_played(SMITHS)  # 574 against 319
    assert clearly_most_played([STEVE])
    assert not clearly_most_played([])


def test_a_misspelt_name_offers_close_spellings():
    v_kohli = Player(person_id="ba607b88", name="V Kohli", matches=837)
    a_kohli = Player(person_id="aaaa0003", name="A Kohli", matches=41)
    found = close_spellings("Kholi", [a_kohli, v_kohli, STEVE])
    assert names(found) == ["V Kohli", "A Kohli"]  # most-played first
    assert close_spellings("Jaddu", [v_kohli, STEVE]) == []


def test_a_clue_fits_who_he_is():
    assert clue_fits("the Australian", STEVE)
    assert not clue_fits("the Australian", GRAEME)
    assert clue_fits("in 2005", GRAEME)
    assert not clue_fits("in 2020", GRAEME)  # he had stopped
    assert not clue_fits("the captain", STEVE)  # nothing to check it against
    oman = Player(
        person_id="o1", name="A Khan", teams=("Oman",), first_year=2015, last_year=2024
    )
    assert clue_fits("the Omani bowler", oman)
    assert not clue_fits("the woman who bowls", oman)  # Oman hides inside "woman"


def test_described_without_a_statistic():
    assert (
        describe(STEVE)
        == "SPD Smith; Australia, Sydney Sixers; ODI, T20, T20I, Test; 2010-2026"
    )
    assert describe(KOHLI) == "V Kohli"


def test_known_by_a_given_name_in_his_full_name():
    # the real run: "Kamindu Mendis" was unknown, his K being his third initial
    kamindu = Player(
        person_id="k1",
        name="PHKD Mendis",
        aliases=("Pasqual Handi Kamindu Dilanka Mendis",),
        matches=165,
    )
    kusal = Player(
        person_id="k2",
        name="BKG Mendis",
        aliases=("Balapuwaduge Kusal Gimhan Mendis",),
        matches=402,
    )
    assert names(find_players("Kamindu Mendis", [kamindu, kusal])) == ["PHKD Mendis"]
    assert names(find_players("Kusal Mendis", [kamindu, kusal])) == ["BKG Mendis"]
    assert find_players("Kamindu Perera", [kamindu, kusal]) == []


def test_namesakes_by_the_whole_name_do_not_hide_him():
    # two players registered as "Rohit Sharma" (7 and 5 matches) hid RG Sharma (785)
    rohit = Player(
        person_id="r1",
        name="RG Sharma",
        aliases=("Rohit Gurunath Sharma",),
        matches=785,
    )
    others = [
        Player(person_id="r2", name="Rohit Sharma", matches=7),
        Player(person_id="r3", name="Rohit Sharma", matches=5),
    ]
    found = find_players("Rohit Sharma", [*others, rohit])
    assert names(found) == ["RG Sharma", "Rohit Sharma", "Rohit Sharma"]
    assert clearly_most_played(found)


def test_a_shared_name_is_listed_by_register_id_for_c6():
    one = Player(person_id="a1a1a1a1", name="Imran Khan", matches=125)
    two = Player(person_id="b2b2b2b2", name="Imran Khan", matches=60)
    assert name_list([one, two, KOHLI]) == {
        "a1a1a1a1": ("Imran Khan", "Khan"),
        "b2b2b2b2": ("Imran Khan", "Khan"),
        "V Kohli": ("Kohli",),
    }
