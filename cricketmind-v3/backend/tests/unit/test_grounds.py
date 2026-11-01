"""Day 58, part 2b: one ground, one name (L-028), and which ground a question means.
Every case is one the real data has."""

from cricketmind_v3.domain.grounds import (
    Ground,
    describe_ground,
    find_grounds,
    places_of,
    same_grounds,
)


def merged(rows):
    """The groups of names that became one ground, each sorted."""
    groups = {}
    for key, main in same_grounds(rows).items():
        if main not in groups:
            groups[main] = []
        groups[main].append(key)
    found = []
    for main, keys in groups.items():
        if len(keys) > 1:
            found.append((main, sorted(keys)))
    return sorted(found)


def test_a_renamed_city_and_a_dot_are_one_ground():
    rows = [
        ("M Chinnaswamy Stadium, Bangalore", ("Bangalore",), 92),
        ("M Chinnaswamy Stadium, Bengaluru", ("Bengaluru",), 61),
        ("M.Chinnaswamy Stadium", ("Bengaluru",), 20),
    ]
    assert merged(rows) == [
        (
            "M Chinnaswamy Stadium, Bangalore",  # the most matches: the main name
            [
                "M Chinnaswamy Stadium, Bangalore",
                "M Chinnaswamy Stadium, Bengaluru",
                "M.Chinnaswamy Stadium",
            ],
        )
    ]


def test_the_town_and_the_island_swap_places():
    rows = [
        ("Kensington Oval, Bridgetown", ("Barbados",), 76),
        ("Kensington Oval, Bridgetown, Barbados", ("Bridgetown",), 68),
        ("Kensington Oval, Barbados", ("Bridgetown",), 8),
    ]
    assert len(merged(rows)[0][1]) == 3


def test_one_name_in_two_places_is_two_grounds():
    rows = [
        ("Lord's", ("London",), 312),
        ("Lords", ("Bermuda",), 3),  # Bermuda's Lords, not London's
        ("County Ground, Taunton", ("Taunton",), 79),
        ("County Ground, Bristol", ("Bristol",), 235),
        ("ACA Stadium, Mangalagiri", (), 7),
        ("ACA Stadium, Barsapara", (), 10),  # Guwahati, not Andhra
        ("Airforce Complex ground, Palam", (), 21),
        ("Airforce Complex ground, Palam II", (), 14),  # the second ground
    ]
    assert merged(rows) == []


def test_a_name_with_no_place_joins_nothing_by_rule():
    rows = [("Nehru Stadium, ?", (), 4), ("Nehru Stadium, Kochi", ("Kochi",), 9)]
    assert merged(rows) == []


def test_the_ones_checked_by_hand():
    rows = [
        ("Arun Jaitley Stadium", ("Delhi",), 104),
        ("Feroz Shah Kotla", ("Delhi",), 84),  # renamed in 2019
        ("Shere Bangla National Stadium", ("Dhaka", "Mirpur"), 496),
        ("Sher-e-Bangla National Cricket Stadium", ("Mirpur",), 1),
        ("Narendra Modi Stadium", ("Ahmedabad",), 69),
        ("Sardar Patel Stadium", ("Ahmedabad",), 29),  # rebuilt: kept apart
    ]
    assert merged(rows) == [
        ("Arun Jaitley Stadium", ["Arun Jaitley Stadium", "Feroz Shah Kotla"]),
        (
            "Shere Bangla National Stadium",
            ["Sher-e-Bangla National Cricket Stadium", "Shere Bangla National Stadium"],
        ),
    ]


def test_places_renamed_and_squashed():
    assert places_of("St George's Park, Port Elizabeth", ()) == {"gqeberha"}
    assert places_of("Kensington Oval, Bridgetown", ("Barbados",)) == {
        "bridgetown",
        "barbados",
    }


GROUNDS = [
    Ground(name="Wankhede Stadium", places=("Mumbai",), matches=213),
    Ground(name="Dr DY Patil Sports Academy, Mumbai", places=("Mumbai",), matches=73),
    Ground(
        name="M Chinnaswamy Stadium, Bangalore",
        other_names=("M.Chinnaswamy Stadium",),
        places=("Bangalore", "Bengaluru"),
        matches=173,
    ),
    Ground(name="Lord's", places=("London",), matches=312),
    Ground(name="Lords", places=("Bermuda",), matches=3),
]


def names(found):
    result = []
    for ground in found:
        result.append(ground.name)
    return result


def test_by_name_before_by_place():
    assert names(find_grounds("Chinnaswamy", GROUNDS)) == [
        "M Chinnaswamy Stadium, Bangalore"
    ]
    assert names(find_grounds("Lord's", GROUNDS)) == ["Lord's", "Lords"]


def test_a_place_after_the_comma_is_a_place_not_a_name():
    # "Mumbai" in "Dr DY Patil Sports Academy, Mumbai" must not hide Wankhede
    assert names(find_grounds("Mumbai", GROUNDS)) == [
        "Wankhede Stadium",
        "Dr DY Patil Sports Academy, Mumbai",
    ]
    assert names(find_grounds("Bengaluru", GROUNDS)) == [
        "M Chinnaswamy Stadium, Bangalore"
    ]


def test_nothing_found():
    assert find_grounds("Hogwarts", GROUNDS) == []
    assert find_grounds("", GROUNDS) == []


def test_described_without_a_statistic():
    ground = Ground(
        name="Lord's", places=("London",), matches=312, first_year=2005, last_year=2026
    )
    assert describe_ground(ground) == "Lord's; London; 2005-2026"


def test_a_name_before_a_place():
    # "Melbourne" names the MCG, and is only the place of the Junction Oval
    mcg = Ground(name="Melbourne Cricket Ground", places=("Melbourne",), matches=192)
    junction = Ground(name="Junction Oval", places=("Melbourne",), matches=40)
    assert names(find_grounds("Melbourne", [junction, mcg])) == [
        "Melbourne Cricket Ground"
    ]
