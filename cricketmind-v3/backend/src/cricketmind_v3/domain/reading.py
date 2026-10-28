"""Reading a question by simple rules (Day 56): only words the system already knows.

A language model reads better from Day 58; these rules stay as the plain fallback. What
they cannot read stays empty: a name two players share (never a guess; Day 58 asks
which), "six" written as a word, a second batter.
"""

from cricketmind_v3.domain.faith.gazetteer import Entry
from cricketmind_v3.domain.faith.text import find_phrases, tokens
from cricketmind_v3.domain.subject import CricketSubject

# the format names of D-047, as people type them
FORMATS = {
    "t20i": "T20I",
    "t20is": "T20I",
    "t20": "T20",
    "t20s": "T20",
    "odi": "ODI",
    "odis": "ODI",
    "test": "Test",
    "tests": "Test",
}
BALLS = ("ball", "balls", "delivery", "deliveries")
AGAINST = ("against", "vs", "v", "versus", "facing")  # the name after it bowls
SAME_SIDE = ("or", "and")  # "against Bumrah or Starc": Starc bowls too
WOMEN = ("women", "womens")


def read_by_rules(question: str, names: dict[str, list[Entry]]) -> CricketSubject:
    """The subject of a question, from the names in a gazetteer (cricket_gazetteer) and
    a few rules: "Kohli out within 6 balls in the T20 powerplay against pace?"."""
    words = tokens(question)
    fields = {}
    side = "batting"
    for phrase, start in find_phrases(words, names):
        entries = names[phrase]
        if len(entries) != 1:
            continue  # a name two players share: never a guess
        entry = entries[0]
        before = ""
        if start > 0:
            before = words[start - 1]
        if "player" in entry.fields:
            if before in AGAINST:
                side = "bowling"
            elif before not in SAME_SIDE:
                side = "batting"
            if side == "bowling":
                _fill(fields, ("bowler", "other_bowler"), entry.value)
            else:
                _fill(fields, ("player",), entry.value)
        elif "shot" in entry.fields:
            _fill(fields, ("shot", "other_shot"), entry.value)
        else:
            _fill(fields, entry.fields, entry.value)

    for i in range(len(words) - 1):
        if words[i].isdigit() and words[i + 1] in BALLS:
            fields["k"] = int(words[i])  # "6 balls"; "6 overs" is not k
            break
    for word in words:
        if word in FORMATS:
            fields["match_format"] = FORMATS[word]
            break
    for word in words:
        if word in WOMEN:
            fields["gender"] = "female"
    return CricketSubject(**fields)


def _fill(fields: dict, names: tuple[str, ...], value: str) -> None:
    # the first empty field of these; a name said twice fills one
    if value in fields.values():
        return
    for name in names:
        if name not in fields:
            fields[name] = value
            return
