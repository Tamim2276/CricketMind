"""Reading a question (Days 56 and 58).

By simple rules (Day 56): only words the system already knows. What they cannot read
stays empty: a name two players share (never a guess), "six" written as a word, a second
batter. A language model reads better (Day 58); the rules stay as the plain fallback.

What a reading is (Day 58), and the checks that keep a model's reading honest: every
name and every number it gives must be in the question's own words.
"""

from dataclasses import dataclass

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


# what a reading is (Day 58)


@dataclass(frozen=True, kw_only=True)
class Unsure:
    """A name the question uses that several players have: N3 asks which one."""

    field: str  # "player"
    written: str  # "Mendis", as the question has it
    candidates: tuple  # players, or grounds (Day 58, part 2b): most-played first
    why: str = "several players have this name"  # or "nobody has it; close spellings"


@dataclass(frozen=True, kw_only=True)
class AskValue:
    """Something the question says that could mean more than one thing: asked back."""

    field: str  # "k"
    written: str  # "the next over"
    why: str  # "an over is six balls, but he may face fewer of them"
    suggestion: str | None = None  # "6"


@dataclass(frozen=True, kw_only=True)
class Reading:
    """What N3 read: the subject, and everything it could not put in it."""

    subject: CricketSubject
    by: str  # "model" or "rules"
    kind: str  # one of the five (Day 59): "stats", "prediction", "video", ...
    unknown: tuple[str, ...] = ()  # names nobody in the register has
    unsure: tuple[Unsure, ...] = ()  # names several players share
    not_in_question: tuple[str, ...] = ()  # what the model gave that the question lacks
    notes: tuple[str, ...] = ()  # "read as men's cricket: no gender named"
    asks: tuple[AskValue, ...] = ()  # "the next over": his next 6 balls?
    # a register id in the subject -> who he is, for the read-back (Day 59)
    shown: tuple[tuple[str, str], ...] = ()


# the checks on a model's reading

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
}
TENS = ("twenty", "thirty")  # "twenty four"
INNINGS_WORDS = {
    1: ("1", "1st", "first"),
    2: ("2", "2nd", "second", "chasing", "chase"),
    3: ("3", "3rd", "third"),
    4: ("4", "4th", "fourth"),
}


def words(text: str) -> list[str]:
    """The text's words, dots gone first: "M.S. Dhoni's" -> ms dhoni."""
    return tokens(text.replace(".", ""))


def appears_in(written: str, question: str) -> bool:
    """The name, word for word, somewhere in the question: "Stern" is not in "Starc"."""
    wanted = words(written)
    said = words(question)
    if not wanted:
        return False
    for start in range(len(said) - len(wanted) + 1):
        if said[start : start + len(wanted)] == wanted:
            return True
    return False


def says_number(value: int, question: str) -> bool:
    """The question says this number, in digits or in words: "6", "six", "twenty four"."""
    said = words(question)
    for i, word in enumerate(said):
        if word == str(value) or NUMBER_WORDS.get(word) == value:
            return True
        if word in TENS and i + 1 < len(said):
            ones = NUMBER_WORDS.get(said[i + 1], 0)
            if 1 <= ones <= 9 and NUMBER_WORDS[word] + ones == value:
                return True
    return False


def says_digits(value: str, question: str) -> bool:
    """Every run of digits in the value is in the question: "2024/25" needs 2024 and 25."""
    said = words(question)
    found_any = False
    for part in words(value):
        if part.isdigit():
            found_any = True
            if part not in said:
                return False
    return found_any


def says_innings(value: int, question: str) -> bool:
    """The question says which innings: "first", "2nd", "chasing"."""
    for word in words(question):
        if word in INNINGS_WORDS.get(value, ()):
            return True
    return False


MONTHS = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)


def says_date(value: str, question: str) -> bool:
    """The question says this day: "2024-04-12" needs 2024, 12, and April or 4."""
    year, month, day = date_parts_said(value, question)
    return year and month and day


def date_parts_said(value: str, question: str) -> tuple[bool, bool, bool]:
    """Which of the day's year, month and day the question says."""
    parts = value.split("-")
    if len(parts) != 3 or not (parts[0] + parts[1] + parts[2]).isdigit():
        return False, False, False
    year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
    said = words(question)
    has_year = str(year) in said
    has_day = False
    has_month = False
    for word in said:
        if word.isdigit() and int(word) == day:
            has_day = True
        if word.isdigit() and int(word) == month:
            has_month = True
        if 1 <= month <= 12 and word in (MONTHS[month - 1], MONTHS[month - 1][:3]):
            has_month = True
    return has_year, has_month, has_day


def next_overs(question: str) -> int | None:
    """How many overs "the next over" or "the next two overs" says; asked back as balls,
    since he may face fewer than six of an over's balls."""
    said = words(question)
    for i, word in enumerate(said):
        if word not in ("over", "overs"):
            continue
        if i >= 1 and said[i - 1] == "next":
            return 1
        if i >= 2 and said[i - 2] == "next":
            count = said[i - 1]
            if count.isdigit():
                return int(count)
            if count in NUMBER_WORDS:
                return NUMBER_WORDS[count]
    return None
