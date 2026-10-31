"""N3 reads a question with the language model, and never trusts it (Day 58).

The model fills a form whose fields are the subject's, with allowed values wherever the
data has them. Then plain code checks it: every name, value and number must be in the
question's own words. Each name is found by a small agentic search (resolver.py): one
player is kept; several are told apart by the question's checked clues, or asked back,
never guessed; nobody gets close spellings, offered. "The next over" and a day with no
year are asked back too. A reply that does not fit is asked again once, with the
reason; if it fails again, the rules read the question.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cricketmind_v3.application.ports import LLM, LLMError
from cricketmind_v3.application.resolver import find_player
from cricketmind_v3.domain.faith.gazetteer import BOWLER_TYPES, PHASES, SHOTS, Entry
from cricketmind_v3.domain.names import Clues, Player
from cricketmind_v3.domain.reading import (
    FORMATS,
    MONTHS,
    WOMEN,
    AskValue,
    Reading,
    appears_in,
    date_parts_said,
    next_overs,
    read_by_rules,
    says_date,
    says_digits,
    says_innings,
    says_number,
)
from cricketmind_v3.domain.subject import CricketSubject

PEOPLE = ("player", "bowler", "other_bowler")
NOT_LOOKED_UP = ("team", "opposition", "venue")  # their registers come in part 2
YEAR_OR_DAY = r"^\d{4}(-\d{2}-\d{2})?$"


class QuestionForm(BaseModel):
    """The form the model fills: one field for each thing a question can say."""

    model_config = ConfigDict(extra="forbid")

    player: str | None = Field(description="the batter, written as in the question")
    bowler: str | None = Field(
        description="the bowler bowling to the batter, or the bowler the question is "
        "about, written as in the question"
    )
    other_bowler: str | None = Field(
        description="a second bowler to compare with the first, as in 'Bumrah or Starc'"
    )
    team: str | None = Field(
        description="the team asked about, written as in the question"
    )
    opposition: str | None = Field(description="the side played against, as written")
    venue: str | None = Field(
        description="the ground or city, written as in the question"
    )
    shot: Literal[tuple(SHOTS)] | None = Field(description="the shot asked about")
    other_shot: Literal[tuple(SHOTS)] | None = Field(
        description="a second shot to compare with the first"
    )
    phase: Literal[tuple(PHASES)] | None = Field(description="the part of the innings")
    bowler_type: Literal[tuple(BOWLER_TYPES)] | None = Field(
        description="pace or spin, only if the question says so"
    )
    match_format: Literal["T20I", "T20", "ODI", "Test"] | None = Field(
        description="T20I: internationals only; T20: every twenty-over match, leagues too"
    )
    gender: Literal["male", "female"] | None = Field(
        description="only if the question says men or women"
    )
    k: int | None = Field(description="N in 'out within N balls'", ge=1)
    season: str | None = Field(
        description="'2024' or '2024/25'", pattern=r"^\d{4}(/\d{2})?$"
    )
    since: str | None = Field(
        description="from this year or day on", pattern=YEAR_OR_DAY
    )
    before: str | None = Field(
        description="up to this year or day", pattern=YEAR_OR_DAY
    )
    innings: Literal[1, 2, 3, 4] | None = Field(description="1 if batting first")
    batting_hand: Literal["left", "right"] | None = Field(
        description="left- or right-handers"
    )
    side: Literal["batting", "fielding"] | None = Field(
        description="who asked for a review"
    )
    date: str | None = Field(
        description="the match day, as '2024-04-12'", pattern=r"^\d{4}-\d{2}-\d{2}$"
    )


SYSTEM = (
    "You read cricket questions for a statistics system. Fill in the form from the "
    "question alone. Copy every name exactly as the question writes it. The player is "
    "the batter; the bowler bowls to him, or is the bowler the question is about. In "
    "'Would Bumrah or Starc get Smith out sooner?', Smith is the player, Bumrah the "
    "bowler and Starc the other bowler. Use null for anything the question does not "
    "say. Never guess, and never add a name or a number."
)

# the words that say each allowed value; the model's value must be among the question's
SAID_BY = {
    "phase": PHASES,
    "bowler_type": BOWLER_TYPES,
    "shot": SHOTS,
    "other_shot": SHOTS,
    "gender": {"female": WOMEN, "male": ("men", "mens", "male")},
    "batting_hand": {"left": ("left",), "right": ("right",)},
    "side": {"batting": ("batting",), "fielding": ("fielding",)},
}


def read_question(
    question: str,
    llm: LLM | None,
    players: tuple[Player, ...],
    names: dict[str, list[Entry]],
) -> Reading:
    """The model reads, the code checks; without a model, or when its reply does not
    fit twice, the rules read (Day 56)."""
    if llm is None:
        return Reading(subject=read_by_rules(question, names), by="rules")
    try:
        form = _ask(llm, question)
    except LLMError as error:
        reason = str(error).splitlines()[0]
        return Reading(
            subject=read_by_rules(question, names),
            by="rules",
            notes=(f"read by rules: the model's reply did not fit twice ({reason})",),
        )
    return _check(form, question, players, llm)


def _ask(llm: LLM, question: str) -> QuestionForm:
    try:
        return llm.answer(SYSTEM, question, QuestionForm)
    except LLMError as error:
        again = (
            f"{question}\n\nYour last answer did not fit the form: {error}\n"
            "Answer again."
        )
        return llm.answer(SYSTEM, again, QuestionForm)


def _check(
    form: QuestionForm, question: str, players: tuple[Player, ...], llm: LLM
) -> Reading:
    fields = {}
    unknown = []
    unsure = []
    not_in_question = []
    not_looked_up = []
    notes = []
    asks = []

    # first the values, so that only checked ones can tell players apart
    for name, said_by in SAID_BY.items():
        value = getattr(form, name)
        if value is None:
            continue
        if _says(value, said_by, question):
            fields[name] = value
        else:
            not_in_question.append(f"{name}={value}")  # a filter nobody asked for

    if form.match_format is not None:
        if _says_format(form.match_format, question):
            fields["match_format"] = form.match_format
        else:
            not_in_question.append(f"match_format={form.match_format}")

    for name in ("k", "innings", "season", "since", "before", "date"):
        value = getattr(form, name)
        if value is None:
            continue
        if _says_number(name, value, question):
            fields[name] = value
        elif name == "date" and _day_without_year(value, question):
            asks.append(
                AskValue(field="date", written=_day_said(value), why="which year?")
            )
        else:
            not_in_question.append(f"{name}={value}")  # 12 balls when it said 6

    overs = next_overs(question)
    if overs is not None and "k" not in fields:
        balls = 6 * overs
        asks.append(
            AskValue(
                field="k",
                written=_overs_said(overs),
                why=f"an over is six balls, but he may face fewer: his next {balls} balls?",
                suggestion=str(balls),
            )
        )

    for name in NOT_LOOKED_UP:
        written = getattr(form, name)
        if written is None:
            continue
        if appears_in(written, question):
            not_looked_up.append((name, written))
        else:
            not_in_question.append(written)

    # a team or ground with no gender named is men's cricket, and the reading says so
    if not_looked_up and "gender" not in fields:
        fields["gender"] = "male"
        notes.append(
            "read as men's cricket: a team or ground, and no gender named (D-046)"
        )

    # then the people, each found by a small agentic search (resolver.py)
    clues = _clues(fields, not_looked_up)
    for name in PEOPLE:
        written = getattr(form, name)
        if written is None:
            continue
        if not appears_in(written, question):
            not_in_question.append(written)  # "Stern" for "Starc"
            continue
        found = find_player(name, written, question, clues, players, llm)
        if found.player is not None:
            fields[name] = found.player.name
            if found.note is not None:
                notes.append(found.note)
        elif found.unsure is not None:
            unsure.append(found.unsure)
        else:
            unknown.append(written)

    return Reading(
        subject=CricketSubject(**fields),
        by="model",
        unknown=tuple(unknown),
        unsure=tuple(unsure),
        not_in_question=tuple(not_in_question),
        not_looked_up=tuple(not_looked_up),
        notes=tuple(notes),
        asks=tuple(asks),
    )


def _clues(fields: dict, not_looked_up: list[tuple[str, str]]) -> Clues:
    # only what the checks kept: his side or the other, the format, the gender, the years
    teams = []
    for name, written in not_looked_up:
        if name in ("team", "opposition"):
            teams.append(written)
    from_year = None
    to_year = None
    if "season" in fields:
        from_year = int(fields["season"][:4])
        to_year = from_year
        if "/" in fields["season"]:
            to_year = from_year + 1
    if "since" in fields:
        from_year = int(fields["since"][:4])
    if "before" in fields:
        to_year = int(fields["before"][:4])
    return Clues(
        teams=tuple(teams),
        match_format=fields.get("match_format"),
        gender=fields.get("gender"),
        from_year=from_year,
        to_year=to_year,
    )


def _day_without_year(value: str, question: str) -> bool:
    year, month, day = date_parts_said(value, question)
    return month and day and not year


def _day_said(value: str) -> str:
    # "2024-04-12" -> "12 April"
    return f"{int(value[8:10])} {MONTHS[int(value[5:7]) - 1].title()}"


def _overs_said(overs: int) -> str:
    if overs == 1:
        return "the next over"
    return f"the next {overs} overs"


def _says(value: str, said_by: dict, question: str) -> bool:
    # the value itself, or one of its other names: "seamers" says pace
    if appears_in(value, question):
        return True
    for other in said_by.get(value, ()):
        if appears_in(other, question):
            return True
    return False


def _says_format(value: str, question: str) -> bool:
    # a word the rules read as this format: "Tests" says Test
    for written, meant in FORMATS.items():
        if meant == value and appears_in(written, question):
            return True
    return False


def _says_number(name: str, value, question: str) -> bool:
    # each kind of number, said its own way
    if name == "k":
        return says_number(value, question)
    if name == "innings":
        return says_innings(value, question)
    if name == "season":
        return says_digits(value, question)
    if name == "date":
        return says_date(value, question)
    if len(value) == 4:  # since or before: a year...
        return says_digits(value, question)
    return says_date(value, question)  # ...or a day
