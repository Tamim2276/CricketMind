"""Which player a name means: a small agentic search (Day 58, part 2a; D-055).

Look the name up. If several players have it, the question's clues narrow them (his side,
the format, the years). If one has played far more than the rest, he is kept, and the
reading says so. Otherwise the model may search again with a fuller name from the
question, or point to the words in it that show which one; nothing it says is taken
unless the code can check it. At most three look-ups, and the model never sees a
statistic, only who each player is. Still unsure: the user is asked. A name nobody has
gets close spellings: offered, never taken.
"""

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from cricketmind_v3.application.ports import LLM, LLMError
from cricketmind_v3.domain.names import (
    Clues,
    Player,
    clearly_most_played,
    close_spellings,
    clue_fits,
    describe,
    find_players,
    prefer_by_clues,
)
from cricketmind_v3.domain.reading import Unsure, appears_in, words

LOOK_UPS = 3  # D-055
SHOWN = 10  # at most this many players are shown to the model


@dataclass(frozen=True, kw_only=True)
class Found:
    """What the search ended with: one player, a question back, or nobody."""

    player: Player | None = None
    unsure: Unsure | None = None
    note: str | None = None  # how he was told apart, for the read-back
    look_ups: int = 0


class NameStep(BaseModel):
    """What the model may do next."""

    model_config = ConfigDict(extra="forbid")

    search: str | None = Field(
        description="a fuller name for this person, copied word for word from the question"
    )
    choose: str | None = Field(
        description="the id of the one player the question means"
    )
    clue: str | None = Field(
        description="the words of the question that show which one, copied word for word"
    )


SYSTEM = (
    "You help find which cricketer a question means. You see the question, the name it "
    "uses, and the players who have that name: their sides, formats and years. If the "
    "question has a fuller name for this person, put it in search. If words in the "
    "question show which player is meant, put his id in choose and copy those words into "
    "clue. Otherwise leave all three null, and the user will be asked. Never guess."
)


def find_player(
    field: str,
    written: str,
    question: str,
    clues: Clues,
    players: tuple[Player, ...],
    llm: LLM | None,
) -> Found:
    query = written
    kept = []
    look_ups = 0
    while look_ups < LOOK_UPS:
        look_ups += 1
        candidates = find_players(query, list(players))
        if not candidates:
            break  # nobody, or a fuller name that found nobody: keep what we had
        kept = prefer_by_clues(candidates, clues)
        if len(kept) == 1:
            return Found(
                player=kept[0], note=_why_one(written, candidates), look_ups=look_ups
            )
        if clearly_most_played(kept):
            note = (
                f"{written} read as {kept[0].name}: by far the most-played of "
                f"{len(kept)} players called {query}"
            )
            return Found(player=kept[0], note=note, look_ups=look_ups)
        step = _ask(llm, question, query, kept)
        if step is None:
            break
        if _fuller(step.search, query, question):
            query = step.search  # search again: the agentic part
            continue
        chosen = _checked(step, kept, question)
        if chosen is not None:
            note = f'{written} read as {chosen.name}: "{step.clue}" in the question'
            return Found(player=chosen, note=note, look_ups=look_ups)
        break

    if kept:
        return Found(
            unsure=Unsure(field=field, written=written, candidates=tuple(kept)),
            look_ups=look_ups,
        )
    close = close_spellings(written, list(players))
    if close:
        unsure = Unsure(
            field=field,
            written=written,
            candidates=tuple(close),
            why="nobody has this name; did you mean one of these?",
        )
        return Found(unsure=unsure, look_ups=look_ups)
    return Found(look_ups=look_ups)  # unknown


def _why_one(written: str, candidates: list[Player]) -> str | None:
    if len(candidates) == 1:
        return None  # only one player has the name: nothing to explain
    return (
        f"{written} read as the only one of {len(candidates)} the question's clues fit"
    )


def _ask(
    llm: LLM | None, question: str, query: str, kept: list[Player]
) -> NameStep | None:
    if llm is None:
        return None
    lines = [f"Question: {question}", f'The name "{query}" belongs to these players:']
    for player in kept[:SHOWN]:
        lines.append(f"- id {player.person_id}: {describe(player)}")
    try:
        return llm.answer(SYSTEM, "\n".join(lines), NameStep)
    except LLMError:
        return None  # no help from the model: the user is asked


def _fuller(search: str | None, query: str, question: str) -> bool:
    # a longer name, and in the question word for word: "Kamindu Mendis" for "Mendis"
    if search is None:
        return False
    return len(words(search)) > len(words(query)) and appears_in(search, question)


def _checked(step: NameStep, kept: list[Player], question: str) -> Player | None:
    """The model's pick, only if its clue is in the question and fits him alone."""
    if step.choose is None or step.clue is None or not appears_in(step.clue, question):
        return None
    chosen = None
    fitting = 0
    for player in kept[:SHOWN]:
        if clue_fits(step.clue, player):
            fitting += 1
        if player.person_id == step.choose:
            chosen = player
    if chosen is None or fitting != 1 or not clue_fits(step.clue, chosen):
        return None
    return chosen
