"""N7a: the balls each side of a question is about (Day 60)."""

from cricketmind_v3.application.ports import DeliveryRepository, FilterError
from cricketmind_v3.domain.balls import tally
from cricketmind_v3.domain.evidence import Evidence, sides
from cricketmind_v3.domain.registry import is_empty
from cricketmind_v3.domain.subject import CricketSubject

# why a side has no balls
WORDS_ONLY = "a rules question is answered in words, from no balls"
NO_PERSON = "no player or bowler named: grounds and teams have their own records"
NO_DATABASE = "no ball database was given"
NO_BALLS = "no ball fits every filter"


def retrieve(
    subject: CricketSubject, kind: str, repo: DeliveryRepository | None
) -> tuple[Evidence, ...]:
    """The evidence for every side of the question, each filed under its own subject."""
    found = []
    for side in sides(subject):
        found.append(_one_side(side, kind, repo))
    return tuple(found)


def _one_side(
    side: CricketSubject, kind: str, repo: DeliveryRepository | None
) -> Evidence:
    if kind == "rules":
        return Evidence(subject=side, none_because=WORDS_ONLY)
    # without a person, every ball of a ground or a team would come back: millions
    if is_empty(side.player) and is_empty(side.bowler):
        return Evidence(subject=side, none_because=NO_PERSON)
    if repo is None:
        return Evidence(subject=side, none_because=NO_DATABASE)
    try:
        balls = repo.deliveries(side)
    except FilterError as error:
        # "cannot pick balls by shot yet": never a filter quietly left out
        return Evidence(subject=side, none_because=str(error))
    if not balls:
        return Evidence(subject=side, none_because=NO_BALLS)
    return Evidence(subject=side, count=len(balls), balls=tally(balls))
