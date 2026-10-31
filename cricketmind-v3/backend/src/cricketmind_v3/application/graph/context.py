"""What the nodes are given, not what they write (Day 56): the names N3 knows, the
language model and the player register (Day 58); the engines later. It is passed in when
the graph runs, so a test can give fakes."""

from dataclasses import dataclass, field

from cricketmind_v3.application.ports import LLM
from cricketmind_v3.domain.faith.gazetteer import Entry
from cricketmind_v3.domain.names import Player


@dataclass(frozen=True, kw_only=True)
class GraphContext:
    names: dict[str, list[Entry]] = field(default_factory=dict)  # cricket_gazetteer
    llm: LLM | None = None  # without one, N3 reads by rules
    players: tuple[Player, ...] = ()  # the register, for the model's names
