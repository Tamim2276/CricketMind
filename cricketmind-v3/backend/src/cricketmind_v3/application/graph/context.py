"""What the nodes are given, not what they write (Day 56): the names N3 knows, the
language model and the player register (Day 58); the balls and the engines (Days 60, 61).
It is passed in when the graph runs, so a test can give fakes."""

from dataclasses import dataclass, field

from cricketmind_v3.application.ports import LLM, DeliveryRepository, FactSource
from cricketmind_v3.domain.faith.gazetteer import Entry
from cricketmind_v3.domain.grounds import Ground
from cricketmind_v3.domain.names import Player
from cricketmind_v3.domain.teams import Team


@dataclass(frozen=True, kw_only=True)
class GraphContext:
    names: dict[str, list[Entry]] = field(default_factory=dict)  # cricket_gazetteer
    llm: LLM | None = None  # without one, N3 reads by rules
    players: tuple[Player, ...] = ()  # the register, for the model's names
    grounds: tuple[Ground, ...] = ()  # one name per ground (Day 58, part 2b)
    teams: tuple[Team, ...] = ()  # the team register (Day 58, part 2c)
    deliveries: DeliveryRepository | None = None  # the balls, for N7a (Day 60)
    engines: FactSource | None = None  # N8 and N9 (Day 61)
