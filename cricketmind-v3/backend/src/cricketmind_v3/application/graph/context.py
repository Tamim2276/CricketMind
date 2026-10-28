"""What the nodes are given, not what they write (Day 56): the names N3 knows today; the
language model and the engines later. It is passed in when the graph runs, so a test can
give fakes."""

from dataclasses import dataclass, field

from cricketmind_v3.domain.faith.gazetteer import Entry


@dataclass(frozen=True, kw_only=True)
class GraphContext:
    names: dict[str, list[Entry]] = field(default_factory=dict)  # cricket_gazetteer
