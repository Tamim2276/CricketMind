"""A small made-up world for running the whole graph with fakes (Days 73-79).

Every number here is invented, and the engines and the model are fakes: these tests say
how the graph behaves, and nothing about any real player."""

import json

from cricketmind_v3.application.graph.build import build_graph
from cricketmind_v3.application.graph.context import GraphContext
from cricketmind_v3.application.ports import FakeFactSource, FakeLLMByWho
from cricketmind_v3.application.reader import QuestionForm
from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.names import Player
from cricketmind_v3.domain.receipts import Receipt
from cricketmind_v3.domain.subject import CricketSubject
from cricketmind_v3.infrastructure.prompt_files import read_rules, read_specs

SPECS = read_specs()
RULES = read_rules()
READER = "You read cricket questions"  # words in each prompt, to file a fake's replies
ROUTER = "You send cricket questions"
REPORTER = "You join specialists"

KOHLI_PLAYER = Player(person_id="ba607b88", name="V Kohli", matches=600)
BUMRAH_PLAYER = Player(person_id="462411b3", name="JJ Bumrah", matches=400)
NAMES = cricket_gazetteer(
    {
        "V Kohli": ("Kohli", "Virat Kohli"),
        "RG Sharma": ("Rohit", "Rohit Sharma"),
        "JJ Bumrah": ("Bumrah",),
    }
)

QUESTION = "Kohli's strike rate in T20s?"
KOHLI = CricketSubject(player="V Kohli", match_format="T20")
RATE = Fact(value=135.1, n=9841)
ABOVE = Fact(value=12.0, n=9841, interval=(8.0, 16.0))
LEVEL = Fact(value=1.0, n=9841, interval=(-2.0, 4.0))
BELOW = Fact(value=-12.0, n=9841, interval=(-16.0, -8.0))
THIN = Fact(value=-40.0, n=6, interval=(-90.0, -5.0))


def receipted(metric: str, subject: CricketSubject, fact: Fact, number: int) -> Fact:
    # as the real engines do: a number comes with where it came from
    receipt = Receipt(
        data="a1b2c3d4e5f6a7b8",
        engine="N8.M1",
        version="1",
        query=f'{{"metric":"{metric}","number":{number}}}',
        value=fact.value,
        n=fact.n,
        interval=fact.interval,
    )
    return Fact(value=fact.value, n=fact.n, interval=fact.interval, receipt=receipt)


def engines(*extra) -> FakeFactSource:
    """The engines of this world: Kohli's strike rate, and any (metric, subject, fact)."""
    source = FakeFactSource()
    source.add("strike_rate", KOHLI, receipted("strike_rate", KOHLI, RATE, 1))
    for number, (metric, subject, fact) in enumerate(extra, start=2):
        source.add(metric, subject, receipted(metric, subject, fact, number))
    return source


def reading(**fields) -> str:
    """What the fake model replies when N3 asks: every box empty but these."""
    reply = dict.fromkeys(QuestionForm.model_fields)
    reply.update(kind="stats", player="Kohli", match_format="T20")
    reply.update(fields)
    return json.dumps(reply)


def routed(*agents: str) -> str:
    """What the fake model replies when N10 asks."""
    return json.dumps({"reason": "a test", "agents": list(agents)})


def words(template: str) -> str:
    """What the fake model replies when an agent or N12 asks."""
    return json.dumps({"template": template})


def model(
    agents: dict[str, list],
    chosen: tuple[str, ...] = ("N11b", "N11c"),
    report: list | None = None,
    read: str | None = None,
) -> FakeLLMByWho:
    """A fake model with replies for N3, N10, N12 and each agent, by the agent's name."""
    replies = {READER: [read or reading()], ROUTER: [routed(*chosen)]}
    replies[REPORTER] = report if report is not None else []
    replies.update(agents)
    return FakeLLMByWho(replies)


def context_of(llm, source=None) -> GraphContext:
    if source is None:
        source = engines()
    return GraphContext(
        names=NAMES,
        llm=llm,
        players=(KOHLI_PLAYER, BUMRAH_PLAYER),
        engines=source,
        specs=SPECS,
        rules=RULES,
    )


def ask(llm, source=None, question: str = QUESTION) -> dict:
    """The whole graph, once."""
    return build_graph().invoke(
        {"question": question, "clip": None}, context=context_of(llm, source)
    )


def asked_of(llm, who: str) -> list[tuple[str, str, str]]:
    """Every call the model had that was for this one."""
    found = []
    for call in llm.asked:
        if who in call[0]:
            found.append(call)
    return found
