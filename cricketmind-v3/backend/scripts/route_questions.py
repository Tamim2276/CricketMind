"""N3 and N10 with a real model, on the real database, to see the router work (Day 64, by
hand). It reads the database only, and scores nothing: Days 170-173 measure the routers.

Run from backend/, with Ollama running and no CricShot training on the GPU:
    uv run python scripts/route_questions.py ../data/cricket.duckdb qwen2.5:7b
"""

import sys
import time

import duckdb

from cricketmind_v3.application.graph.build import build_graph
from cricketmind_v3.application.graph.context import GraphContext
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.names import name_list
from cricketmind_v3.domain.routing import BY_KEYWORDS, decide, propose_by_keywords
from cricketmind_v3.infrastructure.duckdb_engines import engines_on
from cricketmind_v3.infrastructure.duckdb_repo import DuckDBDeliveryRepository
from cricketmind_v3.infrastructure.langchain_llm import LangChainLLM
from cricketmind_v3.pipeline.grounds import load_grounds
from cricketmind_v3.pipeline.players import load_players
from cricketmind_v3.pipeline.teams import load_teams

QUESTIONS = [
    "Kohli's strike rate in the T20 powerplay?",
    "Should I pick Kohli as my fantasy captain in T20s?",
    "How should we set up to get Kohli out early in T20s?",
    "Is it worth our franchise keeping Kohli for another T20 season?",
    "How can Kohli score faster against spin in T20s?",
    "Would Kohli suit a T20 side we are looking at recruiting for?",
    "Give me a line for the TV broadcast about Kohli in T20s.",
    "What should young batters learn from Kohli in T20s?",
    "What should Kohli's manager tell a team about his T20 record?",
    "Has Bumrah been bowled too much up to 4 January 2025?",
    "Bumrah's economy in T20s?",
    "How often do India's men win a review?",
    "What is LBW?",
]


def main(database: str, model: str) -> None:
    db = duckdb.connect(database, read_only=True)
    players = tuple(load_players(db))
    context = GraphContext(
        names=cricket_gazetteer(name_list(list(players))),
        llm=LangChainLLM(model),  # the door that stayed (D-024)
        players=players,
        grounds=tuple(load_grounds(db)),
        teams=tuple(load_teams(db)),
        deliveries=DuckDBDeliveryRepository(db.cursor()),
        engines=engines_on(db.cursor()),
    )
    graph = build_graph()
    for question in QUESTIONS:
        started = time.time()
        state = graph.invoke({"question": question, "clip": None}, context=context)
        print(f"{time.time() - started:5.1f}s  {question}")
        routing = state.get("routing")
        if routing is None:
            print("        no router: the question went straight to the answer")
            continue
        words = decide(propose_by_keywords(question), state["capable"], BY_KEYWORDS)
        print(f"        can answer: {', '.join(state['capable']) or 'nobody'}")
        print(
            f"        {routing.by}: {', '.join(routing.chosen) or 'nobody'} ({routing.how})"
        )
        print(f"        words would choose: {', '.join(words.chosen) or 'nobody'}")
        if routing.no_number:
            print(f"        wanted, no number: {', '.join(routing.no_number)}")
        if routing.reason:
            print(f"        its reason: {routing.reason}")
        if routing.note:
            print(f"        note: {routing.note}")
    db.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
