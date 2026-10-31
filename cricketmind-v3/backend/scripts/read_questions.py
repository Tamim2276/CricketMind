"""N3 reads a few questions with a real model and the real player register (Day 58, by
hand). It reads the database only.

Run from backend/, with Ollama running and no CricShot training on the GPU:
    uv run python scripts/read_questions.py ../data/cricket.duckdb qwen2.5:7b
    uv run python scripts/read_questions.py ../data/cricket.duckdb qwen3:8b
"""

import dataclasses
import sys
import time

import duckdb

from cricketmind_v3.application.reader import read_question
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.names import describe, name_list
from cricketmind_v3.infrastructure.ollama_llm import OllamaLLM
from cricketmind_v3.pipeline.players import load_players

QUESTIONS = [
    "Kohli out within 6 balls in the T20 powerplay against pace?",
    "What is Babar Azam's strike rate in ODIs?",
    "Would Bumrah or Starc get Smith out sooner in Tests?",
    "Australia's Smith out within 6 balls in Tests?",
    "Is Kamindu Mendis or Kusal Mendis better against spin?",
    "Kholi out in the next over?",
    "What is LBW?",
    "India's win share at Mirpur?",
]


def main(database: str, model: str) -> None:
    db = duckdb.connect(database, read_only=True)
    players = tuple(load_players(db))
    names = cricket_gazetteer(name_list(list(players)))  # for the rules, if needed
    print(f"{len(players):,} players in the register; the model: {model}\n")
    llm = OllamaLLM(model)
    for question in QUESTIONS:
        started = time.time()
        reading = read_question(question, llm, players, names)
        print(f"{time.time() - started:5.1f}s  {question}")
        print(f"        read by {reading.by}: {filled(reading.subject)}")
        for unsure in reading.unsure:
            print(
                f"        which {unsure.written}? {unsure.why} ({len(unsure.candidates)})"
            )
            for player in unsure.candidates[:3]:
                print(f"          - {describe(player)}")
        for ask in reading.asks:
            print(f"        ask about {ask.written}: {ask.why}")
        for label, items in (
            ("unknown", reading.unknown),
            ("not in the question", reading.not_in_question),
            ("not looked up yet", reading.not_looked_up),
            ("notes", reading.notes),
        ):
            if items:
                print(f"        {label}: {list(items)}")
        print()


def filled(subject) -> dict:
    found = {}
    for field in dataclasses.fields(subject):
        value = getattr(subject, field.name)
        if value is not None:
            found[field.name] = value
    return found


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
