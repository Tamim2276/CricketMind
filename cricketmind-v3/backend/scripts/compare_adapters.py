"""The same five questions through both adapters, with the same model (Day 58, by hand).
A first look before Day 59 scores them on 90 wordings. It reads the database only.

Run from backend/, with Ollama running and no CricShot training on the GPU:
    uv run python scripts/compare_adapters.py ../data/cricket.duckdb qwen2.5:7b
"""

import dataclasses
import sys
import time

import duckdb

from cricketmind_v3.application.reader import read_question
from cricketmind_v3.domain.faith.gazetteer import cricket_gazetteer
from cricketmind_v3.domain.names import name_list
from cricketmind_v3.infrastructure.langchain_llm import LangChainLLM
from cricketmind_v3.infrastructure.ollama_llm import OllamaLLM
from cricketmind_v3.pipeline.grounds import load_grounds
from cricketmind_v3.pipeline.players import load_players
from cricketmind_v3.pipeline.teams import load_teams

QUESTIONS = [
    "Kohli out within 6 balls in the T20 powerplay against pace?",
    "Australia's Smith out within 6 balls in Tests?",
    "Is Kamindu Mendis better against spin since 2022?",
    "India's win share at Mirpur?",
    "Delhi Capitals' win share against RCB?",
]


def main(database: str, model: str) -> None:
    db = duckdb.connect(database, read_only=True)
    players = tuple(load_players(db))
    names = cricket_gazetteer(name_list(list(players)))
    grounds = tuple(load_grounds(db))
    teams = tuple(load_teams(db))
    adapters = (("OllamaLLM", OllamaLLM(model)), ("LangChainLLM", LangChainLLM(model)))
    print(f"the model: {model}\n")
    agreed = 0
    for question in QUESTIONS:
        print(question)
        readings = []
        for label, llm in adapters:
            started = time.time()
            reading = read_question(question, llm, players, names, grounds, teams)
            seconds = time.time() - started
            readings.append(reading)
            print(
                f"  {label:13} {seconds:5.1f}s  by {reading.by}: {filled(reading.subject)}"
            )
            for unsure in reading.unsure:
                print(
                    f"  {'':13}         which {unsure.written}? ({len(unsure.candidates)})"
                )
        verdict = "DIFFERENT"
        if readings[0].subject == readings[1].subject:
            agreed += 1
            verdict = "the same"
        print(f"  {verdict}\n")
    print(f"{agreed} of {len(QUESTIONS)} read the same by both adapters")


def filled(subject) -> dict:
    found = {}
    for field in dataclasses.fields(subject):
        value = getattr(subject, field.name)
        if value is not None:
            found[field.name] = value
    return found


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
