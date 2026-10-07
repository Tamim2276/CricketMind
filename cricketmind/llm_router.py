"""The optional language-model router: Qwen2.5-7B served locally by Ollama.

Only `propose_agents` calls this, and only when `Settings.router_mode == "llm"`.
The rules in N10 run afterwards either way, so the cap, the evidence rule and
every bound they imply hold whichever router proposed. The language model
proposes; the rules decide.

Uses stdlib urllib so the package has no extra dependency.
"""
import json
import time
import urllib.request

from .agents import AGENT_REGISTRY, AGENTS_BY_ID
from .config import MAX_ACTIVE_AGENTS, SEED


def ollama_ready(settings) -> bool:
    """True if Ollama is running and this run's router model is installed."""
    try:
        with urllib.request.urlopen(f"{settings.ollama_url}/api/tags", timeout=3) as reply:
            installed = [m["name"] for m in json.load(reply)["models"]]
    except OSError:                  # not running, refused, timed out
        return False
    return settings.router_model in installed


def llm_route(query: str, settings) -> dict:
    """Ask the local language model which agents should answer a query.

    The model sees each agent's id, stakeholder name and keyword list -- the
    same knowledge the keyword router has. Its reply is forced into a JSON
    shape: a short reason, then 1 to MAX_ACTIVE_AGENTS ids from the registry.
    temperature 0 and a fixed seed make repeated calls as stable as possible.

    Returns {"reason": str, "agents": [agent ids], "seconds": float}.
    Raises OSError if Ollama cannot be reached -- the caller decides what to do.
    """
    roster = "\n".join(
        f"- {a['id']}: {a['stakeholder']} (topics: {', '.join(a['keywords'])})"
        for a in AGENT_REGISTRY)
    system = ("You route questions about cricket to expert agents. Choose only the agents "
              "whose expertise the question actually needs. Judge by what the question "
              "means, not by the exact words it uses. Answer with a one-sentence reason "
              "that explains your choice, then the list of agents.\n\nAgents:\n" + roster)

    schema = {                      # the only shape Ollama will let it answer in
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "agents": {"type": "array", "minItems": 1, "maxItems": MAX_ACTIVE_AGENTS,
                       "items": {"type": "string", "enum": list(AGENTS_BY_ID)}},
        },
        "required": ["reason", "agents"],
    }
    body = {"model": settings.router_model, "stream": False, "format": schema,
            "options": {"temperature": 0, "seed": SEED},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": query}]}

    request = urllib.request.Request(f"{settings.ollama_url}/api/chat",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as reply:
        message = json.load(reply)["message"]["content"]
    answer = json.loads(message)

    answer["agents"] = list(dict.fromkeys(answer["agents"]))   # drop repeats, keep order
    answer["seconds"] = time.perf_counter() - start
    return answer
