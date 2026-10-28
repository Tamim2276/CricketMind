"""Draw the graph as it stands today, from the real code (Day 55).

Run from backend/, with the day's number:
    uv run python scripts/draw_graph.py 55
Writes docs/graph/day55.md. GitHub draws it, and so does VS Code with the
"Markdown Preview Mermaid Support" extension.
"""

import sys
from pathlib import Path

from cricketmind_v3.application.graph.build import build_graph

DOCS = Path(__file__).resolve().parents[2] / "docs" / "graph"


def main(day: str) -> None:
    picture = build_graph().get_graph().draw_mermaid()
    if not picture.endswith("\n"):
        picture += "\n"
    text = (
        f"# The graph on Day {day}\n\n"
        "Drawn from the code by `backend/scripts/draw_graph.py`; dotted arrows are "
        "branches.\n\n"
        "```mermaid\n" + picture + "```\n"
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / f"day{day}.md"
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main(sys.argv[1])
