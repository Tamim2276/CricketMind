"""Where the Faith Check stops protecting the user (result R3).

The gate verifies *values*, not the claims wrapped around them. Four versions
of one agent's answer show the line:

| case          | what it does                             | gate     | reaches user |
|---------------|------------------------------------------|----------|--------------|
| honest        | cites the real figures, says them right  | passes   | yes          |
| fabricated    | cites 999.0                              | REJECTED | no           |
| misattributed | cites the real figures, swaps the shots  | passes   | **yes**      |
| uncited       | adds a figure it never cites             | passes   | **yes**      |

The last two are the result: the guarantee holds only if the citation contract
is enforced on the agents' output, so every numeric token is emitted as a
citation carrying its subject.

Changing what an agent *says* is done through `Settings.template_overrides`,
so nothing in the registry is mutated and no run can leak into another.
"""
from ..agents import AGENTS_BY_ID, sanctioned_facts
from ..config import CLIP, QUERY, Settings
from ..state import new_state
from .profile import default_graph

R3_AGENT = "n11b_tactical_analysis"
R3_HONEST = AGENTS_BY_ID[R3_AGENT]["template"]

R3_CASES = [
    {"case": "honest", "template": None, "lie_once": (),
     "claim": "The Pull returns SR 112.0 against the Cover Drive's SR 142.0"},

    {"case": "fabricated", "template": None, "lie_once": (R3_AGENT,),
     "claim": "999.0"},

    {"case": "misattributed", "lie_once": (),
     "template": R3_HONEST.replace("SR {worst_sr} against the {best_shot}'s SR {best_sr}",
                                   "SR {best_sr} against the {best_shot}'s SR {worst_sr}"),
     "claim": "The Pull returns SR 142.0 against the Cover Drive's SR 112.0"},

    {"case": "uncited", "lie_once": (),
     "template": R3_HONEST + " He scores 61% of his runs from boundaries.",
     "claim": "61% of his runs from boundaries"},
]
assert R3_CASES[2]["template"] != R3_HONEST, "the swap did not match the template"


def run_with_template(agent_id: str, template: str = None, lie_once=(), graph=None) -> dict:
    """Run the evaluation query with one agent's sentence template swapped.

    The override changes what the agent *writes* but not what it *cites*,
    which is exactly the distinction R3 is about. Returns the final state.
    """
    overrides = {agent_id: template} if template is not None else {}
    settings = Settings(confidence=0.92, lie_once=lie_once, trace=False,
                        template_overrides=overrides)
    return (graph or default_graph()).invoke(new_state(CLIP, QUERY, settings))


def run_boundary_cases(graph=None) -> list:
    """Run all four cases and report what the gate did and what reached the user."""
    rows = []
    for c in R3_CASES:
        final = run_with_template(R3_AGENT, c["template"], c["lie_once"], graph)
        coach = next(f for f in final["agent_findings"] if f["agent"] == R3_AGENT)
        rows.append({
            "case":            c["case"],
            "rejected":        final["regen_count"] > 0,   # the gate sent it back
            "regens":          final["regen_count"],
            "cites":           coach["cites"],
            "claim_in_report": c["claim"] in final["report"],
            "status":          final["status"],
        })
    return rows


def boundary_facts(graph=None) -> dict:
    """The sanctioned facts for the evaluation query, for checking R3's claims."""
    return sanctioned_facts(run_with_template(R3_AGENT, graph=graph))
