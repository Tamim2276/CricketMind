"""The Faith Check, its bounded regeneration loop, and the withhold path.

The gate compares every cited figure against the sanctioned facts. If an agent
cites a figure that does not match, only that agent is sent back; if it still
fails once the budget is spent, its finding is withheld and the report says so.

What the gate checks is **citations, not prose**. A correct figure attributed
to the wrong subject, or a figure an agent states without citing, lies outside
what it can detect -- see the boundary experiment.
"""
from ..agents import sanctioned_facts
from ..config import MAX_REGENS
from ..state import CricketMindState, trace


def n_faith_check(state: CricketMindState) -> dict:
    """Faith Check -- the purple diamond.

    Compares every cited figure against the sanctioned facts. Writes the failure
    list (replacing it, not appending: `faith_failures` has no reducer on
    purpose, so a corrected agent does not keep looking broken).
    """
    truth = sanctioned_facts(state)
    failures = []

    for f in state.get("agent_findings") or []:
        if f.get("rejected"):
            continue
        for key, claimed in f["cites"].items():
            actual = truth.get(key)
            if actual != claimed:
                failures.append({"agent": f["agent"], "key": key,
                                 "claimed": claimed, "actual": actual})

    if failures:
        for x in failures:
            trace(state, "FAITH", f"REJECT {x['agent']}: {x['key']}={x['claimed']} "
                                  f"but engines say {x['actual']}")
    else:
        trace(state, "FAITH",
              f"all {len(state.get('agent_findings') or [])} findings verified")

    return {"faith_failures": failures}


def faith_gate(state: CricketMindState) -> str:
    """Route on the Faith Check result. Returns a route name only."""
    fails = state.get("faith_failures") or []
    if not fails:
        return "faithful"
    if state.get("regen_count", 0) >= MAX_REGENS:
        trace(state, "FAITH", f"regenerate budget spent ({MAX_REGENS}) -> dropping claims")
        return "exhausted"
    return "regenerate"


def regen_prep(state: CricketMindState) -> dict:
    """Body of the regenerate edge: owns the counter that bounds the loop."""
    n = state.get("regen_count", 0) + 1
    who = sorted({f["agent"] for f in state["faith_failures"]})
    trace(state, "REGEN", f"pass {n}/{MAX_REGENS}, re-running {len(who)}: {', '.join(who)}")
    return {"regen_count": n}


def dispatch_regen(state: CricketMindState) -> list:
    """Fan back out to exactly the agents that failed -- not all of them."""
    return sorted({f["agent"] for f in state.get("faith_failures") or []})


def drop_unfaithful(state: CricketMindState) -> dict:
    """Budget spent: withhold the unverifiable findings rather than publish them.

    Marks rather than deletes, because `agent_findings` merges through
    upsert_findings -- returning a shorter list would not remove anything. N12
    skips anything flagged, and the report says plainly what was withheld.
    """
    bad = {f["agent"] for f in state["faith_failures"]}
    marked = [{**f, "rejected": True}
              for f in state["agent_findings"] if f["agent"] in bad]
    trace(state, "DROP", f"withholding {len(marked)} unverifiable finding(s)")
    return {"agent_findings": marked, "status": "unverified_claims_dropped"}
