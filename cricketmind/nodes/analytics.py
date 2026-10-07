"""N8 Stats Engine and N9 What-If Simulator -- the two sanctioned engines.

These are the ONLY components permitted to produce a number. Every other
component must cite one of their values rather than compute its own, and the
Faith Check validates agent claims against these two dicts and nothing else.
"""
import random

from ..config import SEED, SIM_DELIVERIES, SIM_TRIALS
from ..state import CricketMindState, trace


def n8_stats_engine(state: CricketMindState) -> dict:
    """N8 --- Statistical Analysis Engine.

    Deterministic aggregation over the deliveries N7a retrieved. No sampling, no
    generation: every number it emits is a count or a ratio that can be
    recomputed by hand from `stats_hits`. That reproducibility is what makes the
    Faith Check downstream meaningful.
    """
    hits = state.get("stats_hits") or []
    assert hits, "n8_stats_engine: no stats_hits -- did n7a_stats_retrieval run?"

    by_shot = {}
    for r in hits:
        a = by_shot.setdefault(r["shot"], {"balls": 0, "runs": 0, "dismissals": 0})
        a["balls"] += 1
        a["runs"] += r["runs"]
        a["dismissals"] += int(r["out"])

    for a in by_shot.values():
        a["strike_rate"] = round(100.0 * a["runs"] / a["balls"], 1)
        a["dismissal_rate"] = round(a["dismissals"] / a["balls"], 4)

    ranked = sorted(by_shot.items(), key=lambda kv: kv[1]["strike_rate"], reverse=True)
    results = {
        "by_shot":     by_shot,
        "best_shot":   ranked[0][0],
        "worst_shot":  ranked[-1][0],
        "sample_size": len(hits),
    }

    trace(state, "N8", "  ".join(f"{s}: SR {a['strike_rate']}" for s, a in ranked))
    return {"stats_results": results}


def n9_whatif_simulator(state: CricketMindState) -> dict:
    """N9 --- What-If Simulator (Monte Carlo).

    The only source of forward-looking, non-observational claims in the system.
    Answers questions no historical record can answer directly by sampling from
    the distribution the retrieved deliveries describe, and always reports an
    interval alongside the point estimate.
    """
    hits = state.get("stats_hits") or []
    assert hits, "n9_whatif_simulator: no stats_hits -- did n7a_stats_retrieval run?"

    # per-ball dismissal risk by shot, from the same rows N8 reads
    tally = {}
    for r in hits:
        t = tally.setdefault(r["shot"], {"balls": 0, "outs": 0})
        t["balls"] += 1
        t["outs"] += int(r["out"])

    vulnerable = max(tally, key=lambda s: tally[s]["outs"] / tally[s]["balls"])
    p_ball = tally[vulnerable]["outs"] / tally[vulnerable]["balls"]

    rng = random.Random(SEED)
    dismissed = 0
    for _ in range(SIM_TRIALS):
        for _ in range(SIM_DELIVERIES):
            if rng.random() < p_ball:
                dismissed += 1
                break

    p = dismissed / SIM_TRIALS
    se = (p * (1 - p) / SIM_TRIALS) ** 0.5

    results = {
        "scenario":       f"{SIM_DELIVERIES} consecutive deliveries targeting "
                          f"the {vulnerable.lower()}",
        "target_shot":    vulnerable,
        "p_ball":         round(p_ball, 4),
        "p_dismissal":    round(p, 4),
        "ci95":           [round(p - 1.96 * se, 4), round(p + 1.96 * se, 4)],
        "analytic_check": round(1 - (1 - p_ball) ** SIM_DELIVERIES, 4),
        "trials":         SIM_TRIALS,
    }

    trace(state, "N9", f"{vulnerable}: p_ball={p_ball:.3f} -> "
                       f"P(out in {SIM_DELIVERIES}) = {p:.3f} +/- {1.96 * se:.3f}")
    return {"sim_results": results}
