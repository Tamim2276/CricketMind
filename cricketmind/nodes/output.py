"""N12 synthesis and N13 final output."""
from ..state import CricketMindState, trace


def n12_supervisor_synthesis(state: CricketMindState) -> dict:
    """N12 -- Supervisor LLM. Composes the verified findings into one report.

    Real version: an LLM call that resolves overlap between agents and writes
    connective prose. The ordering and the withheld-claims section are policy,
    not generation, so they stay here either way.
    """
    all_findings = state.get("agent_findings") or []
    kept = [f for f in all_findings if not f.get("rejected")]
    dropped = [f for f in all_findings if f.get("rejected")]

    rank = {a: i for i, a in enumerate(state.get("active_agents") or [])}
    kept.sort(key=lambda f: rank.get(f["agent"], 99))

    dna, stats, sim = state["shot_dna"], state["stats_results"], state["sim_results"]
    sr_line = ", ".join(f"{s} SR {a['strike_rate']}" for s, a in stats["by_shot"].items())

    lines = [
        f"CRICKETMIND REPORT -- {dna['filters'].get('player', 'batsman')}",
        "=" * 72,
        f"Query      : {state['query']}",
        (f"Vision     : {dna['shot']} at {dna['confidence']:.0%} confidence "
         f"({dna['attempts']} attempt(s))") if dna.get("shot")
        else "Vision     : no clip supplied -- text-only query",
        f"Evidence   : {stats['sample_size']} deliveries -- {sr_line}",
        f"Simulation : {sim['scenario']} -> {sim['p_dismissal']:.1%} "
        f"(95% CI {sim['ci95'][0]:.1%}-{sim['ci95'][1]:.1%}, {sim['trials']:,} trials)",
        "",
        f"FINDINGS ({len(kept)} agent(s))",
        "-" * 72,
    ]
    for f in kept:
        lines.append(f"[{f['stakeholder']}]")
        lines.append(f"  {f['text']}")
        lines.append("")

    if dropped:
        lines += ["WITHHELD -- failed verification against N8/N9", "-" * 72]
        lines += [f"  {f['stakeholder']} ({f['agent']})" for f in dropped]
        lines.append("")

    trace(state, "N12", f"synthesised {len(kept)} finding(s), withheld {len(dropped)}")
    return {"report": "\n".join(lines)}


def n13_final_output(state: CricketMindState) -> dict:
    """N13 -- the terminal node. Stamps the run's overall status.

    Only `ok` and `unverified_claims_dropped` can be stamped here. A clip that
    failed the confidence gate never reaches this node: it terminates at
    `low_confidence_exit` with no report at all.
    """
    report = state.get("report")
    assert report, "n13_final_output: n12 produced no report"
    status = state.get("status", "ok")
    trace(state, "N13", f"report ready ({len(report.splitlines())} lines), status={status}")
    return {"status": status}
