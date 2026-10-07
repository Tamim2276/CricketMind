"""The twelve stakeholder agents, and the sanctioned-fact surface they cite.

Each registry entry carries:

  keywords  stand-in for a real semantic router
  template  the agent's voice; formatted against the sanctioned facts
  cites     which sanctioned numbers this agent is allowed to quote. The
            Faith Check validates exactly these keys, so an agent cannot
            smuggle in a figure N8 and N9 never produced.
"""
from .config import SIM_DELIVERIES
from .state import CricketMindState, settings_of, trace

AGENT_REGISTRY = [
    {"id": "n11a_club_operations", "stakeholder": "Franchises / Boards",
     "keywords": ("budget", "auction", "squad", "contract", "fixture", "travel", "roster"),
     "cites": ("best_sr", "sample_size"),
     "template": "Retention case for {player}: {best_shot} returns SR {best_sr} across "
                 "{sample_size} tracked deliveries in this phase, which supports the current "
                 "salary band at the next auction."},

    {"id": "n11b_tactical_analysis", "stakeholder": "Coaches",
     "keywords": ("tactic", "bowl", "field", "matchup", "match-up", "strategy",
                  "weakness", "opposition", "plan", "attack"),
     "cites": ("best_sr", "worst_sr", "p_pct"),
     "template": "Bowl short at {player}. The {worst_shot} returns SR {worst_sr} against "
                 "the {best_shot}'s SR {best_sr}; {deliveries} consecutive back-of-a-length "
                 "deliveries carry a {p_pct}% chance of dismissal."},

    {"id": "n11c_personal_performance", "stakeholder": "Players",
     "keywords": ("technique", "shot", "form", "drive", "pull", "sweep", "cut",
                  "trend", "milestone", "selection"),
     "cites": ("best_sr", "worst_sr", "weight_transfer"),
     "template": "Weight transfer on the {best_shot} measures {weight_transfer}, and it is "
                 "the stronger stroke at SR {best_sr}. The {worst_shot} lags at SR {worst_sr} "
                 "-- shot selection, not technique, is the gap to close."},

    {"id": "n11d_global_scouting", "stakeholder": "Scouts",
     "keywords": ("scout", "emerging", "league", "prospect", "talent pool", "recruit"),
     "cites": ("best_sr", "career_sr"),
     "template": "Benchmark for comparable recruits: SR {best_sr} on the {best_shot} against "
                 "a career SR of {career_sr} versus pace sets the bar for this role."},

    {"id": "n11e_fan_concierge", "stakeholder": "Fans",
     "keywords": ("fan", "fantasy", "ticket", "merchandise", "stadium guide", "highlight"),
     "cites": ("best_sr", "venue_sr"),
     "template": "Fantasy note: {player} strikes at {venue_sr} at this venue and {best_sr} "
                 "on the {best_shot} -- a strong captaincy pick while the field is up."},

    {"id": "n11f_stadium_operations", "stakeholder": "Stadiums",
     "keywords": ("pitch prep", "curator", "crowd", "concession", "floodlight", "drs"),
     "cites": ("sample_size",),
     "template": "Curation note: {sample_size} tracked deliveries in this phase suggest "
                 "the surface is holding pace; no change to the rolling schedule."},

    {"id": "n11g_sponsorship_intelligence", "stakeholder": "Sponsors",
     "keywords": ("sponsor", "brand", "logo", "roi", "impression", "visibility"),
     "cites": ("best_sr",),
     "template": "Visibility: boundary strokes drive replay volume, and the {best_shot} at "
                 "SR {best_sr} is this batsman's most replayed shot -- prime logo exposure."},

    {"id": "n11h_content_production", "stakeholder": "Broadcasters",
     "keywords": ("commentary", "broadcast", "overlay", "wagon wheel", "clip",
                  "highlight", "graphic"),
     "cites": ("best_sr", "worst_sr"),
     "template": "Ticker line: \"{player} -- {best_shot} SR {best_sr}, {worst_shot} SR "
                 "{worst_sr} in the powerplay.\" Wagon wheel overlay ready."},

    {"id": "n11i_compliance", "stakeholder": "Governing Bodies",
     "keywords": ("eligibility", "compliance", "icc", "audit", "bio-bubble", "sanction"),
     "cites": ("sample_size",),
     "template": "No eligibility or conduct flags attached to the {sample_size} deliveries "
                 "reviewed. Match officials' records are consistent."},

    {"id": "n11j_talent_development", "stakeholder": "Academies",
     "keywords": ("academy", "u-16", "u-19", "drill", "coaching plan", "development"),
     "cites": ("worst_sr", "weight_transfer"),
     "template": "Drill focus for age-group squads: replicate the {weight_transfer} weight "
                 "transfer of the {best_shot}; the {worst_shot} at SR {worst_sr} is the "
                 "stroke to rebuild first."},

    {"id": "n11k_injury_management", "stakeholder": "Medical Teams",
     "keywords": ("injury", "workload", "fatigue", "fitness", "recovery",
                  "hamstring", "return-to-play"),
     "cites": ("sample_size",),
     "template": "Workload: {sample_size} deliveries faced in this phase sits inside the "
                 "weekly threshold. No soft-tissue risk flag raised."},

    {"id": "n11l_player_representation", "stakeholder": "Player Agents / Managers",
     "keywords": ("valuation", "market", "agent", "transfer", "salary", "worth"),
     "cites": ("best_sr", "career_sr"),
     "template": "Valuation input: SR {best_sr} on the {best_shot} against a career SR of "
                 "{career_sr} versus pace supports a top-bracket ask this cycle."},
]

AGENTS_BY_ID = {a["id"]: a for a in AGENT_REGISTRY}

# Agents that cannot answer without a posture measurement, i.e. without a clip.
POSTURE_AGENTS = {a["id"] for a in AGENT_REGISTRY if "weight_transfer" in a["cites"]}


def agent_name(agent_id: str) -> str:
    """Short stakeholder name for tables: "Franchises / Boards" -> "Franchises"."""
    return AGENTS_BY_ID[agent_id]["stakeholder"].split(" /")[0]


def sanctioned_facts(state: CricketMindState) -> dict:
    """Every number an agent is permitted to use, in one flat dict.

    Sourced only from N8 (`stats_results`), N9 (`sim_results`), N6's posture
    metrics and N7b's context rows. If a value is not in here, no agent may
    quote it -- and the Faith Check enforces that.

    The surface is **dynamic**. A text-only query carries no clip, so no posture
    was measured and `weight_transfer` is simply absent rather than defaulted.
    The router reads the same rule and declines to wake any agent whose
    citations cannot be satisfied.
    """
    stats = state["stats_results"]
    sim = state["sim_results"]
    dna = state["shot_dna"]

    best, worst = stats["best_shot"], stats["worst_shot"]
    ctx = {r["metric"]: r["value"] for r in (state.get("context_hits") or [])}

    facts = {
        "player":      dna["filters"].get("player", "the batsman"),
        "best_shot":   best,
        "worst_shot":  worst,
        "best_sr":     stats["by_shot"][best]["strike_rate"],
        "worst_sr":    stats["by_shot"][worst]["strike_rate"],
        "sample_size": stats["sample_size"],
        "p_pct":       round(sim["p_dismissal"] * 100, 1),
        "deliveries":  SIM_DELIVERIES,
        "target_shot": sim["target_shot"],
        "career_sr":   ctx.get("career_sr_vs_pace"),
        "venue_sr":    ctx.get("venue_sr"),
    }

    # video-only facts: present only when a clip was actually analysed
    if dna.get("shot"):
        facts["shot"] = dna["shot"]
    if dna.get("posture"):
        facts["weight_transfer"] = dna["posture"]["weight_transfer"]

    return facts


def make_agent(config: dict):
    """Build one stakeholder agent node from its registry entry.

    A factory rather than twelve functions: the agents differ only in voice and
    in which facts they may quote, so that difference belongs in data. The node
    keeps the standard `(state) -> dict` signature, so an agent can later become
    a real LLM call without the graph changing.

    Two things come from the run's Settings rather than a global:
      - whether this agent fabricates a cited value (`lie_once`/`lie_always`)
      - whether its sentence template has been replaced, which is how the
        Faith Check boundary experiment changes what an agent *says* without
        changing what it *cites*
    """
    def _agent(state: CricketMindState) -> dict:
        cid = config["id"]
        cfg = settings_of(state)
        facts = sanctioned_facts(state)

        lying = (cid in cfg.lie_always) or (
            cid in cfg.lie_once and state.get("regen_count", 0) == 0)
        if lying:
            # corrupt the fact itself, so the prose and the citation agree --
            # exactly the failure mode the Faith Check exists to catch
            facts = {**facts, config["cites"][0]: 999.0}

        template = cfg.template_overrides.get(cid, config["template"])
        text = template.format(**facts)
        cites = {k: facts[k] for k in config["cites"]}

        trace(state, "N11", f"{config['stakeholder']}: cites {list(cites)}"
                            + ("   <-- FABRICATED" if lying else ""))
        return {"agent_findings": [{
            "agent":       cid,
            "stakeholder": config["stakeholder"],
            "text":        text,
            "cites":       cites,
        }]}

    _agent.__name__ = config["id"]
    return _agent


# One node per registry entry. All twelve are registered on the graph; the
# router decides which of them actually run.
AGENT_NODES = {a["id"]: make_agent(a) for a in AGENT_REGISTRY}
