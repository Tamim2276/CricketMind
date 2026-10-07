"""N6 Shot DNA and the N7a / N7b retrieval fan-out.

The Shot DNA is the interface boundary of the whole system: everything
downstream consumes it and never sees a pixel.
"""
from ..corpus import DEFAULT_POSTURE, FAKE_CONTEXT, FAKE_CRICSHEET, POSTURE_BY_SHOT
from ..state import CricketMindState, trace


def n6_shot_dna(state: CricketMindState) -> dict:
    """Fuse whatever grounding is available into one structured record.

    Two ways in. On the **video** path the Conf. Check has accepted a
    classification, so `shot` and `posture` are populated. On the **text-only**
    path there was no clip at all, so both are None and the query's own filters
    carry the whole grounding. N3 must have run either way -- without a query
    there is nothing to ground.

    Posture is deliberately left as None rather than filled with a default when
    no clip was supplied. A default would be an invented measurement entering
    the sanctioned-fact surface, which is precisely the fabrication the Faith
    Check exists to prevent.
    """
    label = state.get("shot_label")
    emb = state.get("query_embedding")
    assert emb is not None, "n6_shot_dna: n3_text_encoder has not run"

    from_video = label is not None

    dna = {
        "modality":   "video" if from_video else "text",
        "shot":       label,
        "confidence": state.get("shot_confidence") if from_video else None,
        "posture":    POSTURE_BY_SHOT.get(label, DEFAULT_POSTURE) if from_video else None,
        "filters":    state.get("query_filters", {}),
        "attempts":   state.get("retry_count", 0) + 1 if from_video else 0,
    }

    if from_video:
        trace(state, "N6", f"shot_dna: {dna['shot']} p={dna['confidence']:.2f} "
                           f"filters={dna['filters']}")
    else:
        trace(state, "N6", f"shot_dna: text-only, no clip -- filters={dna['filters']}")
    return {"shot_dna": dna}


def n7a_stats_retrieval(state: CricketMindState) -> dict:
    """N7a --- narrow, high-precision retrieval from the ball-by-ball index.

    Filters on player, phase and bowler type from the query, and on the union of
    the shot the video showed and any shots the query named. On a text-only
    query the video contributes nothing, so the query must name at least one
    shot itself -- otherwise there is no retrieval key at all.
    """
    dna = state.get("shot_dna")
    assert dna is not None, "n7a_stats_retrieval: n6_shot_dna has not run"
    f = dna["filters"]

    wanted = set(f.get("shots", []))
    if dna.get("shot"):
        wanted.add(dna["shot"])
    assert wanted, ("n7a_stats_retrieval: nothing to retrieve -- a text-only query "
                    "must name at least one shot")

    hits = [
        r for r in FAKE_CRICSHEET
        if r["shot"] in wanted
        and (("player" not in f) or r["player"] == f["player"])
        and (("phase" not in f) or r["phase"] == f["phase"])
        and (("bowler_type" not in f) or r["bowler_type"] == f["bowler_type"])
    ]

    trace(state, "N7a", f"shots={sorted(wanted)} -> {len(hits)} deliveries "
                        f"(from {len(FAKE_CRICSHEET)})")
    return {"stats_hits": hits}


def n7b_context_retrieval(state: CricketMindState) -> dict:
    """N7b --- broad, high-recall retrieval of historical context.

    Deliberately does NOT filter by shot. Its job is the surrounding envelope:
    career form, venue behaviour, comparable conditions.
    """
    dna = state.get("shot_dna")
    assert dna is not None, "n7b_context_retrieval: n6_shot_dna has not run"
    player = dna["filters"].get("player")

    hits = [r for r in FAKE_CONTEXT if player is None or r["player"] == player]
    trace(state, "N7b", f"player={player} -> {len(hits)} context rows")
    return {"context_hits": hits}
