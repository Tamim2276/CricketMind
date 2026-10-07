"""N1-N5, N3, and the Conf. Check gate with its bounded retry loop.

Every node keeps the signature the real implementation will use:

    def nX_something(state: CricketMindState) -> dict:
        ...
        return {"only": "the", "keys": "it changed"}

Making a node real means replacing its *body*. The signature, the node name
and the edges never change -- that is the whole reason for building the graph
with stubs first.
"""
from ..config import (CLASS_TO_IDX, CONF_THRESHOLD, FEATURE_DIM, MAX_RETRIES,
                      NUM_FRAMES, SHOT_CLASSES)
from ..state import CricketMindState, FakeTensor, settings_of, trace


def n1_video_input(state: CricketMindState) -> dict:
    """Sample NUM_FRAMES frames from the clip.

    Real version: decode the video and uniformly sample 15 frames at 224x224.
    """
    path = state["video_path"]
    assert path, "N1: no video_path in state"
    frames = [f"{path}#f{i:02d}" for i in range(NUM_FRAMES)]
    trace(state, "N1", f"sampled {len(frames)} frames from {path}")
    return {"frames": frames}


def n2_vision_encoder(state: CricketMindState) -> dict:
    """Per-frame spatial embedding via EfficientNetV2-S.

    Frame-independent: no temporal reasoning happens here.
    """
    frames = state.get("frames")
    assert frames, "N2: N1 did not run (no frames in state)"
    feats = FakeTensor((len(frames), FEATURE_DIM), "EfficientNetV2-S per-frame")
    trace(state, "N2", f"encoded {len(frames)} frames -> {feats.shape}")
    return {"frame_features": feats}


def n4_temporal_fusion(state: CricketMindState) -> dict:
    """Temporal transformer (2 layers, 4 heads) + temporal mean pooling.

    Not a GRU: self-attention lets frame 0 (stance) relate directly to
    frame 14 (follow-through) instead of through 14 hidden states.
    """
    feats = state.get("frame_features")
    assert feats is not None, "N2 did not run (no frame_features in state)"
    pooled = FakeTensor((FEATURE_DIM,), "transformer mean-pooled")
    trace(state, "N4", f"{feats.shape} -> self-attention (2L/4H) -> {pooled.shape}")
    return {"temporal_features": pooled}


def n5_classifier_head(state: CricketMindState) -> dict:
    """Linear head over the 15 CricShot10k classes.

    Attempt-aware. When the Conf. Check sends control back to N1 the clip is
    re-sampled, so the real head would see different frames and produce a
    different confidence. `Settings.confidence` may therefore be a sequence
    indexed by retry_count.
    """
    pooled = state.get("temporal_features")
    assert pooled is not None, "N5: N4 did not run (no temporal_features in state)"

    cfg = settings_of(state)
    attempt = state.get("retry_count", 0)
    if isinstance(cfg.confidence, (list, tuple)):
        conf = cfg.confidence[min(attempt, len(cfg.confidence) - 1)]
    else:
        conf = cfg.confidence

    label = cfg.label
    assert label in SHOT_CLASSES, f"N5: {label!r} is not a CricShot10k class"
    trace(state, "N5", f"attempt {attempt}: {label} (idx {CLASS_TO_IDX[label]}) p={conf:.2f}")
    return {"shot_label": label, "shot_confidence": conf}


def n3_text_encoder(state: CricketMindState) -> dict:
    """Embed the query and pull out the retrieval filters it implies.

    Also extracts which shots the query *names*. The running example compares a
    cover drive with a pull, so retrieval must fetch both -- filtering on the
    shot the video happened to contain would silently drop half the comparison.

    Real version: a sentence embedding plus a structured parse (entity linking
    for the player, phase and bowler-type extraction). The keyword matching here
    is a stand-in that keeps the stub deterministic.
    """
    q = state.get("query")
    assert q, "n3_text_encoder: no query in state"

    ql = q.lower()
    filters = {}

    if "spin" in ql:
        filters["bowler_type"] = "spin"
    elif "pace" in ql or "fast" in ql:
        filters["bowler_type"] = "pace"

    if "powerplay" in ql:
        filters["phase"] = "powerplay"
    elif "death" in ql:
        filters["phase"] = "death"

    # The player named FIRST in the question is the one it is about:
    # "What if Rohit faced the attack instead of Kohli?" is about Rohit.
    # (Checking the names in a fixed order used to pick Kohli here.)
    named_players = [p for p in ("kohli", "rohit", "babar", "root", "smith") if p in ql]
    if named_players:
        first = min(named_players, key=ql.index)    # smallest position in the text
        filters["player"] = first.capitalize()

    # which shots does the QUERY mention, regardless of what the video showed
    named = [s for s in SHOT_CLASSES if s.lower() in ql]
    if named:
        filters["shots"] = named

    emb = FakeTensor((768,), "query embedding")
    trace(state, "N3", f"query -> {emb.shape}, filters={filters}")
    return {"query_embedding": emb, "query_filters": filters}


def conf_gate(state: CricketMindState) -> str:
    """Conf. Check --- the red diamond in Figure 1.

    Returns a ROUTE NAME only. It must not return a state update; LangGraph
    ignores anything a conditional function returns other than the route.
    """
    conf = state.get("shot_confidence", 0.0)
    tries = state.get("retry_count", 0)

    if conf >= CONF_THRESHOLD:
        trace(state, "GATE", f"p={conf:.2f} >= {CONF_THRESHOLD} -> accept")
        return "accept"

    if tries >= MAX_RETRIES:
        trace(state, "GATE", f"p={conf:.2f} < {CONF_THRESHOLD}, budget spent "
                             f"({tries}/{MAX_RETRIES}) -> exhausted")
        return "exhausted"

    trace(state, "GATE", f"p={conf:.2f} < {CONF_THRESHOLD} -> retry {tries + 1}/{MAX_RETRIES}")
    return "retry"


def retry_prep(state: CricketMindState) -> dict:
    """The body of the red dashed edge back to N1.

    Real version: request an alternate camera angle, slide the 15-frame sampling
    window, or apply test-time augmentation. Here it only increments the counter
    -- which is the part that makes the loop terminate.
    """
    n = state.get("retry_count", 0) + 1
    trace(state, "RETRY", f"re-sampling clip, attempt {n}")
    return {"retry_count": n}


def low_confidence_exit(state: CricketMindState) -> dict:
    """Graceful degradation once the retry budget is spent.

    This node is TERMINAL: its only edge is to END. A clip that reaches here
    never produces a report, which is the point -- a label the classifier is
    not sure of must not reach the agents.

    Deliberately does NOT clear shot_label: downstream code should be able to
    see what the model guessed *and* that it was not trusted.
    """
    trace(state, "DEGRD", f"giving up after {state.get('retry_count', 0)} retries")
    return {"status": "low_confidence"}


def has_video(state: CricketMindState) -> str:
    """START router. Decides whether the vision branch runs at all.

    Returns a ROUTE NAME only, like conf_gate and faith_gate.
    """
    route = "video" if state.get("video_path") else "text_only"
    trace(state, "START", f"{route} query")
    return route
