"""The synthetic evidence the stub graph retrieves over.

Stand-in for the Cricsheet ball-by-ball index and the broader historical
index. Deliberately contains noise: if N7a's filtering is wrong, the extra
blocks leak in and the strike rates come out wrong, so the retrieval test has
something real to catch.

None of this is real match data. Values illustrate the flow of data only.
"""


def _expand(shot, balls, runs, outs, phase, bowler_type, player):
    """Expand an aggregate into deterministic ball-level rows summing to `runs`.

    A repeating scoring pattern is adjusted until the total matches exactly, so
    the corpus is reproducible and the strike rates are arithmetic rather than
    assertion.
    """
    pattern = [1, 0, 4, 1, 2, 0, 1, 1, 0, 2]
    seq = [pattern[i % len(pattern)] for i in range(balls)]

    diff, i = runs - sum(seq), 0
    while diff and i < balls * 8:
        j = i % balls
        if diff > 0 and seq[j] < 6:
            seq[j] += 1
            diff -= 1
        elif diff < 0 and seq[j] > 0:
            seq[j] -= 1
            diff += 1
        i += 1
    assert sum(seq) == runs, f"_expand: could not hit {runs} in {balls} balls"

    return [
        {"player": player, "shot": shot, "runs": r, "out": k < outs,
         "phase": phase, "bowler_type": bowler_type}
        for k, r in enumerate(seq)
    ]


# The last two blocks are deliberate noise -- if N7a's filtering is wrong they
# will leak into the results and the strike rates will come out wrong.
FAKE_CRICSHEET = (
    _expand("Cover Drive", balls=50, runs=71, outs=1,
            phase="powerplay", bowler_type="pace", player="Kohli")      # SR 142
    + _expand("Pull", balls=50, runs=56, outs=4,
              phase="powerplay", bowler_type="pace", player="Kohli")    # SR 112
    + _expand("Cover Drive", balls=30, runs=33, outs=1,
              phase="powerplay", bowler_type="spin", player="Kohli")    # noise
    + _expand("Pull", balls=20, runs=34, outs=1,
              phase="death", bowler_type="pace", player="Kohli")        # noise
)

# Aggregate context rather than individual deliveries.
FAKE_CONTEXT = [
    {"metric": "career_sr_vs_pace", "player": "Kohli", "value": 138.0},
    {"metric": "venue_sr", "player": "Kohli", "value": 151.2,
     "venue": "M. Chinnaswamy Stadium"},
    {"metric": "powerplay_dismissal_rate", "player": "Kohli", "value": 0.08},
    {"metric": "career_sr_vs_spin", "player": "Kohli", "value": 121.4},
]

# Deterministic stand-in for the continuous visual metrics the real N6 would
# read off the pose estimator. Keyed by shot so the numbers at least move in a
# plausible direction between strokes.
POSTURE_BY_SHOT = {
    "Cover Drive": {"weight_transfer": 0.88, "bat_face_angle": -4.0, "stride_len": 0.72},
    "Pull":        {"weight_transfer": 0.61, "bat_face_angle": 11.5, "stride_len": 0.34},
    "Sweep":       {"weight_transfer": 0.70, "bat_face_angle": 6.0, "stride_len": 0.55},
}
DEFAULT_POSTURE = {"weight_transfer": 0.65, "bat_face_angle": 0.0, "stride_len": 0.50}
