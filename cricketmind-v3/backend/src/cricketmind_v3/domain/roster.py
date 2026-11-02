"""Which measures each agent may use (Day 62): the v4 roster table in the menu's own names.

C1 checks a claim against this list, and the capability filter reads it (D-083).
"""

ROSTER: dict[str, tuple[str, ...]] = {
    # franchises and boards
    "N11a": (
        "strike_rate",
        "dismissal_rate",
        "venue_first_innings_avg",
        "venue_chasing_win_share",
        "venue_wicket_share",
        "team_win_share",
    ),
    # coaches
    "N11b": (
        "strike_rate",
        "dismissal_rate",
        "sr_difference",
        "sr_vs_average",
        "p_dismissal_within_k",
        "matchup_difference",
    ),
    # players
    "N11c": (
        "strike_rate",
        "sr_difference",
        "sr_vs_average",
        "runs",
        "batting_average",
        "boundary_share",
        "dot_ball_share",
        "weight_transfer",
    ),
    # scouts
    "N11d": ("strike_rate", "sr_vs_average"),
    # fans
    "N11e": (
        "strike_rate",
        "runs",
        "batting_average",
        "team_win_share",
        "match_won_by_runs",
        "match_won_by_wickets",
    ),
    # stadiums and curators
    "N11f": (
        "venue_first_innings_avg",
        "venue_chasing_win_share",
        "venue_wicket_share",
    ),
    # bowlers and bowling coaches
    "N11g": ("wickets", "economy", "bowling_strike_rate", "matchup_difference"),
    # broadcasters and sponsors
    "N11h": ("strike_rate", "sr_difference", "sr_vs_average", "boundary_share"),
    # governing bodies and umpires
    "N11i": ("review_overturn_rate",),
    # academies
    "N11j": ("strike_rate", "sr_difference", "sr_vs_average", "weight_transfer"),
    # medical teams
    "N11k": ("workload_ratio_flag",),
    # player agents
    "N11l": ("strike_rate",),
    # anyone asking about rules: words only
    "N11m": (),
}
