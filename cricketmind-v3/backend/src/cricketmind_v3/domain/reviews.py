"""M4: how often a team's DRS reviews change the umpire's decision (Day 42)."""

from cricketmind_v3.domain.facts import Fact

# how a review ended, in Cricsheet's words
CHANGED = "upheld"  # the review was right: the decision was changed
STOOD = "struck down"  # the decision stood; on umpire's call the team keeps its review


def overturn_rate(decisions: dict[str, int]) -> Fact | None:
    """Of the reviews that decided something, the share that changed the decision.

    decisions maps how a review ended to how many ended so. n is the reviews counted.
    Umpire's call is a decision that stood. A technical failure decided nothing, so it
    is left out (2 of the 17,153 reviews in the data).
    """
    decided = 0
    changed = 0
    for decision, times in decisions.items():
        if decision not in (CHANGED, STOOD):
            continue
        decided += times
        if decision == CHANGED:
            changed += times
    if decided == 0:
        return None
    return Fact(value=changed / decided, n=decided)
