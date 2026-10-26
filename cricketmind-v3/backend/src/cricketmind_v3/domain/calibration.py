"""Day 53: the engines' chances, made from older matches, set against the newest year.

Every model is fitted once on the train matches of a kind of cricket, then gives a chance
for each test ball (is he out on it?) and each window of his next balls (out within them?).
The pre-registration (thesis/prereg/day53_calibration.md) says which, and how they are judged.
"""

from dataclasses import dataclass

from cricketmind_v3.domain.matchups import Pair, expected, in_order, meetings_worth
from cricketmind_v3.domain.shrinkage import Prior, population_prior, shrunk_rate
from cricketmind_v3.domain.stages import (
    ORDERS,
    STARTS,
    Record,
    by_batter,
    in_stage,
    stage_of,
    usual_order,
)
from cricketmind_v3.domain.stages import in_order as records_in_order

K = 6  # the engine's question: out within his next 6 balls
MODELS = ("D", "A", "B", "C", "E")


@dataclass(frozen=True, kw_only=True)
class Moment:
    """One ball a batter faced, or was out on, as the test sees it."""

    match: str
    innings: int
    batter: str  # register ids
    bowler: str
    faced_before: int  # his balls faced in the innings before this one
    order: str  # where he batted in that innings
    out: bool
    faced: bool  # False for a wide he was out on


@dataclass(frozen=True, kw_only=True)
class Window:
    """From one ball: his next K balls, or fewer if the innings ended or he retired first.
    His own wicket never shortens it: that is what is being predicted."""

    start: Moment
    balls: int  # K, or fewer if the innings ended first
    out: bool  # out within them


def windows(moments: list[Moment], k: int = K) -> list[Window]:
    """A window from every moment: the moments of his innings from this one on, up to k
    of them. One cut short by the innings ending is kept, with fewer balls; one where he
    was out keeps all k (found on Day 53's first run: shortened at his wicket, a window
    was scored on a 2-ball chance for a wicket that fell, against the pre-registration)."""
    innings = {}
    for moment in moments:
        key = (moment.match, moment.innings, moment.batter)
        innings.setdefault(key, []).append(moment)
    found = []
    for his in innings.values():
        his.sort(key=_in_order)
        for i, start in enumerate(his):
            ahead = his[i : i + k]
            out = False
            for moment in ahead:
                if moment.out:
                    out = True
            balls = len(ahead)
            if out:
                balls = k
            found.append(Window(start=start, balls=balls, out=out))
    return found


@dataclass(frozen=True, kw_only=True)
class Fitted:
    """Everything the models learned from the train matches of one kind of cricket."""

    totals: dict[str, tuple[int, int]]  # batter -> (outs, faced)
    by_stage: dict[tuple[str, int], tuple[int, int]]  # (batter, stage) -> (outs, faced)
    usual: dict[str, str]  # batter -> where he usually bats
    every: Prior | None  # A: every batter
    order: dict[str, Prior | None]  # B: his part of the order
    stage: dict[tuple[str, int], Prior | None]  # C: and the stage
    # E: within each part of the order, every batter's and bowler's shrunk rate, the
    # normal rate, how much meetings are worth, and the meetings themselves
    batter_rates: dict[str, dict[str, float]]
    bowler_rates: dict[str, dict[str, float]]
    normal: dict[str, float]
    worth: dict[str, float]
    met: dict[tuple[str, str, str], tuple[int, int]]


def fit(records: list[Record], pairs: list[Pair]) -> Fitted:
    """Fit every model on the train matches: records by stage and part of the order, and
    every batter-bowler pair."""
    totals = {}
    by_stage = {}
    his_records = {}
    for record in records:
        outs, faced = totals.get(record.batter, (0, 0))
        totals[record.batter] = (outs + record.outs, faced + record.faced)
        key = (record.batter, record.stage)
        outs, faced = by_stage.get(key, (0, 0))
        by_stage[key] = (outs + record.outs, faced + record.faced)
        his_records.setdefault(record.batter, []).append(record)
    usual = {}
    for batter, his in his_records.items():
        usual[batter] = usual_order(his)

    order = {}
    stage = {}
    for name, _ in ORDERS:
        here = records_in_order(records, name)
        order[name] = population_prior(by_batter(here))
        for number in range(1, len(STARTS) + 1):
            stage[(name, number)] = population_prior(in_stage(here, number))

    batter_rates = {}
    bowler_rates = {}
    normal = {}
    worth = {}
    met = {}
    for name, _ in ORDERS:
        here = in_order(pairs, name)
        batters, bowlers, rate = _pair_totals(here)
        batter_prior = order[name]
        bowler_prior = population_prior(list(bowlers.values()))
        if batter_prior is None or bowler_prior is None or rate is None:
            continue
        batter_rates[name] = {}
        for who, (outs, faced) in batters.items():
            batter_rates[name][who] = shrunk_rate(batter_prior, outs, faced)
        bowler_rates[name] = {}
        for who, (outs, faced) in bowlers.items():
            bowler_rates[name][who] = shrunk_rate(bowler_prior, outs, faced)
        normal[name] = rate
        worth[name] = meetings_worth(here, batter_rates[name], bowler_rates[name], rate)
        for pair in here:
            met[(name, pair.batter, pair.bowler)] = (pair.outs, pair.faced)
    return Fitted(
        totals=totals,
        by_stage=by_stage,
        usual=usual,
        every=population_prior(by_batter(records)),
        order=order,
        stage=stage,
        batter_rates=batter_rates,
        bowler_rates=bowler_rates,
        normal=normal,
        worth=worth,
        met=met,
    )


def ball_chance(fitted: Fitted, model: str, moment: Moment) -> float | None:
    """The model's chance he is out on this ball; None when it gives none."""
    beta = posterior(fitted, model, moment.batter, stage_of(moment.faced_before + 1))
    if beta is not None:
        outs, survived = beta
        return outs / (outs + survived)
    if model == "E":
        return _matchup(fitted, moment)
    return None


def window_chance(fitted: Fitted, model: str, window: Window) -> float | None:
    """The model's chance he is out within the window's balls: one minus the chance he
    survives them all, averaged over the Beta curve, exactly (no replays). For E, its
    mean rate played over the balls."""
    start = window.start
    if model == "E":
        rate = _matchup(fitted, start)
        if rate is None:
            return None
        return 1 - (1 - rate) ** window.balls
    # the window's balls, by the Beta curve they share: one for D, A and B, one for each
    # stage for C
    counts = {}
    for ball in range(window.balls):
        number = stage_of(start.faced_before + ball + 1)
        if model != "C":
            number = 1
        counts[number] = counts.get(number, 0) + 1
    still = 1.0
    for number, count in counts.items():
        beta = posterior(fitted, model, start.batter, number)
        if beta is None:
            return None
        outs, survived = beta
        # surviving c balls, averaged over Beta(a, b): the product of (b + j) / (a + b + j)
        for j in range(count):
            still *= (survived + j) / (outs + survived + j)
    return 1 - still


def displayed_middle(fitted: Fitted, window: Window) -> float | None:
    """B's chance as the engine shows it: the middle of its replays, here the Beta curve's
    median (Kerman's (a - 1/3) / (a + b - 2/3)), then played over the window's balls."""
    beta = posterior(fitted, "B", window.start.batter, 1)
    if beta is None:
        return None
    outs, survived = beta
    median = (outs - 1 / 3) / (outs + survived - 2 / 3)
    median = min(max(median, 0.0), 1.0)
    return 1 - (1 - median) ** window.balls


def posterior(
    fitted: Fitted, model: str, batter: str, stage: int
) -> tuple[float, float] | None:
    """The Beta curve of his out-rate: the prior's outs and balls survived plus his own.
    For C, his record and the prior in this stage only."""
    if batter not in fitted.totals:
        return None  # no balls before the test year: the engine gives no answer either
    outs, faced = fitted.totals[batter]
    if faced < 1 or outs > faced:
        return None  # as the engine: no balls of his own, no answer about him
    if model == "D":
        return (0.5 + outs, 0.5 + faced - outs)
    if model == "A":
        prior = fitted.every
    elif model == "B":
        prior = fitted.order.get(fitted.usual[batter])
    elif model == "C":
        prior = fitted.stage.get((fitted.usual[batter], stage))
        outs, faced = fitted.by_stage.get((batter, stage), (0, 0))
    else:
        return None
    if prior is None or outs > faced:
        return None
    return (prior.outs + outs, prior.survived + faced - outs)


def _matchup(fitted: Fitted, moment: Moment) -> float | None:
    # E: the odds rule on B's groups, nudged by their meetings (the posterior mean)
    if moment.batter not in fitted.usual:
        return None
    name = fitted.usual[moment.batter]
    if name not in fitted.worth:
        return None
    batter = fitted.batter_rates[name].get(moment.batter)
    bowler = fitted.bowler_rates[name].get(moment.bowler)
    if batter is None or bowler is None:
        return None
    rate = expected(batter, bowler, fitted.normal[name])
    worth = fitted.worth[name]
    outs, faced = fitted.met.get((name, moment.batter, moment.bowler), (0, 0))
    return (rate * worth + outs) / (worth + faced)


def _pair_totals(pairs: list[Pair]):
    batters = {}
    bowlers = {}
    outs = 0
    balls = 0
    for pair in pairs:
        old = batters.get(pair.batter, (0, 0))
        batters[pair.batter] = (old[0] + pair.outs, old[1] + pair.faced)
        old = bowlers.get(pair.bowler, (0, 0))
        bowlers[pair.bowler] = (old[0] + pair.outs, old[1] + pair.faced)
        outs += pair.outs
        balls += pair.faced
    if balls == 0:
        return batters, bowlers, None
    return batters, bowlers, outs / balls


def _in_order(moment: Moment) -> tuple[int, int]:
    # his balls in the order he met them: a wide he was out on comes after the ball before
    if moment.faced:
        return (moment.faced_before, 0)
    return (moment.faced_before, 1)
