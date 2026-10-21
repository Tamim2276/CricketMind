from decimal import ROUND_HALF_UP, Decimal

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.registry import lookup


class RenderError(ValueError):
    """A claim that cannot be turned into words."""


# metric: (words with a {} for the number, number style, decimal places)
METRICS = {
    "strike_rate": ("a strike rate of {}", "fixed", 1),
    "dismissal_rate": ("a dismissal rate of {} per ball", "percent", 1),
    "sr_difference": ("a strike-rate difference of {}", "signed", 1),
    "sr_vs_average": (
        "a strike-rate difference of {} from the average batter",
        "signed",
        1,
    ),
    "venue_first_innings_avg": ("a first-innings average of {} runs", "fixed", 0),
    "venue_chasing_win_share": ("a chasing win rate of {}", "percent", 0),
    "venue_wicket_share": ("{} of the wickets", "percent", 0),
    "workload_ratio_flag": ("a workload ratio of {}", "fixed", 2),
    "review_overturn_rate": ("a review overturn rate of {}", "percent", 0),
    "weight_transfer": ("a weight transfer of {}", "fixed", 2),
    "p_dismissal_within_k": ("a {} chance", "percent", 0),
    "matchup_difference": (
        "a difference of {} percentage points in the chance of a dismissal",
        "signed_points",
        0,
    ),
}

# "1 ball", "1 review", but "1 innings"
ONE = {"balls": "ball", "innings": "innings", "reviews": "review", "wickets": "wicket"}


def counted(n: int, unit: str) -> str:
    """A sample size in words: 50 balls, 1 ball, 1 innings."""
    if n == 1:
        return "1 " + ONE[unit]
    return f"{n} {unit}"


def _round_half_up(number: Decimal, places: int) -> str:
    if places == 0:
        step = Decimal(1)
    else:
        step = Decimal("0." + "0" * (places - 1) + "1")  # 2 places -> 0.01
    return str(number.quantize(step, rounding=ROUND_HALF_UP))


def _with_sign(text: str) -> str:
    # a difference shows its direction: "30.0" -> "+30.0"
    if Decimal(text) == 0:
        return text.lstrip("-")  # "-0.0" -> "0.0"
    if text.startswith("-"):
        return text
    return "+" + text


def fixed(value: float, places: int) -> str:
    """142.0 -> '142.0'. Halves always round up: 0.125 to 2 places is '0.13'."""
    return _round_half_up(Decimal(str(value)), places)


def percent(value: float, places: int) -> str:
    """0.405 -> '41%'. Multiplied in Decimal, so 0.145 is '15%', never '14%'."""
    return _round_half_up(Decimal(str(value)) * 100, places) + "%"


def signed(value: float, places: int) -> str:
    """30.0 -> '+30.0', -12.5 -> '-12.5'. Zero has no sign."""
    return _with_sign(fixed(value, places))


def signed_points(value: float, places: int) -> str:
    """0.12 -> '+12': the gap between two chances, in percentage points."""
    return _with_sign(_round_half_up(Decimal(str(value)) * 100, places))


def _number(value: float, style: str, places: int) -> str:
    if style == "fixed":
        return fixed(value, places)
    if style == "percent":
        return percent(value, places)
    if style == "signed":
        return signed(value, places)
    if style == "signed_points":
        return signed_points(value, places)
    raise RenderError(f"no number style called '{style}'")


def shown(metric: str, value: float) -> str:
    """A number exactly as a reader sees it: 0.405 as a chance is '41%'."""
    if metric not in METRICS:
        raise RenderError(f"no words for the metric '{metric}'")
    _words, style, places = METRICS[metric]
    return _number(value, style, places)


def _range(interval: tuple[float, float], style: str, places: int) -> str:
    low, high = interval
    return (
        "range " + _number(low, style, places) + " to " + _number(high, style, places)
    )


def render_claim(claim: Claim) -> str:
    """One claim as words plus its tag: 'a strike rate of 142.0 [D, 50 balls]'."""
    if claim.metric not in METRICS:
        raise RenderError(f"{claim.id}: no words for the metric '{claim.metric}'")
    claim_type = lookup(claim.metric)
    if claim_type is None:
        raise RenderError(f"{claim.id}: '{claim.metric}' is not on the menu")
    words, style, places = METRICS[claim.metric]
    text = words.format(_number(claim.value, style, places))

    if claim.kind == Provenance.DESCRIPTIVE:
        tag = "[D, " + counted(claim.n, claim_type.unit)
        if claim.interval is not None:
            tag += ", " + _range(claim.interval, style, places)  # a comparison
        tag += "]"
    elif claim.kind == Provenance.PREDICTIVE:
        tag = "[P, " + _range(claim.interval, style, places) + "]"
    else:
        tag = f"[M, clip {claim.clip_id}]"

    return text + " " + tag


def render(finding: Finding) -> str:
    """The finding's sentence with every slot filled in."""
    text = finding.template
    for claim in finding.claims:
        text = text.replace("{" + claim.id + "}", render_claim(claim))
    return text
