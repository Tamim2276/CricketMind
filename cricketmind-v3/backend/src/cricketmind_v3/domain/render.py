from decimal import ROUND_HALF_UP, Decimal

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance


class RenderError(ValueError):
    """A claim that cannot be turned into words."""


# metric: (words with a {} for the number, number style, decimal places)
METRICS = {
    "strike_rate": ("a strike rate of {}", "fixed", 1),
    "dismissal_rate": ("a dismissal rate of {} per ball", "percent", 1),
    "p_dismissal_within_k": ("a {} chance", "percent", 0),
    "weight_transfer": ("a weight transfer of {}", "fixed", 2),
}


def _round_half_up(number: Decimal, places: int) -> str:
    if places == 0:
        step = Decimal(1)
    else:
        step = Decimal("0." + "0" * (places - 1) + "1")  # 2 places -> 0.01
    return str(number.quantize(step, rounding=ROUND_HALF_UP))


def fixed(value: float, places: int) -> str:
    """142.0 -> '142.0'. Halves always round up: 0.125 to 2 places is '0.13'."""
    return _round_half_up(Decimal(str(value)), places)


def percent(value: float, places: int) -> str:
    """0.405 -> '41%'. Multiplied in Decimal, so 0.145 is '15%', never '14%'."""
    return _round_half_up(Decimal(str(value)) * 100, places) + "%"


def _number(value: float, style: str, places: int) -> str:
    if style == "percent":
        return percent(value, places)
    return fixed(value, places)


def render_claim(claim: Claim) -> str:
    """One claim as words plus its tag: 'a strike rate of 142.0 [D, 50 balls]'."""
    if claim.metric not in METRICS:
        raise RenderError(f"{claim.id}: no words for the metric '{claim.metric}'")
    words, style, places = METRICS[claim.metric]
    text = words.format(_number(claim.value, style, places))

    if claim.kind == Provenance.DESCRIPTIVE:
        if claim.n == 1:
            tag = "[D, 1 ball]"
        else:
            tag = f"[D, {claim.n} balls]"
    elif claim.kind == Provenance.PREDICTIVE:
        low, high = claim.interval
        low_text = _number(low, style, places)
        high_text = _number(high, style, places)
        tag = f"[P, range {low_text} to {high_text}]"
    else:
        tag = f"[M, clip {claim.clip_id}]"

    return text + " " + tag


def render(finding: Finding) -> str:
    """The finding's sentence with every slot filled in."""
    text = finding.template
    for claim in finding.claims:
        text = text.replace("{" + claim.id + "}", render_claim(claim))
    return text
