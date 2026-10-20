from dataclasses import dataclass

from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import Subject


class RegistryError(ValueError):
    """A claim type is written wrongly."""


@dataclass(frozen=True, kw_only=True)
class ClaimType:
    """One item on the menu: a kind of claim, and the engine behind it."""

    name: str  # "strike_rate"
    provenance: Provenance  # DESCRIPTIVE
    source: str  # "N8.M1", the engine that computes it
    required_keys: tuple[str, ...]  # Subject fields that must be filled
    required_any: tuple[str, ...] = ()  # at least one of these must be filled

    def __post_init__(self):
        # ("player") is plain text; ("player",) is a tuple
        if not isinstance(self.required_keys, tuple):
            raise RegistryError(f"{self.name}: required_keys must be a tuple")
        if not isinstance(self.required_any, tuple):
            raise RegistryError(f"{self.name}: required_any must be a tuple")


# the menu: every kind of claim CricketMind may make (invariant I2)
CLAIM_TYPES = (
    # counted from past matches (N8)
    ClaimType(
        name="strike_rate",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M1",
        required_keys=("player",),
    ),
    ClaimType(
        name="dismissal_rate",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M1",
        required_keys=("player",),
    ),
    ClaimType(
        name="sr_difference",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M1",
        required_keys=("player", "shot", "other_shot"),
    ),
    ClaimType(
        name="sr_vs_average",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M1",
        required_keys=("player",),
    ),
    ClaimType(
        name="venue_first_innings_avg",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M2",
        required_keys=("venue", "match_format"),
    ),
    ClaimType(
        name="workload_ratio_flag",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M3",
        required_keys=("bowler", "date"),
    ),
    ClaimType(
        name="review_overturn_rate",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M4",
        required_keys=(),
        required_any=("team", "dismissal_type"),
    ),
    # measured from the clip (N6); the clip itself is a Claim rule (Day 7)
    ClaimType(
        name="weight_transfer",
        provenance=Provenance.MEASURED,
        source="N6",
        required_keys=(),
    ),
    # simulated (N9)
    ClaimType(
        name="p_dismissal_within_k",
        provenance=Provenance.PREDICTIVE,
        source="N9.T1",
        required_keys=("player", "shot", "phase", "bowler_type", "k"),
    ),
    ClaimType(
        name="matchup_difference",
        provenance=Provenance.PREDICTIVE,
        source="N9.T2",
        required_keys=("player", "bowler", "other_bowler", "phase", "k"),
    ),
)


def index_by_name(claim_types: tuple[ClaimType, ...]) -> dict[str, ClaimType]:
    """Name -> row, for fast lookup. A name listed twice is refused."""
    index = {}
    for claim_type in claim_types:
        if claim_type.name in index:
            raise RegistryError(f"{claim_type.name}: listed twice on the menu")
        index[claim_type.name] = claim_type
    return index


_INDEX = index_by_name(CLAIM_TYPES)


def lookup(name: str) -> ClaimType | None:
    """The menu row for this name, or None if it is not on the menu."""
    return _INDEX.get(name)


def missing_keys(claim_type: ClaimType, subject: Subject) -> list[str]:
    """What this claim type needs that the subject leaves empty. Empty list: all there."""
    missing = []
    for key in claim_type.required_keys:
        if is_empty(getattr(subject, key, None)):
            missing.append(key)

    # at least one of required_any must be filled
    if claim_type.required_any:
        found = False
        for key in claim_type.required_any:
            if not is_empty(getattr(subject, key, None)):
                found = True
        if not found:
            missing.append(" or ".join(claim_type.required_any))
    return missing


def is_empty(value) -> bool:
    # None, or "" from a model that left a box blank
    return value is None or value == ""
