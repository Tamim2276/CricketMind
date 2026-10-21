import dataclasses
from dataclasses import dataclass

from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import Subject


class RegistryError(ValueError):
    """A claim type is written wrongly."""


# what the sample size n of a count can count
UNITS = ("balls", "innings", "reviews", "wickets")


@dataclass(frozen=True, kw_only=True)
class ClaimType:
    """One item on the menu: a kind of claim, and the engine behind it."""

    name: str  # "strike_rate"
    provenance: Provenance  # DESCRIPTIVE
    source: str  # "N8.M1", the engine that computes it
    required_keys: tuple[str, ...]  # Subject fields that must be filled
    required_any: tuple[str, ...] = ()  # at least one of these must be filled
    needs_range: bool = False  # comparisons and predictions carry their range (I4)
    unit: str = "balls"  # what n counts (C3, the renderer's tag)
    comparison: bool = False  # a difference between two things: 0 means none (C4, C7)

    def __post_init__(self):
        # ("player") is plain text; ("player",) is a tuple
        if not isinstance(self.required_keys, tuple):
            raise RegistryError(f"{self.name}: required_keys must be a tuple")
        if not isinstance(self.required_any, tuple):
            raise RegistryError(f"{self.name}: required_any must be a tuple")
        if self.provenance == Provenance.PREDICTIVE and not self.needs_range:
            raise RegistryError(f"{self.name}: a prediction always needs its range")
        if self.unit not in UNITS:
            raise RegistryError(f"{self.name}: n cannot count '{self.unit}'")
        if self.comparison and not self.needs_range:
            raise RegistryError(f"{self.name}: a comparison is judged by its range")


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
        needs_range=True,
        comparison=True,
    ),
    ClaimType(
        name="sr_vs_average",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M1",
        required_keys=("player",),
        needs_range=True,
        comparison=True,
    ),
    ClaimType(
        name="venue_first_innings_avg",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M2",
        required_keys=("venue", "match_format", "gender"),
        unit="innings",
    ),
    ClaimType(
        name="venue_chasing_win_share",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M2",
        required_keys=("venue", "match_format", "gender"),
        unit="innings",  # the chases that ended with a winner
    ),
    ClaimType(
        name="venue_wicket_share",
        provenance=Provenance.DESCRIPTIVE,
        source="N8.M2",
        required_keys=("venue", "match_format", "gender", "bowler_type"),
        unit="wickets",  # the bowlers' wickets whose bowler type is known
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
        # by team only: Cricsheet keeps no appeal kind, and the dismissal is known only
        # when the batter ended up out, so a rate by kind would be lopsided (Day 42)
        required_keys=("team", "gender"),
        unit="reviews",
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
        needs_range=True,
    ),
    ClaimType(
        name="matchup_difference",
        provenance=Provenance.PREDICTIVE,
        source="N9.T2",
        required_keys=("player", "bowler", "other_bowler", "phase", "k"),
        needs_range=True,
        comparison=True,
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


def extra_keys(claim_type: ClaimType, subject: Subject) -> list[str]:
    """Filled fields the claim type does not ask for (Day 40). Empty list: none.

    For measures with no filters: "India's first innings at Mirpur" is not the ground's
    average, so the team is extra and the engine gives no answer.
    """
    extra = []
    for field in dataclasses.fields(subject):
        if field.name in claim_type.required_keys:
            continue
        if field.name in claim_type.required_any:
            continue
        if not is_empty(getattr(subject, field.name)):
            extra.append(field.name)
    return extra


def is_empty(value) -> bool:
    # None, or "" from a model that left a box blank
    return value is None or value == ""


def check_registry(
    agent_types: dict[str, list[str]],
    engine_sources: set[str],
    renderer_metrics: set[str],
) -> list[str]:
    """Every problem with the menu, one sentence each. Empty list: all is well."""
    problems = []

    # 1. an agent wants a claim type that is not on the menu
    for agent, names in agent_types.items():
        for name in names:
            if lookup(name) is None:
                problems.append(f"{agent}: declares '{name}', which is not on the menu")

    # 2. a menu row with no engine to compute it
    for claim_type in CLAIM_TYPES:
        if claim_type.source not in engine_sources:
            problems.append(
                f"{claim_type.name}: no engine provides {claim_type.source}"
            )

    # 3. a menu row with no words to show it
    for claim_type in CLAIM_TYPES:
        if claim_type.name not in renderer_metrics:
            problems.append(f"{claim_type.name}: the renderer has no words for it")

    return problems
