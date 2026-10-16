from dataclasses import dataclass

from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.subject import Subject


class ClaimError(ValueError):
    """A claim broke one of its rules."""


@dataclass(frozen=True, kw_only=True)
class Claim:
    """One fact an agent wants to say, as data."""

    id: str  # "c1"
    kind: Provenance
    metric: str  # "strike_rate"
    subject: Subject
    value: float
    n: int | None = None  # balls behind a D claim
    interval: tuple[float, float] | None = None  # 95% range of a P claim
    clip_id: str | None = None  # the clip behind an M claim

    def __post_init__(self):
        # rule 1: Day 8 looks for {c1} in the sentence
        if not looks_like_claim_id(self.id):
            raise ClaimError(f"'{self.id}': a claim id is 'c' plus digits, like 'c1'")

        # rule 2: a count means nothing without its sample size
        if self.kind == Provenance.DESCRIPTIVE and (self.n is None or self.n < 1):
            raise ClaimError(f"{self.id}: a descriptive claim needs n, the number of balls")

        # rule 3: uncertainty travels with the claim (invariant I4)
        if self.kind == Provenance.PREDICTIVE:
            if self.interval is None:
                raise ClaimError(f"{self.id}: a predictive claim needs a range")
            low, high = self.interval
            if not low <= self.value <= high:
                raise ClaimError(
                    f"{self.id}: value {self.value} is outside its range {low} to {high}"
                )

        # rule 4: no measurement without a clip (invariant I5)
        if self.kind == Provenance.MEASURED and not self.clip_id:
            raise ClaimError(f"{self.id}: a measured claim needs the clip it came from")


def looks_like_claim_id(text: str) -> bool:
    """True for 'c1', 'c12'; False for 'claim1', 'c', 'C1'."""
    if len(text) < 2 or text[0] != "c":
        return False
    for char in text[1:]:
        if char not in "0123456789":
            return False
    return True
