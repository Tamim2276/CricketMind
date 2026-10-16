from enum import Enum


class Provenance(Enum):
    """Where a number came from. Only these three are allowed (invariant I1)."""

    MEASURED = "measured"  # N6: measured from the clip
    DESCRIPTIVE = "descriptive"  # N8: counted from past matches
    PREDICTIVE = "predictive"  # N9: simulated
