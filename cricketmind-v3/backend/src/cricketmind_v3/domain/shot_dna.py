"""Shot DNA (N6): one record of the shot a question is about (Day 60).

Until Phase 8 nothing reads the clip: the label and the posture stay empty, and the
record says why. Later nodes read this record, never a pixel.
"""

from dataclasses import dataclass

from cricketmind_v3.domain.subject import CricketSubject

# why the label and the posture are empty
NO_CLIP = "no clip: a question in words only"
NOT_READ_YET = "the clip is not read yet: CricShot's agent comes in Phase 8 (D-072)"


@dataclass(frozen=True, kw_only=True)
class ShotDNA:
    subject: CricketSubject  # who and what, as N3 read it
    clip: str | None = None  # the clip's path, or None
    label: str | None = None  # the shot the classifier saw, once accepted (Phase 8)
    confidence: float | None = None  # how sure it was of that label
    posture: tuple[tuple[str, float], ...] = ()  # measured from the body (M, Day 126)
    empty_because: str | None = None  # why the label and the posture are empty


def stub_shot_dna(subject: CricketSubject, clip: str | None) -> ShotDNA:
    """The record before any perception exists: the subject, and nothing seen."""
    if clip is None:
        return ShotDNA(subject=subject, empty_because=NO_CLIP)
    return ShotDNA(subject=subject, clip=clip, empty_because=NOT_READ_YET)
