from dataclasses import dataclass


class WordingError(ValueError):
    """A chance outside 0 to 1."""


@dataclass(frozen=True, kw_only=True)
class Band:
    """One of five ranges of chance, with the words that fit it."""

    name: str  # "about even"
    words: tuple[str, ...]  # phrases that fit this band
    forbidden: tuple[str, ...]  # phrases that overstate or understate it


# after Kent (1964), his ranges closed into contiguous intervals (D-006, D-029)
BANDS = (
    Band(
        name="almost certainly not",
        words=("almost certainly not", "remote", "very unlikely", "highly unlikely"),
        forbidden=("likely", "probable", "possible"),
    ),
    Band(
        name="probably not",
        words=("probably not", "unlikely"),
        forbidden=("likely", "probable", "expected"),
    ),
    Band(
        name="about even",
        words=(
            "about even",
            "as likely as not",
            "even chance",
            "fifty-fifty",
            "toss-up",
        ),
        forbidden=("likely", "unlikely"),
    ),
    Band(
        name="probable",
        words=("probable", "probably", "likely"),
        forbidden=("almost certain", "will"),
    ),
    Band(
        name="almost certain",
        words=("almost certain", "very likely", "highly likely"),
        forbidden=("certain", "guaranteed"),
    ),
)

# too strong for any chance a model gives, whatever the band
NEVER = ("will", "certain", "guaranteed", "always", "never", "definitely", "surely")

# admitting that the range is wide; no digits, so they pass C5
HEDGES = (
    "uncertain",
    "on limited evidence",
    "on a small sample",
    "on few dismissals",
    "a rough estimate",
    "hard to pin down",
    "the range is wide",
)


def band_for(p: float) -> Band:
    """The band a chance falls in. Edges: 0.13 and 0.87 go outward, 0.40 and 0.60 too."""
    if p < 0 or p > 1:
        raise WordingError(f"{p} is not a chance between 0 and 1")
    if p <= 0.13:
        return BANDS[0]
    if p <= 0.40:
        return BANDS[1]
    if p < 0.60:
        return BANDS[2]
    if p < 0.87:
        return BANDS[3]
    return BANDS[4]


def needs_hedge(interval: tuple[float, float]) -> bool:
    """True when the range reaches into more than one band."""
    low, high = interval
    return band_for(low) is not band_for(high)


def fits(phrase: str, band: Band) -> bool:
    """Can this phrase describe a chance in this band?"""
    if phrase in NEVER or phrase in band.forbidden:
        return False
    if phrase in band.words:
        return True
    for other in BANDS:
        if phrase in other.words:
            return False  # it belongs to another band
    return True  # a word that says nothing about size, like "possible" in the middle


def vocabulary() -> list[str]:
    """Every phrase that says something about a chance."""
    phrases = []
    for band in BANDS:
        for phrase in band.words + band.forbidden:
            if phrase not in phrases:
                phrases.append(phrase)
    for phrase in NEVER:
        if phrase not in phrases:
            phrases.append(phrase)
    return phrases
