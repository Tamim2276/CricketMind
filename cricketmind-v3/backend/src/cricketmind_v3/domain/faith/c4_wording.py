from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.gate import Context, Failure
from cricketmind_v3.domain.faith.text import (
    claims_in,
    find_phrases,
    has_phrase,
    sentences,
    tokens,
)
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.provenance import Provenance
from cricketmind_v3.domain.registry import lookup
from cricketmind_v3.domain.render import shown
from cricketmind_v3.domain.wording import (
    HEDGES,
    NEVER,
    band_for,
    fits,
    needs_hedge,
    vocabulary,
)


def check_c4(finding: Finding, context: Context) -> list[Failure]:
    """C4: do the words match the uncertainty?"""
    failures = []
    for sentence in sentences(finding.template):
        failures.extend(_chances(finding, sentence))
        failures.extend(_differences(finding, sentence))
    return failures


# Part 1: a chance (Day 21)


def _chances(finding: Finding, sentence: str) -> list[Failure]:
    """Words about a chance must fit its band, and a wide range must be admitted."""
    failures = []
    chances = []
    for claim in claims_in(sentence, finding):
        if not _is_chance(claim):
            continue
        if _impossible(claim):
            # report it, never crash: a gate must say what is wrong
            reason = "a chance and its range must lie between 0% and 100%"
            failures.append(_fail(finding, claim.id, reason))
        else:
            chances.append(claim)
    if not chances:
        return failures

    blamed = chances[0].id if len(chances) == 1 else ""
    for phrase, _ in find_phrases(tokens(sentence), vocabulary()):
        if phrase in NEVER:
            reason = f"'{phrase}' is too strong for any chance a model gives"
            failures.append(_fail(finding, blamed, reason))
        elif not _fits_any(phrase, chances):
            reason = f"'{phrase}' does not fit " + _describe(chances)
            failures.append(_fail(finding, blamed, reason))

    for claim in chances:
        if needs_hedge(claim.interval) and not has_phrase(sentence, HEDGES):
            low, high = claim.interval
            reason = (
                f"the range {shown(claim.metric, low)} to {shown(claim.metric, high)} "
                "spans more than one band; admit it, e.g. 'the estimate is very uncertain'"
            )
            failures.append(_fail(finding, claim.id, reason))
    return failures


def _is_chance(claim: Claim) -> bool:
    # a predicted chance; a difference between two chances is part 2
    if claim.kind != Provenance.PREDICTIVE:
        return False
    claim_type = lookup(claim.metric)
    return claim_type is not None and not claim_type.comparison


def _impossible(claim: Claim) -> bool:
    low, high = claim.interval
    for p in (claim.value, low, high):
        if p < 0 or p > 1:
            return True
    return False


def _fits_any(phrase: str, chances: list[Claim]) -> bool:
    # with two chances in one sentence, a word may describe either
    for claim in chances:
        if fits(phrase, band_for(claim.value)):
            return True
    return False


def _describe(chances: list[Claim]) -> str:
    # "a 41% chance, which is 'about even'"
    parts = []
    for claim in chances:
        band = band_for(claim.value)
        parts.append(
            f"a {shown(claim.metric, claim.value)} chance, which is '{band.name}'"
        )
    return " or ".join(parts)


# Part 2: a difference (Day 22)

# saying honestly that a difference may be nothing at all
NO_DIFFERENCE = (
    "no detectable difference",
    "no clear difference",
    "no real difference",
    "not clearly different",
    "too close to call",
    "cannot be separated",
    "can't be separated",
)

# words that declare one side ahead
VERDICTS = (
    "stronger",
    "weaker",
    "better",
    "worse",
    "superior",
    "inferior",
    "outperforms",
    "outscores",
    "more effective",
    "less effective",
)


def _differences(finding: Finding, sentence: str) -> list[Failure]:
    """A difference whose range includes zero must say so, and declare no winner."""
    failures = []
    for claim in claims_in(sentence, finding):
        claim_type = lookup(claim.metric)
        if claim_type is None or not claim_type.comparison or claim.interval is None:
            continue  # C1 reports a comparison without its range
        low, high = claim.interval
        span = f"{shown(claim.metric, low)} to {shown(claim.metric, high)}"
        says_none = has_phrase(sentence, NO_DIFFERENCE)

        if low <= 0 <= high:
            if not says_none:
                reason = (
                    f"the range {span} includes zero, so there may be no difference "
                    "at all; say 'no detectable difference'"
                )
                failures.append(_fail(finding, claim.id, reason))
            for phrase, _ in find_phrases(tokens(sentence), VERDICTS):
                reason = (
                    f"'{phrase}' claims a difference the range {span} does not show"
                )
                failures.append(_fail(finding, claim.id, reason))
        elif says_none:
            reason = (
                f"the range {span} does not include zero: the difference is "
                "detectable, so do not say there is none"
            )
            failures.append(_fail(finding, claim.id, reason))
    return failures


def _fail(finding: Finding, claim_id: str, reason: str) -> Failure:
    return Failure(agent=finding.agent, claim=claim_id, check="C4", reason=reason)
