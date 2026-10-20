from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.gate import Context, Failure
from cricketmind_v3.domain.faith.text import claims_in, find_phrases, sentences, tokens
from cricketmind_v3.domain.findings import Finding
from cricketmind_v3.domain.registry import lookup
from cricketmind_v3.domain.render import shown

# tone words that say a player does better (D-017)
UP = (
    "strong",
    "stronger",
    "strongest",
    "better",
    "dominant",
    "dominates",
    "dominated",
    "excellent",
    "excels",
    "outstanding",
    "impressive",
    "superb",
    "brilliant",
    "thrives",
    "prolific",
    "commanding",
    "in form",
    "in good form",
)

# tone words that say a player does worse
DOWN = (
    "struggles",
    "struggled",
    "struggling",
    "weak",
    "weaker",
    "weakest",
    "weakness",
    "worse",
    "vulnerable",
    "poor",
    "troubled",
    "fragile",
    "susceptible",
    "liability",
    "falters",
    "uncomfortable",
    "out of form",
)

# a word just before a tone word turns it around: "not strong" says worse
NEGATIONS = ("not", "never", "hardly", "isnt", "arent", "wasnt", "werent", "no")


def check_c7(finding: Finding, context: Context) -> list[Failure]:
    """C7: every tone word needs a comparison in its sentence that points its way."""
    failures = []
    for sentence in sentences(finding.template):
        words = tokens(sentence)
        comparisons = _comparisons(claims_in(sentence, finding))
        for phrase, start in find_phrases(words, UP + DOWN):
            better = phrase in UP
            if _negated(words, start):
                better = not better
            if not _backed(better, comparisons):
                failures.append(_why(finding, phrase, better, comparisons))
    return failures


def _comparisons(claims: list[Claim]) -> list[Claim]:
    # only a checked comparison with its range can back a judgment
    found = []
    for claim in claims:
        claim_type = lookup(claim.metric)
        if claim_type is not None and claim_type.comparison and claim.interval:
            found.append(claim)
    return found


def _negated(words: list[str], start: int) -> bool:
    # look one and two words back: "not strong", "not very strong"
    for i in (start - 1, start - 2):
        if i >= 0 and words[i] in NEGATIONS:
            return True
    return False


def _backed(better: bool, comparisons: list[Claim]) -> bool:
    # the whole range must sit on the word's side of zero
    for claim in comparisons:
        low, high = claim.interval
        if better and low > 0:
            return True
        if not better and high < 0:
            return True
    return False


def _why(
    finding: Finding, phrase: str, better: bool, comparisons: list[Claim]
) -> Failure:
    says = "better" if better else "worse"
    if not comparisons:
        reason = (
            f"'{phrase}' is a judgment with no comparison behind it; back it with "
            "one, such as sr_vs_average, in the same sentence"
        )
        return _fail(finding, "", reason)

    claim = comparisons[0]
    low, high = claim.interval
    span = f"{shown(claim.metric, low)} to {shown(claim.metric, high)}"
    if low <= 0 <= high:
        reason = (
            f"'{phrase}' says {says}, but the range {span} includes zero: "
            "it may be no difference at all"
        )
    else:
        other = "worse" if better else "better"
        reason = (
            f"'{phrase}' says {says}, but the comparison says {other}: range {span}"
        )
    return _fail(finding, claim.id, reason)


def _fail(finding: Finding, claim_id: str, reason: str) -> Failure:
    return Failure(agent=finding.agent, claim=claim_id, check="C7", reason=reason)
