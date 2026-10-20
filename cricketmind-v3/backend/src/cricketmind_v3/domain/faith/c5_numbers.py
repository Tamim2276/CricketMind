from cricketmind_v3.domain.faith.gate import Context, Failure
from cricketmind_v3.domain.findings import Finding, references

# cricket names with digits in them; nobody can get these wrong
ALLOWED = ("t20", "t20s", "t20i", "t20is", "u19", "u19s", "u-19")

# punctuation around a word, taken off before the word is judged
EDGES = ".,;:!?()[]\"'"


def check_c5(finding: Finding, context: Context) -> list[Failure]:
    """C5: apart from claim slots and a few names, the template contains no digits."""
    text = finding.template
    for slot in references(text):
        text = text.replace("{" + slot + "}", " ")  # "c1" has a digit but is no number

    failures = []
    for word in text.split():
        bare = word.strip(EDGES).removesuffix("'s").strip(EDGES)
        if _has_digit(bare) and bare.lower() not in ALLOWED:
            reason = (
                f"the sentence types '{bare}' itself; "
                "every number must come from a claim"
            )
            failures.append(
                Failure(agent=finding.agent, claim="", check="C5", reason=reason)
            )
    return failures


def _has_digit(word: str) -> bool:
    # any digit, in any script: 1, ১ (Bengali), ١ (Arabic)
    for char in word:
        if char.isdigit():
            return True
    return False
