from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.gate import Context, Failure
from cricketmind_v3.domain.faith.gazetteer import Entry
from cricketmind_v3.domain.faith.text import claims_in, find_phrases, sentences, tokens
from cricketmind_v3.domain.findings import Finding


def check_c6(finding: Finding, context: Context) -> list[Failure]:
    """C6: every name in a sentence with a claim must agree with that claim's subject."""
    failures = []
    for sentence in sentences(finding.template):
        claims = claims_in(sentence, finding)
        if not claims:
            continue  # no number here to give to the wrong player
        blamed = claims[0].id if len(claims) == 1 else ""
        for phrase, _ in find_phrases(tokens(sentence), context.names):
            entries = context.names[phrase]
            if not _agrees(entries, claims):
                about = _about(entries, claims)
                reason = f"the sentence names '{phrase}', but {about}"
                failures.append(_fail(finding, blamed, reason))
    return failures


def _agrees(entries: list[Entry], claims: list[Claim]) -> bool:
    # the name fits if it fills a matching field of any claim in the sentence
    for entry in entries:
        for claim in claims:
            for field in entry.fields:
                if getattr(claim.subject, field, None) == entry.value:
                    return True
    return False


def _about(entries: list[Entry], claims: list[Claim]) -> str:
    # "c1 is about player V Kohli", or "c1 is about every shot"
    fields = entries[0].fields
    parts = []
    for claim in claims:
        values = []
        for field in fields:
            value = getattr(claim.subject, field, None)
            if value is not None:
                values.append(f"{field} {value}")
        if values:
            parts.append(f"{claim.id} is about " + " and ".join(values))
        else:
            parts.append(f"{claim.id} is about every {fields[0]}")
    return "; ".join(parts)


def _fail(finding: Finding, claim_id: str, reason: str) -> Failure:
    return Failure(agent=finding.agent, claim=claim_id, check="C6", reason=reason)
