from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.faith.gate import Context, Failure
from cricketmind_v3.domain.faith.gazetteer import EVERYDAY, Entry
from cricketmind_v3.domain.faith.text import (
    claims_in,
    find_phrases,
    sentences,
    tokens,
    written_tokens,
)
from cricketmind_v3.domain.findings import Finding


def check_c6(finding: Finding, context: Context) -> list[Failure]:
    """C6: every name in a sentence with a claim must agree with that claim's subject. An
    agent with no measures (the Explainer) may name no player at all (Day 72)."""
    failures = []
    words_only = context.allowed.get(finding.agent) == ()
    for sentence in sentences(finding.template):
        claims = claims_in(sentence, finding)
        if not claims and not words_only:
            continue  # no number here to give to the wrong player
        blamed = claims[0].id if len(claims) == 1 else ""
        written = written_tokens(sentence)
        for phrase, start in find_phrases(tokens(sentence), context.names):
            entries = _meant(context.names[phrase], phrase, written, start)
            if not entries:
                continue
            if words_only:
                if _a_person(entries):
                    reason = f"an explanation may not name a player: '{phrase}'"
                    failures.append(_fail(finding, "", reason))
            elif not _agrees(entries, claims):
                about = _about(entries, claims)
                reason = f"the sentence names '{phrase}', but {about}"
                failures.append(_fail(finding, blamed, reason))
    return failures


def _meant(
    entries: list[Entry], phrase: str, written: list[str], start: int
) -> list[Entry]:
    """What a phrase stands for here. Everyday words are also people: "ball" is JT Ball and
    "to" is YS To. So a name counts only when it is written as one, with a capital; at the
    start of a sentence, where every word has one, an everyday word is the word (found on
    Day 66). Teams and grounds too: "the heat" is not the Brisbane Heat."""
    others = []
    for entry in entries:
        if not _a_proper_name(entry):
            others.append(entry)
    if len(others) == len(entries):
        return entries  # a shot or a phase: no capital needed
    last_word = written[start + len(tokens(phrase)) - 1]
    if not last_word[0].isupper():
        return others
    if start == 0 and phrase in EVERYDAY:
        return others
    return entries


def _a_proper_name(entry: Entry) -> bool:
    for field in ("player", "team", "venue"):
        if field in entry.fields:
            return True
    return False


def _a_person(entries: list[Entry]) -> bool:
    for entry in entries:
        if "player" in entry.fields:
            return True
    return False


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
