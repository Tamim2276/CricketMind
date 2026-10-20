"""Reading a template: sentences, words and phrases. Used by C3, C4, C6 and C7."""

from collections.abc import Iterable

from cricketmind_v3.domain.claims import Claim
from cricketmind_v3.domain.findings import Finding, references


def sentences(template: str) -> list[str]:
    """Split at '.', '!' or '?' when a space or the end comes next."""
    found = []
    current = ""
    for i, char in enumerate(template):
        current += char
        at_end = i == len(template) - 1
        if char in ".!?" and (at_end or template[i + 1] == " "):
            found.append(current.strip())
            current = ""
    if current.strip():
        found.append(current.strip())  # the last sentence may have no full stop
    return found


def tokens(text: str) -> list[str]:
    """Lower-case words without punctuation: "Kohli's cover-drive!" -> kohli cover drive."""
    words = []
    word = ""
    for char in text.lower().replace("’", "'") + " ":
        if char.isalnum() or char == "'":
            word += char
        elif word:
            cleaned = _clean(word)
            if cleaned:
                words.append(cleaned)
            word = ""
    return words


def _clean(word: str) -> str:
    word = word.removesuffix("'s")  # Kohli's -> kohli
    return word.replace("'", "")  # isn't -> isnt


def find_phrases(words: list[str], phrases: Iterable[str]) -> list[tuple[str, int]]:
    """Each phrase found, with the position of its first word.

    Longer phrases are looked for first, and a word they use is crossed out:
    in "as likely as not", "likely" on its own is not found.
    """
    free = list(words)
    found = []
    for phrase in sorted(phrases, key=_length, reverse=True):
        target = tokens(phrase)
        size = len(target)
        for start in range(len(free) - size + 1):
            if free[start : start + size] == target:
                found.append((phrase, start))
                for i in range(start, start + size):
                    free[i] = ""  # crossed out
    found.sort(key=_position)
    return found


def _length(phrase: str) -> int:
    return len(tokens(phrase))


def _position(item: tuple[str, int]) -> int:
    return item[1]


def has_phrase(text: str, phrases: Iterable[str]) -> bool:
    """Does the text contain any of these phrases, as whole words?"""
    return find_phrases(tokens(text), phrases) != []


def sentence_of(template: str, claim_id: str) -> str | None:
    """The sentence that uses this claim's slot, or None if no sentence does."""
    for sentence in sentences(template):
        if claim_id in references(sentence):
            return sentence
    return None


def claims_in(sentence: str, finding: Finding) -> list[Claim]:
    """The finding's claims whose slots appear in this sentence, in finding order."""
    ids = references(sentence)
    found = []
    for claim in finding.claims:
        if claim.id in ids:
            found.append(claim)
    return found
