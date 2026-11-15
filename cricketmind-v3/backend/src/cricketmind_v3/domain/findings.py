from dataclasses import dataclass

from cricketmind_v3.domain.claims import Claim


class FindingError(ValueError):
    """A finding's sentence and its claims do not match."""


def references(template: str) -> list[str]:
    """The slots in a sentence, in order: '{c1} and {c2}' gives ['c1', 'c2']."""
    found = []
    start = 0
    while True:
        open_at = template.find("{", start)
        if open_at == -1:
            return found  # no more slots
        close_at = template.find("}", open_at)
        if close_at == -1:
            raise FindingError("a '{' at position " + str(open_at) + " is never closed")
        found.append(template[open_at + 1 : close_at])
        start = close_at + 1


@dataclass(frozen=True, kw_only=True)
class Finding:
    """What an agent hands in: its claims and one sentence with slots."""

    agent: str  # "N11b"
    claims: tuple[Claim, ...]
    template: str  # "Kohli's cover drive returns {c1}."
    # why the model's reply could not be read (Day 66): nothing to show, and C1 says why
    unreadable: str | None = None

    def __post_init__(self):
        # rule 1: a tuple cannot grow, so no claim can be slipped in later
        if not isinstance(self.claims, tuple):
            raise FindingError(f"{self.agent}: claims must be a tuple, not a list")

        # rule 2: no two claims share an id
        ids = []
        for claim in self.claims:
            if claim.id in ids:
                raise FindingError(f"{self.agent}: two claims are called {claim.id}")
            ids.append(claim.id)

        # rule 3: every slot names one of this finding's claims
        for slot in references(self.template):
            if slot not in ids:
                raise FindingError(
                    f"{self.agent}: the sentence uses {slot}, but no claim has that id"
                )


def latest_findings(findings: list[Finding]) -> list[Finding]:
    """Each agent's last try: every try is kept in the notebook, the last one counts."""
    latest = {}
    for finding in findings:
        latest[finding.agent] = finding
    return list(latest.values())
