from collections.abc import Callable
from dataclasses import dataclass, field

from cricketmind_v3.domain.facts import Fact
from cricketmind_v3.domain.faith.gazetteer import Entry
from cricketmind_v3.domain.findings import Finding


@dataclass(frozen=True, kw_only=True)
class Failure:
    """One reason a finding cannot be shown yet."""

    agent: str  # "N11b"
    claim: str  # "c1", or "" when the sentence as a whole is at fault
    check: str  # "C2"
    reason: str  # one plain sentence the agent can act on


@dataclass(frozen=True, kw_only=True)
class Context:
    """What the checks may look at besides the finding itself."""

    # agent -> the claim types it may assert (C1)
    allowed: dict[str, tuple[str, ...]] = field(default_factory=dict)
    # claim id -> the engine's answer when asked again; None if it has none (C2)
    facts: dict[str, Fact | None] = field(default_factory=dict)
    # name in a sentence -> what it may stand for (C6)
    names: dict[str, list[Entry]] = field(default_factory=dict)
    # every number the engines gave for this question (v2 mode only)
    surface: tuple[float, ...] = ()


# a check reads a finding and its context, and lists what is wrong
Check = Callable[[Finding, Context], list[Failure]]


def run_checks(
    finding: Finding, context: Context, checks: tuple[Check, ...]
) -> list[Failure]:
    """Run every check in order and keep every failure, so the agent hears them all."""
    failures = []
    for check in checks:
        try:
            failures.extend(check(finding, context))  # add all of this check's failures
        except Exception as error:  # noqa: BLE001 - any crash at all must be caught
            # fail closed: a check that breaks holds the finding back, never lets it pass
            name = check.__name__.removeprefix("check_").upper()  # check_c4 -> C4
            reason = f"the check could not run ({type(error).__name__}: {error})"
            failures.append(
                Failure(agent=finding.agent, claim="", check=name, reason=reason)
            )
    return failures
