"""N10 chooses who answers (Day 63): a proposal ranks the agents, and fixed rules decide.

The proposal says who should answer, best first. The rules are the same whoever proposed,
so the cap of four and the need for real numbers always hold.
"""

from dataclasses import dataclass

from cricketmind_v3.domain.capabilities import EXPLAINER
from cricketmind_v3.domain.evidence import Evidence
from cricketmind_v3.domain.faith.text import find_phrases, tokens
from cricketmind_v3.domain.readback import subject_words
from cricketmind_v3.domain.roster import NAMES, ROSTER, STAKEHOLDERS

MAX_ACTIVE_AGENTS = 4
# the coach and the player, when nothing points elsewhere
DEFAULT_PAIR = ("N11b", "N11c")

# who proposed
BY_KEYWORDS = "keywords"
BY_MODEL = "model"  # Day 64

# how the chosen agents were chosen
PROPOSED = "proposed"  # those proposed who have numbers
DEFAULT = "default pair"  # nobody proposed has numbers
WHOEVER_CAN = "whoever can"  # the default pair has no numbers either
ONLY_THE_EXPLAINER = "the Explainer"  # a rules question
NOBODY = "nobody"  # no agent has a number for this question

# words that point to an agent, matched as whole words. They only rank: they cannot
# block an agent, and they cannot wake one that has no numbers.
KEYWORDS: dict[str, tuple[str, ...]] = {
    "N11a": (
        "franchise",
        "franchises",
        "board",
        "boards",
        "squad",
        "squads",
        "retain",
        "retention",
        "signing",
        "signings",
        "budget",
        "fixture",
        "fixtures",
    ),
    "N11b": (
        "coach",
        "coaches",
        "tactic",
        "tactics",
        "tactical",
        "plan",
        "plans",
        "strategy",
        "matchup",
        "matchups",
        "match up",
        "match ups",
        "weakness",
        "bowl to",
        "bowl at",
        "field",
    ),
    "N11c": (
        "technique",
        "form",
        "improve",
        "improving",
        "shot selection",
        "drive",
        "pull",
        "sweep",
        "cut",
        "trend",
        "milestone",
    ),
    "N11d": (
        "scout",
        "scouts",
        "scouting",
        "prospect",
        "prospects",
        "recruit",
        "recruits",
        "emerging",
        "talent pool",
    ),
    "N11e": (
        "fan",
        "fans",
        "fantasy",
        "ticket",
        "tickets",
        "highlight",
        "highlights",
        "trivia",
    ),
    "N11f": (
        "curator",
        "curators",
        "groundsman",
        "pitch",
        "pitches",
        "crowd",
        "floodlight",
        "floodlights",
    ),
    "N11g": (
        "bowling",
        "bowler",
        "bowlers",
        "economy",
        "wickets",
        "spell",
        "spells",
        "yorker",
        "yorkers",
        "bouncer",
        "bouncers",
    ),
    "N11h": (
        "commentary",
        "commentator",
        "commentators",
        "broadcast",
        "broadcaster",
        "broadcasters",
        "sponsor",
        "sponsors",
        "sponsorship",
        "brand",
        "graphic",
        "graphics",
        "headline",
        "wagon wheel",
    ),
    "N11i": (
        "umpire",
        "umpires",
        "referee",
        "review",
        "reviews",
        "drs",
        "overturn",
        "overturned",
        "officials",
        "eligibility",
    ),
    "N11j": (
        "academy",
        "academies",
        "drill",
        "drills",
        "youngster",
        "youngsters",
        "junior",
        "juniors",
        "u19",
        "u-19",
        "age group",
        "coaching plan",
    ),
    "N11k": (
        "workload",
        "overworked",
        "overwork",
        "fatigue",
        "tired",
        "burnout",
        "physio",
        "medical",
        "injury",
        "injured",
        "hamstring",
        "fitness",
        "recovery",
    ),
    "N11l": (
        "agent",
        "agents",
        "manager",
        "valuation",
        "worth",
        "market",
        "salary",
        "contract",
        "endorsement",
        "endorsements",
    ),
}


@dataclass(frozen=True, kw_only=True)
class Routing:
    """What N10 decided, and why, so a run can be read back and measured."""

    chosen: tuple[str, ...]  # who answers, best first, at most four
    how: str  # PROPOSED, DEFAULT, WHOEVER_CAN, ONLY_THE_EXPLAINER or NOBODY
    by: str  # who proposed
    proposed: tuple[str, ...] = ()  # the whole proposal, before the rules
    no_number: tuple[str, ...] = ()  # proposed, but with no number for this question
    reason: str | None = None  # the model's reason, for us: no answer shows it (Day 64)
    note: str | None = None  # why the words proposed when a model was there (Day 64)


def propose_by_keywords(question: str) -> list[str]:
    """The agents the question's words point to: the most different words first, and a
    tie in the roster's order, so a run can be repeated."""
    words = tokens(question)
    scored = []
    for order, agent in enumerate(ROSTER):
        hits = set()
        for phrase, _ in find_phrases(words, KEYWORDS.get(agent, ())):
            hits.add(phrase)
        if hits:
            scored.append((-len(hits), order, agent))
    scored.sort()
    ranked = []
    for _, _, agent in scored:
        ranked.append(agent)
    return ranked


def decide(
    proposed: list[str],
    capable: dict[str, tuple[str, ...]],
    by: str,
    reason: str | None = None,
    note: str | None = None,
) -> Routing:
    """The fixed rules: only agents with numbers, at most four, the default pair when
    nothing else is left, and nobody when no agent has a number."""
    allowed = []
    no_number = []
    for agent in proposed:
        if agent in capable:
            if agent not in allowed:  # a model may name an agent twice
                allowed.append(agent)
        elif agent not in no_number:
            no_number.append(agent)

    if EXPLAINER in capable:
        chosen = [EXPLAINER]  # alone, and it counts as one of the four
        how = ONLY_THE_EXPLAINER
        no_number = []  # a rule wants words: no number is missing
    elif not capable:
        chosen = []
        how = NOBODY
    elif allowed:
        chosen = allowed
        how = PROPOSED
    else:
        chosen = _default_pair(capable)
        how = DEFAULT
        if not chosen:
            chosen = list(capable)  # the roster's order
            how = WHOEVER_CAN

    return Routing(
        chosen=tuple(chosen[:MAX_ACTIVE_AGENTS]),
        how=how,
        by=by,
        proposed=tuple(proposed),
        no_number=tuple(no_number),
        reason=reason,
        note=note,
    )


def _default_pair(capable: dict[str, tuple[str, ...]]) -> list[str]:
    # only those of the pair with numbers: "Bumrah in T20" has none for either
    found = []
    for agent in DEFAULT_PAIR:
        if agent in capable:
            found.append(agent)
    return found


def evidence_summary(
    evidence: tuple[Evidence, ...], capable: dict[str, tuple[str, ...]]
) -> str:
    """What the model is told (Day 64): how many balls each side has, and which measures
    each agent has numbers for. No number from an engine is in it."""
    lines = ["Balls found:"]
    for found in evidence:
        words = ", ".join(subject_words(found.subject)) or "the question"
        if found.none_because is None:
            lines.append(f"- {words}: {found.count}")
        else:
            lines.append(f"- {words}: none ({found.none_because})")
    lines.append("Specialists:")
    for agent in KEYWORDS:
        has = ", ".join(capable.get(agent, ())) or "no numbers"
        lines.append(f"- {agent} {NAMES[agent]}, for {STAKEHOLDERS[agent]}: {has}")
    return "\n".join(lines)
