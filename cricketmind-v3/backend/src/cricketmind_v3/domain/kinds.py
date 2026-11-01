"""The five kinds of question (Day 59), and how the rules tell them apart when no model
reads: a clip makes it video; words about injuries or money make it something nobody
can check; "will" or "chance" a prediction; a question about the laws, with nobody in
it, the rules; anything else, the numbers of what happened.
"""

from cricketmind_v3.domain.faith.text import tokens
from cricketmind_v3.domain.subject import CricketSubject

KINDS = ("stats", "prediction", "video", "rules", "cant_check")

# what no ball-by-ball record holds
CANT_CHECK_WORDS = (
    "injured",
    "injury",
    "injuries",
    "hurt",
    "fitness",
    "salary",
    "salaries",
    "paid",
    "earn",
    "earns",
    "contract",
    "auction",
    "wife",
    "girlfriend",
    "family",
    "religion",
    "retire",
    "retirement",
    "feel",
    "feels",
    "angry",
    "upset",
)
PREDICTION_WORDS = ("will", "chance", "chances", "likely", "predict", "within", "next")
LAW_WORDS = (
    "lbw",
    "law",
    "laws",
    "rule",
    "rules",
    "wide",
    "noball",
    "no",  # "no ball", "no-ball"
    "free",  # "free hit"
    "drs",
    "dls",
    "duckworth",
    "followon",
    "follow",
    "mean",
    "means",
    "explain",
)


def kind_by_rules(question: str, subject: CricketSubject, has_clip: bool) -> str:
    """Which of the five kinds a question is, by its words alone."""
    if has_clip:
        return "video"
    words = tokens(question)
    for word in words:
        if word in CANT_CHECK_WORDS:
            return "cant_check"
    nobody = (
        subject.player is None
        and subject.bowler is None
        and subject.team is None
        and subject.venue is None
    )
    if nobody:
        for word in words:
            if word in LAW_WORDS:
                return "rules"
    for word in words:
        if word in PREDICTION_WORDS:
            return "prediction"
    return "stats"
