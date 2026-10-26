"""A receipt for every number: what anyone needs to re-check it (Day 47, invariant I8)."""

import dataclasses
import json
from dataclasses import dataclass

from cricketmind_v3.domain.subject import CricketSubject, Subject

# the kinds of subject a receipt can name; football's joins it on Day 147
SUBJECTS = {"CricketSubject": CricketSubject}


class ReceiptError(ValueError):
    """A receipt that cannot be read."""


@dataclass(frozen=True, kw_only=True)
class Receipt:
    """Where a number came from, so it can be made again."""

    data: str  # the data snapshot's fingerprint (SHA-256)
    engine: str  # "N8.M1"
    version: str  # the engine's version: it changes when a counting rule changes
    query: str  # the exact question, as one line of text
    value: float  # the number the engine gave
    n: int | None
    interval: tuple[float, float] | None


def query_text(metric: str, subject: Subject, clip_id: str | None) -> str:
    """The question as one line: the same question always gives the same text."""
    fields = {}
    for field in dataclasses.fields(subject):
        value = getattr(subject, field.name)
        if value is not None and value != "":
            fields[field.name] = value
    asked = {
        "metric": metric,
        "subject": {"kind": type(subject).__name__, "fields": fields},
        "clip": clip_id,
    }
    return json.dumps(asked, sort_keys=True, separators=(",", ":"))


def read_query(text: str) -> tuple[str, Subject, str | None]:
    """The question back from its text: metric, subject, clip."""
    try:
        asked = json.loads(text)
        kind = SUBJECTS[asked["subject"]["kind"]]
        subject = kind(**asked["subject"]["fields"])
        return asked["metric"], subject, asked["clip"]
    except (ValueError, KeyError, TypeError) as problem:
        raise ReceiptError(f"cannot read the question: {problem}") from None


def to_line(receipt: Receipt) -> str:
    """A receipt as one line of JSON, for a receipts file."""
    fields = {}
    for field in dataclasses.fields(receipt):
        fields[field.name] = getattr(receipt, field.name)
    return json.dumps(fields, sort_keys=True)


def from_line(line: str) -> Receipt:
    """A receipt back from its line."""
    try:
        fields = json.loads(line)
        if fields["interval"] is not None:
            low, high = fields["interval"]
            fields["interval"] = (low, high)  # JSON keeps a list; a range is a pair
        return Receipt(**fields)
    except (ValueError, KeyError, TypeError) as problem:
        raise ReceiptError(f"cannot read the receipt: {problem}") from None
