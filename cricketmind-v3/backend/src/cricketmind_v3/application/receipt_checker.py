"""The receipt checker: make a number again from its receipt, and compare (Day 48)."""

from cricketmind_v3.application.engine_source import EngineFactSource
from cricketmind_v3.domain.receipts import Receipt, ReceiptError, read_query


def check_receipt(receipt: Receipt, engines: EngineFactSource) -> list[str]:
    """Everything wrong with a receipt, re-run on this data. An empty list: confirmed."""
    if receipt.data != engines.data:
        reason = (
            f"other data: the receipt names {receipt.data[:12]}, "
            f"this database is {engines.data[:12]}"
        )
        return [reason]
    problems = []
    now = engines.version_of(receipt.engine)
    if receipt.version != now:
        problems.append(
            f"{receipt.engine} has changed: the receipt names version "
            f"{receipt.version}, it is now {now}"
        )
    try:
        metric, subject, clip_id = read_query(receipt.query)
    except ReceiptError as problem:
        problems.append(str(problem))
        return problems
    fact = engines.fact(metric, subject, clip_id)
    if fact is None:
        problems.append("the engines give no number for this question")
        return problems
    if fact.receipt.engine != receipt.engine:
        problems.append(
            f"the question is {fact.receipt.engine}'s, not {receipt.engine}'s"
        )
    if fact.value != receipt.value:
        problems.append(
            f"the receipt says {receipt.value}, the engines say {fact.value}"
        )
    if fact.n != receipt.n:
        problems.append(f"the receipt counts {receipt.n}, the engines count {fact.n}")
    if fact.interval != receipt.interval:
        problems.append(
            f"the receipt's range is {receipt.interval}, the engines' is {fact.interval}"
        )
    return problems
