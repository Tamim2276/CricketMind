"""The data's fingerprint, named by every receipt (Day 47)."""

import hashlib


def data_fingerprint(db) -> str:
    """One SHA-256 for everything a number is counted from.

    The Cricsheet zip's own SHA-256 (D-032) covers every ball, name and result, and
    the tables made from them. Batting hands and bowler types come from outside it
    (cricketdata and hand labels, D-037), so every row of those goes in too.
    """
    digest = hashlib.sha256()
    for (zip_sha256,) in db.execute("SELECT zip_sha256 FROM snapshot").fetchall():
        digest.update(zip_sha256.encode("utf-8"))
    rows = db.execute(
        "SELECT person_id, batting_hand, bowler_type FROM attributes ORDER BY person_id"
    ).fetchall()
    for person_id, batting_hand, bowler_type in rows:
        line = f"{person_id}\t{batting_hand}\t{bowler_type}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()
