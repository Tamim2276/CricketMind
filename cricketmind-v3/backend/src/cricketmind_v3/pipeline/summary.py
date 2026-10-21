"""What the database holds: rows per table, and matches by format (Day 37)."""

# every table the pipeline makes, in the order they are built
TABLES = (
    "deliveries",
    "matches",
    "innings",
    "people",
    "players",
    "player_names",
    "attributes",
    "match_venues",
    "venues",
    "innings_scores",
    "bowler_days",
    "reviews",
    "loaded",
    "problems",
)


def table_rows(db) -> list[tuple[str, int]]:
    """Rows in each table that exists (attributes only after the player file is in)."""
    found = []
    for table in TABLES:
        exists = db.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table],
        ).fetchone()[0]
        if exists:
            rows = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            found.append((table, rows))
    return found


def matches_by_format(db) -> list[tuple]:
    """Matches, balls and dates for each format and gender."""
    return db.execute(
        """
        SELECT m.match_type, m.gender, COUNT(DISTINCT m.match_id) AS matches,
               COUNT(d.match_id) AS balls,
               CAST(MIN(m.date) AS VARCHAR), CAST(MAX(m.date) AS VARCHAR)
        FROM matches m LEFT JOIN deliveries d ON d.match_id = m.match_id
        GROUP BY m.match_type, m.gender
        ORDER BY balls DESC
        """
    ).fetchall()


def summary_report(db, file_bytes: int) -> str:
    sha = db.execute("SELECT zip_sha256 FROM snapshot").fetchone()[0]
    lines = [
        f"Snapshot: `all_json.zip`, SHA-256 `{sha}`",
        f"Database file: {file_bytes:,} bytes",
        "",
        "| Table | Rows |",
        "|---|---|",
    ]
    for table, rows in table_rows(db):
        lines.append(f"| {table} | {rows:,} |")
    lines.append("")
    lines.append("| Format | Gender | Matches | Balls | First | Last |")
    lines.append("|---|---|---|---|---|---|")
    for match_type, gender, matches, balls, first, last in matches_by_format(db):
        lines.append(
            f"| {match_type} | {gender} | {matches:,} | {balls:,} | {first} | {last} |"
        )
    return "\n".join(lines)
