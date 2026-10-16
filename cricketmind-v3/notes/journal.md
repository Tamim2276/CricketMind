# CricketMind v3 — journal

One entry per day, at the end of the day. Five minutes. Copy the template.

```markdown
## Day N — YYYY-MM-DD — <title from the plan>

**Learned:** one or two sentences, in your own words.
**Built:** what exists now that did not this morning.
**Tests:** how many pass; anything skipped and why.
**Surprise:** one thing that did not go as expected.

**For the paper** (leave a line out if there is nothing):
- Decision: D-0xx in thesis/decisions.md
- Number: N-0xx in thesis/numbers.md
- Evidence: E-0xx in thesis/evidence.md
- Limitation: L-0xx in thesis/limitations.md
- Section it feeds: e.g. Method §3.4
**Commit:** <short hash>
```

---

## Day 0 — 2026-10-15/16 — Designing v3 (done with Claude, before the build starts)

**Learned:** The v2 Faith Check checks *values*, not *claims*. A correct number attached to the
wrong player passes it. Several v2 agents also say things no engine ever computed.

**Built:** The v3 architecture document (`docs/CricketMind_Architecture_v3.pdf`, copied on
Day 1) and this build plan.

**Surprise:** Two worked examples on the v2 synthetic corpus changed conclusions, not just
formatting:
- The running example ("the cover drive is the stronger stroke") has a difference interval
  that contains zero, so v3 would make the agent say "no detectable difference" (N-001, N-002).
- v2's N9 interval only measures Monte Carlo error. With estimation uncertainty it is about
  20 times wider (N-003, N-004).

**For the paper:**
- Decisions: D-001 to D-013 in thesis/decisions.md
- Numbers: N-001 to N-008 in thesis/numbers.md
- Evidence: E-001 to E-003 verified by reading v2 code; E-004 onwards planned
- Limitations: L-001 to L-012
- A first literature search found no 2025–26 multi-agent LLM system for cricket besides
  FanCric (2024). It also found close work on verifying numbers (Proof-Carrying Numbers,
  tool receipts), which the paper must position against (related_work.md).
- Sections it feeds: Introduction (the gaps), Method (all), Related work

---

## Day 1 — 2026-10-16 — The workshop: worktree, environment, first test

**Learned:** A git worktree is a second folder for the same repository with its own branch,
so CricShot and v3 never share a checkout and nobody switches branches. `uv run` always uses
the project's own .venv, so nothing needs activating.
**Built:** The cricketmind-v3 worktree and branch; the backend project (uv, Python 3.12,
pytest, ruff); pytest settings so CricShot's pytest.ini is never used; a .gitignore for v3;
the architecture PDF in docs/.
**Tests:** 1 passed (test_first.py); ruff clean.
**Surprise:** My default Python is 3.10, so uv init needed --python 3.12. Without the
[tool.pytest.ini_options] block, pytest would have climbed up and used CricShot's settings.

**For the paper:** nothing today (setup is process, not method).
