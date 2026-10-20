# Paper outline (living)

The journal is chosen on Day 14, with the supervisor, and confirmed on Day 188. A natural
option is an extended journal
version of the ICCIT paper. Either way, the new paper must **clearly extend** the ICCIT paper
and cite it: the claim contract, uncertainty-faithful checks, real data, measured posture,
the shot bridge and the website are new. That keeps it clear of duplicate publication.

| § | Section | What it argues | Fed by (plan phases, thesis files) | Draft written on |
|---|---|---|---|---|
| — | Abstract | Problem, method, three headline results | Written last | Day 197 |
| 1 | Introduction | LLM agents state numbers they never verify; v2 closed part of the gap; five gaps remain; three contributions (D-026) | v4 doc §1 (and v3 doc §1); E-001–E-003; N-007 | Day 197 |
| 2 | Related work | Numeric verification (PCN, receipts, EG-VAR, VeriFin), faithful uncertainty, tone and framing, abstention, cricket and sports MAS; what stays ours (D-026) | `related_work.md` (systematic search on Day 13), `references.bib` | Day 196 |
| 3 | Architecture and method | Provenance classes, claim contract, registry, Faith Check C1–C7, engines with honest uncertainty, the evidence pipeline, perception (no accuracy) | Phases 1, 2, 4, 5, 8, 9; `decisions.md` | Days 12, 27, 54, 79, 127, 140 (drafts) → 190–191 |
| 4 | Implementation | Clean architecture, the API, the website, the deployment | Phases 6, 7, 11; D-001, D-003, D-013 | Days 91, 110 (drafts) → 192 |
| 5 | Evaluation setup | Questions (the locked exam of Day 15), data, metrics, pass rules, baselines, AI checkers, the three models, CricBench, the human study | Day 150 | 193 |
| 6 | Results | Containment F1–F8 (v2 mode, v3 design without C7, full gate), false rejections, cost of containment, calibration, posture AUC, bridge κ, routing, coverage on the locked exam, baselines, ablation, AI checkers, three models, tricky questions, CricBench, the football case study, receipts, human ratings, cost bounds | E-004–E-022; `numbers.md` | 179 (draft) → 194–195 |
| 7 | Limitations | Honest threats to validity | `limitations.md` | 198 |
| 8 | Conclusion | What v3 guarantees, and what it does not | — | 198 |

## Headline results (fill in as they come)

1. …
2. …
3. …

## Target journal (Day 14)

**Status:** shortlist ready, waiting for the supervisor meeting.
Shortlist and reasons: `drafts/day14_journal_shortlist.md`. Suggested order: Information
Processing & Management, then Expert Systems with Applications, then Knowledge-Based Systems
or Engineering Applications of AI.

| Item | Value |
|---|---|
| Journal (first choice) | … (decided with the supervisor) |
| Backups | … |
| Length limit | … (from the Guide for Authors) |
| How its papers evaluate | … (from two recent papers: baselines, significance tests, user studies, ablation, code and data) |
| New paper or extended ICCIT version | … |
| Ethics approval for the human study | … (needed? applied on …) |
| CricBench early access | … (email sent on …) |
