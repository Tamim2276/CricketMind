# CricketMind v3: brief for the supervisor meeting (Day 14)

*Prepared 2026-10-21. One page, plus the journal shortlist on the next page.*

## The project in three sentences

CricketMind lets several language-model agents answer cricket questions for coaches, analysts
and fans, but **no agent ever types a number**. Every number is a checked *claim*, computed by
an engine from real ball-by-ball data (Cricsheet), and a deterministic gate (the Faith Check,
C1–C7) checks each claim before release: is it on the menu of allowed claims, does the engine
agree for *this* player, shot and phase, and do the words fit the uncertainty? v3 extends our
ICCIT paper (v2, under review), whose gate checked values but not the claims around them.

## Status (Day 13 of a 202-day plan)

- **Built and tested (83 unit tests):** the claim language (provenance classes, subjects,
  self-checking claims, findings with slots, the renderer) and the registry of claim types
  with a start-up check that refuses claim types no engine computes or the renderer cannot
  show.
- **Done:** a systematic literature search (16 fixed queries; `thesis/related_work.md`).
- **Design:** `docs/CricketMind_Architecture_v4.pdf` (17 pages).

## What the paper will claim (narrowed after the search, D-026)

1. **A subject-keyed re-query for multi-filter statistics.** Each number is re-computed for
   exactly the subject it is about (player × shot × phase × bowler type …), and the names in
   the prose must match. Close work (VeriFin, RAGShield) binds single values to an entity.
2. **Uncertainty wording gated against an external interval.** "Will probably get him out"
   is refused when the engine's chance is 41% [16%, 70%]. A 2026 study *measured* that models
   choose miscalibrated words for such predictions; no system found enforces it.
3. **Judgments as checked comparisons.** "Kohli struggles against spin" needs a checked
   comparison pointing the same way. Thomson and Reiter (INLG 2020) found such words wrong
   in sports summaries by hand; no system found gates them.

## Planned evaluation

A locked 120-question exam written before the system is built (SHA-256 recorded); baselines;
an ablation; two AI checkers (an LLM judge, Chain-of-Verification); three language models;
CricBench's questions if released; a football case study (same checks, new sport); a human
study; confidence intervals and significance tests throughout.

## Decisions I need from you

1. **Target journal.** My shortlist is on the next page; I suggest *Information Processing &
   Management* first. Which do you prefer, and what are the backups?
2. **New paper or extended ICCIT version?** If extended, what does ICCIT's publisher require
   (new-content share, disclosure)?
3. **Ethics approval.** The human study (about Day 187) asks at least 10 cricket fans,
   players or coaches to rate anonymised answers from three systems, with a consent form. Does
   MIST require approval for this? If so, I will apply this week.
4. **CricBench data.** May I email the authors for early access to their questions (draft
   ready), and may I copy you?
5. **Authorship and timeline:** who is on the paper, and are you happy with submission after
   Day 217?
