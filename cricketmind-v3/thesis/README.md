# thesis/ — everything the CricketMind v3 paper will need

The paper is written at the end (Phase 12), but its raw material is made **every day**.
If something is not written down on the day it happens, it is forgotten: why a choice was
made, which seed made a figure, which commit produced a number.

## What goes where

| File / folder | What goes in it | When you add to it |
|---|---|---|
| `../notes/journal.md` | One short entry per day (template inside) | End of every day |
| `decisions.md` | Every design decision: what, why, what else you considered | When you choose between options |
| `numbers.md` | Every number the paper might print, with where it came from | When a script prints a number worth keeping |
| `evidence.md` | Every sentence the paper will claim, and the file that proves it | When a claim becomes provable, or is planned |
| `limitations.md` | Every weakness you notice | The moment you notice it |
| `references.bib` | Every paper you cite (BibTeX) | When you read a paper |
| `related_work.md` | Notes per paper: what they did, how CricketMind differs | When you read a paper |
| `outline.md` | The paper skeleton, and what feeds each section | On review days |
| `results/` | CSV/JSON written **by scripts**, plus `ledger.jsonl` (one row per run) | Automatically, from Day 23 |
| `figures/` | PNG/PDF made **by scripts** in `scripts/` | When you make a figure |
| `scripts/` | One script per figure or table | Same |
| `drafts/` | Rough section drafts, written on review days | Review days |

## Three rules

1. **No number in the paper unless it is in `numbers.md` with a source.** CricketMind refuses
   to print a number nobody computed. The paper obeys the same rule.
2. **Every figure and table can be rebuilt by running one script.** Never edit a figure by
   hand.
3. **Synthetic results are labelled synthetic,** and **the CricShot classifier's accuracy never
   appears here.** It belongs to the separate CricShot paper.
