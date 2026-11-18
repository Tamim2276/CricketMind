# CLAUDE.md

**Cricket MLOps:** the chance each side wins a men's T20 match, updated after every ball, built
the way production machine learning is built. The research part is the season replay: when the
game changes, does a pipeline that detects drift and retrains keep the model honest better than
one left alone or retrained on a timetable? We copy the GreenAirOps paper's pipeline shape and
add the experiment that paper lacks.

- Brief: `docs/PROJECT_BRIEF.md`. Plan, day by day: `docs/BUILD_PLAN.md`.
- Paper: `docs/papers/greenairops.pdf` (CC BY-NC-ND 4.0: cite it, never copy its text or figures).
- Settled 2026-11-18: a separate paper; deployed on AWS; men's T20, all leagues, with the IPL as
  the main case; the GitHub repo is public. Open points are listed at the end of the plan.

## Who the user is

A CSE student at MIST, Bangladesh. A beginner, who learns by building the real project and wants
to understand every piece of its code.

## How to work

- Explain in very simple language, short, with an example.
- Write the code one small piece at a time, and explain every line in simple words before it
  runs. The user reads, asks, then says "next".
- No practice exercises: teach each idea on the real code the first time it is needed.
- The user runs commits, installs, downloads and long runs; give the exact commands. Claude may
  write project files, run tests and do quick read-only checks.
- After every finished step, hand over a commit command. No `Co-Authored-By` or any other
  attribution line in commits.
- Commit only this folder's work, path-limited (`git add <paths>`). Never `git add -A`. Never
  switch branches.
- Do every plan step. Never suggest skipping or deferring one to save effort.
- Power cuts happen (load shedding): anything long must be resumable, and save as it goes.
- One GPU user at a time: CricShot may be training. Tree models run on the CPU, so this project
  should rarely need the GPU. If it does, code runs unchanged on Intel Arc (XPU), then CUDA,
  then CPU.

## Code style

- Plain, readable Python: loops and if/else, not comprehensions, lambdas or clever one-liners.
- Comments very short and human. Long explanations go in chat, not in the code.
- Tests with pytest; lint and format with ruff; the environment with uv (Python 3.12):
  `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`

## Keep the projects apart

- Never edit, import from or read the code of **CricketMind v3** (branch `cricketmind-v3`,
  `D:\Cricket\cricketmind-v3-work`) or **CricShot** (`D:\Cricket\cricshot-improve`, branch
  `langGraph`). Each has its own chat.
- Download our own copy of the cricket data (Cricsheet). Do not read CricketMind's database.
- CricketMind's results never appear in this project's paper, and this project's results never
  appear in CricketMind's. CricShot's classifier accuracy appears in neither.
- The GitHub repo `Tamim2276/CricketMind` (public) also holds those projects' branches. Push
  only this branch: `git push origin cricket-mlops`.

## Research rules that must not break

From the brief, Sections 4 and 5.

- A feature is only what is known at that ball: never the final score, the result, or anything
  from later in the match or a later match. No "season" or "era" feature, on purpose.
- Split by season, in time order. Never by random balls.
- Judge a probability by Brier score and log loss first, then calibration. Accuracy at 50% is
  reported, never the main number.
- Credit Cricsheet (Open Data Commons Attribution License).
- Numbers found on the web (brief, Section 5) are leads to check, never results.

## Keep the record as you go

- `notes/journal.md`: one entry per day.
- `docs/decisions.md`: each choice, why, and what else was considered.
- `docs/numbers.md`: no number in the paper without a row here saying how it was made.
- `docs/limitations.md`: each weakness, the day it is noticed.
- Any test rule (what counts as better) is written down **before** the run it judges.
