# CricketMind v3: rules for this chat

The user is a CSE student at MIST, Bangladesh. CricketMind v3 is their thesis
project, built from scratch to learn. The plan is `docs/BUILD_PLAN.md`; work one
day at a time: understand, learn, build, test, repeat.

## Where things live

- This folder is a git worktree on the `cricketmind-v3` branch:
  `D:\Cricket\cricketmind-v3-work\cricketmind-v3`.
- Only files inside `cricketmind-v3/` change here. The worktree root also holds a
  copy of CricShot's files: never edit them.
- Commit from the worktree root, path-limited:
  `cd /d/Cricket/cricketmind-v3-work && git add cricketmind-v3 && git commit -m "day N: ..."`.
  Never `git add -A`. Never switch branches.
- CricShot (the classifier) lives in `D:\Cricket\cricshot-improve` on `langGraph`, in a
  different chat. Never touch it from here. v3 never imports CricShot code; the
  only bridge is the exported model file (Day 112).
- `cricketmind/` (no suffix) is the old v2 prototype: reference only, never edit.

## How to work with this user

- Explain in very simple language, short, with an example. They are a beginner.
- Give commands; the user runs them. That covers commits, installs, long runs and
  downloads. Claude may run tests and quick read-only checks.
- After every finished step, hand over a commit command. No `Co-Authored-By` or any
  attribution line in commits.
- Do every plan step. Never suggest skipping or deferring one to save effort.
- Power cuts happen (load shedding): anything long must be resumable.

## Code style

- Plain, readable Python: loops and if/else, not comprehensions or clever one-liners.
- Comments very short and human. Long explanations go in chat, not in the code.
- GPU code runs unchanged on Intel Arc (XPU), then CUDA, then CPU.

## Thesis tracking (every step)

- With each commit command, also give the `notes/journal.md` entry and any new rows
  for `thesis/decisions.md`, `numbers.md`, `evidence.md`, `limitations.md`.
- No number in the paper without a row in `numbers.md`; no claim without a row in
  `evidence.md`. Synthetic results are labelled synthetic.
- The CricShot classifier's accuracy never appears in the v3 paper, website or slides.
  It belongs to the separate CricShot paper.
- The paper must cite the ICCIT (v2) paper and clearly extend it.
