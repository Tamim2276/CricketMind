# CricketMind v3 — Build Plan (from scratch, beginner-friendly)

This plan builds **CricketMind v3** (the design in `CricketMind_Architecture_v3.pdf`) as a
real website, from an empty folder, one small piece at a time.

**It is part of the thesis.** A paper is written at the end (Phase 12), and everything it
needs is recorded as you go, in `thesis/` and `notes/journal.md` (Section 9).

**Where it lives:** `D:\Cricket\cricketmind-v3-work\cricketmind-v3\`, a git worktree on its
own branch, `cricketmind-v3` (Section 3). Not to be confused with
`cricshot-improve/cricketmind/`, the old v2 prototype.

Every day follows the same loop: **understand → learn → build → test → commit.**
Nothing is copied from the old notebook or the old `cricketmind/` prototype. You may *look*
at them to remember an idea, but you type everything yourself. That is how it sticks.

---

## Contents

1. [The big picture, in simple words](#1-the-big-picture-in-simple-words)
2. [How long will it take? (worst case)](#2-how-long-will-it-take-worst-case)
3. [Keeping the two projects apart](#3-keeping-the-two-projects-apart)
4. [The daily loop](#4-the-daily-loop)
5. [The tech stack (and the frontend choice)](#5-the-tech-stack)
6. [Clean architecture, in simple words](#6-clean-architecture-in-simple-words)
7. [How we test](#7-how-we-test)
8. [Rules that never change](#8-rules-that-never-change)
9. [The thesis paper: track everything](#9-the-thesis-paper-track-everything)
10. [Milestones](#10-milestones)
11. [Day-by-day plan](#11-day-by-day-plan) — Phases 0 to 12, Days 1 to 177
12. [When you get stuck](#12-when-you-get-stuck)
13. [Glossary](#13-glossary)

---

## 1. The big picture, in simple words

CricketMind answers cricket questions for twelve kinds of people — coaches, players,
scouts, fans, medical staff and so on — and **every number it says has been checked.**

Think of it as a newsroom:

- **Reporters** (the 12 agents) write the story for their audience.
- Reporters are **not allowed to make up numbers**. Numbers come only from the
  **statisticians** (the engines: N8 counts the past, N9 simulates the future) and from
  the **camera** (N6 measures the batter's body in a video clip).
- An **editor** (the Faith Check) reads every story before it is printed. If a reporter
  got a number wrong, named the wrong player, sounded too sure, or said something no
  statistician measured, the story goes back. After two tries, it is not printed — and
  the report says so.

**One question, start to finish:**

> *"Kohli against pace in the powerplay — is his cover drive better than his pull?"*

1. **N3** reads the question: player = Kohli, phase = powerplay, bowler type = pace,
   shots = Cover Drive and Pull.
2. **N7a** fetches every matching ball from the database.
3. **N8** counts: cover drive strike rate 142.0 on 50 balls; pull 112.0 on 50 balls.
   It also works out that the difference could be anywhere from −16 to +76.
4. **N10** picks the right reporters: Tactical Analysis (coaches) and Personal
   Performance (players).
5. Each reporter writes **claims** (facts as data) and a sentence template.
6. The **Faith Check** sees the difference range includes 0, so "the cover drive is
   better" is not allowed. The reporter must say *"no detectable difference on this
   evidence"*.
7. **N12** writes the final report. Every number carries a tag like `[D, 50 balls]`.

The website lets anyone ask this, optionally upload a clip, watch the system work live,
and read the checked report.

---

## 2. How long will it take? (worst case)

**One "day" in this plan = one focused session of about 4–5 hours.**

The worst case assumes you are new to React, TypeScript, LangGraph, DuckDB and the
statistics, that every task takes its longer estimate, and that each phase ends with a
review day.

| Phase | What you build | Best case | Worst case |
|---|---|---|---|
| 0 | Foundations: setup, Python types, Pydantic, clean architecture, FastAPI, LangGraph | 6 | 8 |
| 1 | Claims: the facts agents are allowed to say | 5 | 7 |
| 2 | Faith Check v3: the editor (checks C1–C6) | 8 | 11 |
| 3 | Real cricket data: Cricsheet into a database | 7 | 11 |
| 4 | Engines: N8 (past) and N9 (future, with uncertainty) | 10 | 15 |
| 5 | The graph: all nodes and 12 agents, working in the terminal | 15 | 22 |
| 6 | The web API (FastAPI) | 8 | 12 |
| 7 | The website (React) | 16 | 25 |
| 8 | Video: classifier and pose | 11 | 16 |
| 9 | The commentary-to-shot bridge | 9 | 15 |
| 10 | Evaluation for the thesis | 8 | 12 |
| 11 | Ship it: Docker, README, demo | 5 | 8 |
| 12 | Write the paper | 10 | 15 |
| | **Total** | **118** | **177** |

Add **about 10% for load shedding, exams and sick days**: the worst case is
**about 195 sessions.**

| Sessions per week | Worst-case calendar time |
|---|---|
| 5 (CricketMind full time) | about 39 weeks ≈ **9 months** |
| 3 (shared with CricShot) | about 65 weeks ≈ **15 months** |

The order is chosen so you have something working early: a referee that works on fake
data by Day 26, answers in the terminal by Day 74, and a working website by Day 110.

---

## 3. Keeping the two projects apart

CricShot10k (the classifier and its accuracy paper) and CricketMind (this system) are
**two separate projects**.

| Rule | What it means in practice |
|---|---|
| **Separate folder and branch** | CricketMind v3 lives in its own **git worktree**, `D:\Cricket\cricketmind-v3-work\`, on the **`cricketmind-v3` branch**. The project folder is `cricketmind-v3-work\cricketmind-v3\`. CricShot stays in `D:\Cricket\cricshot-improve\` on its own branch, so the two never mix (branch rules below). Commands use full paths, so any VS Code window works. |
| **Two folders, two names** | `cricketmind/` (no suffix) is the old v2 prototype; CricShot's tests still import it, so never edit it. `cricketmind-v3/` is the new project, and its Python package is `cricketmind_v3`, so an import can never pick up the old one by accident. |
| **Separate Python environment** | Each project has its own `uv` environment. The new one is `cricketmind-v3/backend/.venv`. Don't activate CricShot's `.venv` while working on v3; `uv run` picks the right one. |
| **Separate journal** | `cricketmind-v3/notes/journal.md`. CricShot keeps its own. |
| **One bridge only: a file and a spec** | CricketMind never imports CricShot code. CricShot *exports* the final classifier (a TorchScript file + `classes.json` + a model card that says exactly what input it expects). CricketMind copies those files into its `models/` folder and checks the checksum. Day 112. |
| **One GPU at a time** | Never train CricShot while CricketMind uses the GPU (Ollama, the classifier, pose). |
| **Separate papers** | The classifier's accuracy is reserved for the CricShot paper. The CricketMind website and paper never report it. They may show a single clip's prediction and confidence. |
| **Old code is reference only** | The old notebook and `cricshot-improve/cricketmind/` are for remembering ideas, not for copying. |

### Git branch rules

1. **Two folders, two branches.** v3 work happens in the worktree
   `/d/Cricket/cricketmind-v3-work` on the `cricketmind-v3` branch. CricShot work happens in
   `/d/Cricket/cricshot-improve` on its own branch. A worktree is a second folder for the same
   repository, so CricShot can train in one folder while you write v3 code in the other.
   Day 1 creates it.
2. **Never switch branches in either folder.** Each folder keeps its own branch for good.
   (Git refuses to check out one branch in two folders anyway.)
3. **In the v3 worktree, only files inside `cricketmind-v3/` change.** Commit from the
   worktree root with `git add cricketmind-v3 && git commit -m "..."`. Never use
   `git add -A` there: the worktree also holds a copy of CricShot's files.
4. **The CricShot folder holds no v3 files.** The old copy is deleted on Day 1, so CricShot's
   usual `git add -A` can never pick up v3 work.
5. **Ignored things** (`.venv`, `data/`, `models/`, `node_modules/`) live only in the v3
   worktree and are never committed.
6. **Never delete the worktree folder by hand.** If you ever need to remove it, commit first,
   then run `git worktree remove /d/Cricket/cricketmind-v3-work` from the CricShot folder.

---

## 4. The daily loop

Every day has the same five steps.

1. **Understand (15 min).** Read the day's row. Write 2–3 lines in the journal in your own
   words: *what am I building today, and why does the system need it?*
2. **Learn (30–60 min).** Try the day's concept on a tiny example in `learning/`.
   No project code yet.
3. **Build (2–3 h).** Write the smallest version that works.
4. **Test (1 h).** Write tests, run them, make them pass. The "Done when" column is the
   finish line.
5. **Log, track and commit (15 min).** Write the day's entry in `notes/journal.md` (the
   template is at the top of that file). If the day produced a decision, a number, a
   limitation or a paper you read, add one line to the matching file in `thesis/`
   (Section 9). Then commit:

```bash
git add cricketmind-v3 && git commit -m "day 12: renderer turns claims into sentences"
```

Run it from the worktree root, `/d/Cricket/cricketmind-v3-work`. Always `git add
cricketmind-v3`, never `git add -A`, on this branch (see the branch rules in Section 3).

### A worked example: Day 12

> **Understand.** "Agents never type numbers. They write `{c1}` and the renderer puts the
> checked number in. Today I build the renderer."
>
> **Learn.** In `learning/day12_format.py`, try `f"{142.0:.1f}"` and `f"{0.405:.0%}"`.
>
> **Build.** `domain/render.py`: given a claim, return `"a strike rate of 142.0 [D, 50 balls]"`.
>
> **Test.** One test per metric type, plus one for a predictive claim with an interval.
>
> **Commit.**

---

## 5. The tech stack

### Backend (Python)

| Tool | What it does, simply |
|---|---|
| **Python 3.12 + uv** | Python, plus a fast tool that creates the environment and installs packages. You already use uv. |
| **FastAPI** | Turns Python functions into web endpoints, and writes the API documentation for you. |
| **Pydantic v2** | Checks that data has the right shape ("is `value` really a number?"). |
| **pydantic-settings** | Reads settings from a `.env` file. |
| **LangGraph** | Runs the flowchart of nodes, with loops and branches. |
| **DuckDB** | A database in one file that is very fast at "count and group" questions. It holds the ball-by-ball evidence. |
| **SQLite + SQLModel** | A small database for the website's own data (saved reports, history). |
| **Ollama + qwen2.5:7b** | The local language model, already installed. Behind a "port", so a cloud model can replace it later without touching the rest. |
| **PyTorch, OpenCV, Ultralytics** | Video frames, the classifier, and pose keypoints (Phase 8). |
| **pytest, ruff** | Tests, and a tool that keeps the code tidy. |

### Frontend — recommended stack

**React + TypeScript + Vite + Tailwind CSS + shadcn/ui + TanStack Query + React Router,
with an API client generated from FastAPI's OpenAPI file.**

| Piece | What it does, simply |
|---|---|
| **React** | Builds pages from small reusable pieces called components, like LEGO bricks. |
| **TypeScript** | JavaScript that warns you about mistakes before you run it — the same idea as Pydantic on the backend. |
| **Vite** | Starts a development server in a second and builds the final site. |
| **Tailwind CSS** | Style things by adding classes (`rounded border p-2`) instead of writing CSS files. |
| **shadcn/ui** | Ready-made, accessible components (Button, Card, Badge, Dialog) copied into your project, so you own and can read them. |
| **TanStack Query** | Fetches data from your API and handles loading, errors and caching for you. |
| **React Router** | Moves between pages (Ask, Live, Report, History). |
| **openapi-typescript + openapi-fetch** | Reads FastAPI's `/openapi.json` and writes TypeScript types. When the backend changes a field, the frontend shows a red error instead of a silent bug. |
| **Vitest + Testing Library, Playwright** | Tests for components, and tests that click through the real site. |

**Why this stack:**

- It is the most widely used combination, so tutorials, answers and jobs are everywhere.
- The site needs real interactivity: uploading a video with a progress bar, a live timeline
  of nodes, expandable "why was this withheld?" details. React handles that well.
- Vite keeps things simple: one frontend that talks to your FastAPI backend, with no second
  server to learn.

**Why not the others:**

- **Next.js** adds server-side rendering and its own server. You already have FastAPI.
- **Streamlit or Gradio** make quick demos, not real websites, and they blur the layers that
  clean architecture keeps apart.
- **HTMX + Jinja templates** is the simpler fallback if React feels too heavy: no JavaScript
  build at all, but less interactivity and less to learn for your career.

Node.js 24 is already installed. Docker Desktop is installed on Day 155.

---

## 6. Clean architecture, in simple words

### The restaurant

- The **chef** (domain) knows the recipes: what a claim is, what the six checks are. The
  chef does not care whether an order came by phone or app.
- The **kitchen manager** (application) runs the order: parse the question, fetch the
  evidence, call the engines, call the agents, run the Faith Check.
- The **suppliers** (infrastructure) deliver the ingredients: DuckDB, Ollama, the video
  reader, the classifier.
- The **waiter** (API) takes orders from customers (the website) and brings back the dish.

You can change a supplier without retraining the chef. That is the whole point.

### The layers

```
   frontend (React)  --HTTP-->  api (FastAPI)
                                    |
                                    v
                             application  <----  infrastructure
                         (use cases, ports,     (DuckDB, Ollama, video,
                          engines, the graph)    classifier, pose, SQLite)
                                    |
                                    v
                                 domain
                  (claims, registry, wording bands, checks C1-C6)
```

**The one rule:** arrows only point inward. `domain` imports nothing from the other layers
or any framework. `application` imports `domain` only. Infrastructure *implements* the ports
that application defines. A test on Day 8 enforces this rule automatically.

### A port, a fake and a real adapter

A **port** is a promise: "something that can give me deliveries." The engine is written
against the promise, not against DuckDB.

```python
# application/ports.py
from typing import Protocol

class DeliveryRepository(Protocol):
    def deliveries(self, subject): ...
```

The engine uses the port, with plain loops:

```python
# application/engines/strike_rate.py
def strike_rate(repo, subject):
    balls = 0
    runs = 0
    for d in repo.deliveries(subject):
        if d.is_wide:
            continue            # a wide is not a ball faced
        balls += 1
        runs += d.runs_batter
    return round(100.0 * runs / balls, 1)
```

Tests use a **fake** with five hand-made balls. No database is needed, and it runs in a
millisecond:

```python
# tests/fakes.py
class FakeDeliveryRepository:
    def __init__(self, rows):
        self.rows = rows

    def deliveries(self, subject):
        return self.rows
```

The real app uses the **DuckDB adapter** in `infrastructure/`, which runs SQL. Same promise,
different supplier.

### The folder layout

```
cricketmind-v3/
├── backend/
│   ├── pyproject.toml
│   ├── src/cricketmind_v3/
│   │   ├── domain/            # claims, registry, wording, faith checks (pure Python)
│   │   ├── application/       # ports, use cases, engines N8/N9, the LangGraph graph
│   │   ├── infrastructure/    # duckdb_repo, ollama_llm, video, classifier, pose, sqlite
│   │   ├── pipeline/          # offline jobs: Cricsheet ingest, attributes, shot bridge
│   │   └── api/               # FastAPI app, routers, schemas, live events
│   ├── prompts/               # one prompt file per agent
│   └── tests/                 # unit/, integration/, api/, graph/
├── frontend/                  # React + Vite + TypeScript
├── docs/                      # this plan, the v3 architecture, decision records
├── notes/journal.md
├── learning/                  # daily tiny experiments
├── data/                      # Cricsheet files, DuckDB file      (git-ignored)
└── models/                    # exported classifier, pose weights (git-ignored)
```

---

## 7. How we test

**The test pyramid:** many small, fast tests at the bottom, and a few big, slow ones at the
top.

| Level | What it tests | Speed | Example |
|---|---|---|---|
| **Unit** | domain and engines, with fakes | milliseconds | "C6 rejects a sentence naming Rohit for a claim about Kohli" |
| **Integration** | one adapter against the real thing, small | seconds | "the DuckDB repository returns 50 balls from a 3-match test file" |
| **Graph** | the whole flowchart, with a fake LLM | seconds | "an agent that lies once is fixed; one that always lies is withheld" |
| **API** | endpoints with `TestClient` | seconds | "`POST /api/queries` returns a job id" |
| **Frontend** | components with Vitest | seconds | "a predictive claim shows its interval" |
| **End-to-end** | the real site, clicked by Playwright | minutes | "ask a question, see the report" |

**The fake LLM.** Real language models give different answers each run and need Ollama
running. Tests use a fake that returns scripted replies, so every test run gives the same
result:

```python
class FakeLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def complete_json(self, prompt, schema):
        return self.replies.pop(0)
```

Tests that use the real model are marked `@pytest.mark.llm` and are run by hand.

---

## 8. Rules that never change

1. **Commit every day, on the `cricketmind-v3` branch, with `git add cricketmind-v3`.** A power
   cut should never cost more than one session.
2. **Long jobs are resumable.** Every batch job (ingest, labelling, pose) keeps a manifest of
   finished items and writes files atomically. You learned this the hard way in CricShot.
3. **Plain, readable Python.** Loops and if/else, not clever one-liners. Short, human comments.
4. **One GPU user at a time** (Section 3).
5. **The device rule:** GPU code picks Intel XPU, then CUDA, then CPU, so the same code runs
   on the Arc B580 and on the RTX 3070. It is written once, on Day 113.
6. **Numbers come only from N6, N8 and N9.** No agent, prompt or language model ever
   produces a number a user sees.
7. **No classifier accuracy in CricketMind** (Section 3).
8. **Tests before "done".** A day is finished when its "Done when" check passes.
9. **Track for the paper every day** (Section 9). Five minutes now saves weeks at the end.

---

## 9. The thesis paper: track everything

### Why every day, not at the end

The paper is written in Phase 12, but its raw material is made every day. If you don't write
it down the day it happens, you will forget why you chose something, which seed made a
figure, or which commit produced a number. CricShot taught this: a summary that only ever
appears in the terminal is gone the moment the window closes.

### Where things go

The `thesis/` folder already exists, and `thesis/README.md` explains it. In short:

| Something happened… | Write it in | Example |
|---|---|---|
| Any day ends | `notes/journal.md` | "Day 21: C4 rejects 'will probably' at p = 0.405" |
| You chose between options | `thesis/decisions.md` | "D-014: hedge required when the interval crosses a band — why, alternatives" |
| A script printed a number worth keeping | `thesis/numbers.md` | "N-009: Brier score 0.21 on the 2025 season — source file, seed, commit" |
| A claim became provable, or you plan one | `thesis/evidence.md` | "E-007: N9 is calibrated → `results/calibration_t1.csv`" |
| You noticed a weakness | `thesis/limitations.md` | "L-013: bowling type missing for 8% of balls" |
| You read a paper | `thesis/references.bib` + `thesis/related_work.md` | BibTeX entry, plus "what they did / how we differ" |

Day 0 is already filled in: the v3 design work from 2026-10-15/16, with decisions D-001 to
D-013, numbers N-001 to N-008, evidence E-001 to E-012 and limitations L-001 to L-012.

### The evidence map: practise what you preach

CricketMind refuses to print a number nobody computed. Your paper follows the same rule:

> **No number goes into the paper unless it has a row in `numbers.md`, and no claim unless it
> has a row in `evidence.md` pointing to the file that proves it.**

On Day 174 you check the finished draft against these two files, line by line. You are the
paper's Faith Check.

### The results ledger

From Day 25, every experiment script saves its output to `thesis/results/` and appends one
row to `thesis/results/ledger.jsonl`: date, commit hash, experiment name, settings, seed, the
key numbers, and the output files. You never type a result by hand. It is the same idea as
CricShot's `results.jsonl` and `caches.jsonl`.

### Drafts on review days

On every phase review day, spend one hour writing a rough draft of the paper section that
phase fed, in `thesis/drafts/`. `thesis/outline.md` lists which day drafts which section.
Rough is fine; it is raw material, not the paper. On the same days, read one paper from the
"to read" list in `related_work.md` and write its notes.

### Rules for this paper

- **The CricShot classifier's accuracy never appears.** It belongs to the CricShot paper.
- **Cite the ICCIT paper (v2), and clearly extend it.** The claim contract, uncertainty-
  faithful checks, real data, measured posture, the shot bridge and the website are new.
  Saying so keeps the paper clear of duplicate publication.
- **Synthetic results are labelled synthetic.**
- **Every figure and table is rebuilt by one script** in `thesis/scripts/`.

### What Claude does

After every finished step, along with the commit command, Claude gives you the journal entry
and any new lines for `decisions.md`, `numbers.md`, `evidence.md` or `limitations.md`, for you
to check and paste.

---

## 10. Milestones

| | When | What you can show |
|---|---|---|
| **M1** | Day 26 | The Faith Check catches all seven kinds of fabrication, on fake data |
| **M2** | Day 52 | The engines answer from real Cricsheet data, with honest intervals |
| **M3** | Day 74 | Full answers to text questions in the terminal |
| **M4** | Day 110 | The website answers text questions, live |
| **M5** | Day 127 | Questions with a video clip work end to end |
| **M6** | Day 140 | The commentary bridge result (positive or negative) |
| **M7** | Day 154 | Thesis results: tables and figures |
| **M8** | Day 161 | Shipped: one command starts everything, plus a demo video |
| **M9** | Day 177 | The paper, ready to submit |

---

## 11. Day-by-day plan

How to read the tables: **Learn** is the concept for the day, **Build** is what you make,
and **Done when** is the test that ends the day. A dash in Learn means the concept was
already covered and you go straight to building.

---

### Phase 0 — Foundations (Days 1–8)

**In simple words:** set up a clean workshop, then learn the four tools you will use every
day (dataclasses, Pydantic, FastAPI and LangGraph), using a toy app first.

**The toy app:** a *strike-rate calculator*. Give it a player, and it returns their strike
rate from a small CSV file. It is small enough to finish in a few days, and it teaches every
layer of clean architecture.

| Day | Learn | Build | Done when |
|---|---|---|---|
| 1 | What a branch, a worktree, a virtual environment and a test are | The v3 worktree and branch; the backend with uv; pytest and ruff; pytest's own settings; a `.gitignore` for v3; the architecture PDF in `docs/`; read `thesis/README.md` | `uv run pytest` shows 1 passed; first commit on the v3 branch |
| 2 | Type hints and `@dataclass(frozen=True)` | `learning/day02_ball.py`: a `Ball` with runs and is_wicket | Tests: equal balls compare equal; changing a frozen ball raises an error |
| 3 | Pydantic: automatic input checks | `learning/day03_query.py`: `QueryIn` with text (minimum length) and an optional player | Test: bad input raises a clear error |
| 4 | Clean architecture: a domain function, a port (`typing.Protocol`), a fake, a use case | Toy `strike_rate()`, `BallRepository` port, `FakeBallRepository`, `GetStrikeRate` use case | Tests use the fake only; no file is read |
| 5 | FastAPI basics; `Depends` (dependency injection) | Toy `GET /strike-rate/{batter}` calling the use case; a real CSV adapter | `TestClient` test passes with the fake swapped in |
| 6 | LangGraph 1: state, nodes, edges | A toy 2-node graph: `count_runs` → `compute_strike_rate` | Test: run the graph, check the final state |
| 7 | LangGraph 2: conditional edges, a bounded loop, fan-out | A toy retry loop that stops after 2 tries; two parallel nodes | Test proves the loop never runs a third time |
| 8 | Read the v3 architecture once, slowly | The real skeleton: `domain/`, `application/`, `infrastructure/`, `pipeline/`, `api/`; an architecture test that fails if `domain` imports a framework | Architecture test passes, and fails when you break it on purpose; journal: v3 in your own words, 10 lines |

#### How to use this guide

- Every day has the same parts: **Goal**, **Why it matters for v3**, **Learn**, numbered
  **Steps**, **Done when**, and **Finish the day**.
- **Type the code; don't paste it.** Typing is how it sticks, and when you mistype, the error
  teaches you something.
- After each step, compare what you see with the **You should see** line. If it is different,
  stop and fix it before moving on.
- **Two folders matter.** Git commands run in the worktree root; `uv`, `pytest` and `ruff`
  run in `backend/`:

| Short name | Full path | What you run there |
|---|---|---|
| worktree root | `/d/Cricket/cricketmind-v3-work` | `git …` |
| `backend/` | `/d/Cricket/cricketmind-v3-work/cricketmind-v3/backend` | `uv …` |

- All toy code goes in `cricketmind-v3/learning/`. It is practice, not project code. It is
  committed so you can look back at it, but nothing in `backend/src/` ever imports it.
- **Every day ends the same way** (Section 4): check *Done when*, write the journal entry,
  then commit from the worktree root:

```bash
cd /d/Cricket/cricketmind-v3-work
git status                  # only paths inside cricketmind-v3/ should be listed
git add cricketmind-v3
git commit -m "day N: what you built"
```

---

#### Day 1 — The workshop: worktree, environment, first test

**Goal:** a v3 folder on its own branch, its own Python environment, and one passing test.

**Why it matters for v3:** everything after today depends on three things. v3 can never
touch CricShot's files, `uv run` always uses v3's own packages, and pytest uses v3's own
settings.

**Learn (30 min)**

- **Branch:** a separate line of commits. `langGraph` holds CricShot; today you create
  `cricketmind-v3`.
- **Worktree:** a second folder for the same repository, with a different branch checked
  out. CricShot stays in `cricshot-improve/`; v3 lives in `cricketmind-v3-work/`. Nobody ever
  switches branches, so a running CricShot job can never lose a file, and neither folder sees
  the other's unsaved work.
- **Virtual environment (`.venv`):** a private box of packages for one project. uv builds it,
  and `uv run <command>` always runs inside it, so you never "activate" anything.
- **Test:** a function whose name starts with `test_`. pytest finds them and runs them all. If
  an `assert` is false, the test fails and pytest shows you both sides.

**Step 1 — Create the worktree and the branch.** Uncommitted CricShot work can stay as it
is. The worktree starts from CricShot's last *commit* and leaves its working files alone.

```bash
cd /d/Cricket/cricshot-improve
git worktree add -b cricketmind-v3 /d/Cricket/cricketmind-v3-work langGraph
git worktree list
```

You should see `Preparing worktree (new branch 'cricketmind-v3')`, then `HEAD is now at …`
(CricShot's latest commit). `git worktree list` prints two lines: `cricshot-improve` with
`[langGraph]`, and `cricketmind-v3-work` with `[cricketmind-v3]`.

**Step 2 — Move the plan into the worktree.** The plan, the journal and `thesis/` were
written in the CricShot folder. Copy them across:

```bash
cp -r /d/Cricket/cricshot-improve/cricketmind-v3 /d/Cricket/cricketmind-v3-work/
cd /d/Cricket/cricketmind-v3-work
git status
```

You should see `On branch cricketmind-v3` and, under *Untracked files*, `cricketmind-v3/`.

If `cricketmind-v3/` is **not** listed, CricShot's `.gitignore` is hiding it. Open
`/d/Cricket/cricketmind-v3-work/.gitignore`, delete the `cricketmind-v3/` line and the two
comment lines above it, run `git status` again, and add `.gitignore` to today's commit
(Step 10).

**Step 3 — Stay where you are.** You don't need a new window. Every command in this plan
starts with a `cd` to a full path, so it works from any terminal. To open a v3 file, run
`code` with its full path. It opens as a tab in the window you are in:

```bash
code /d/Cricket/cricketmind-v3-work/cricketmind-v3/docs/BUILD_PLAN.md
```

**Step 4 — Create the backend project.**

```bash
cd /d/Cricket/cricketmind-v3-work/cricketmind-v3
uv init --package --vcs none --python 3.12 --name cricketmind-v3 backend
```

| Flag | Why |
|---|---|
| `--package` | makes a real, installable package in `src/cricketmind_v3/`, so tests import it the same way the app will |
| `--vcs none` | doesn't make a second git repository inside this one |
| `--python 3.12` | your default `python` is 3.10; uv already has 3.12, and this pins it in `.python-version` |
| `--name cricketmind-v3` | the package becomes `cricketmind_v3`, so it can never be confused with the old `cricketmind` |

You should see ``Initialized project `cricketmind-v3` at …\backend``. Look inside `backend/`:
`pyproject.toml`, `.python-version`, `README.md` and `src/cricketmind_v3/__init__.py`.

**Step 5 — Add the dev tools.**

```bash
cd backend
uv add --dev pytest ruff
```

This creates `.venv/` and `uv.lock`. The lock file records the exact versions, so the project
installs the same way next year. A yellow *Failed to hardlink files* warning is harmless:
uv's cache is on `C:` and the project is on `D:`.

**Step 6 — Give pytest its own settings.** Add this at the end of `backend/pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
```

Why: CricShot's `pytest.ini` sits at the worktree root. Without this block, pytest walks up
the folders, finds that file, and uses CricShot's settings.

**Step 7 — Write and run the first test.** Create `backend/tests/test_first.py`:

```python
def test_one():
    assert 1 + 1 == 2
```

```bash
uv run pytest
```

You should see `configfile: pyproject.toml` near the top and `1 passed` at the bottom. If it
says `configfile: pytest.ini`, Step 6 is missing.

Now break it on purpose. Change the `2` to `3`, run it again, read the red `assert 2 == 3`,
then change it back. Then run the tidiness tool:

```bash
uv run ruff check .
```

You should see `All checks passed!`

**Step 8 — Write a `.gitignore` for v3.** Create `cricketmind-v3/.gitignore`:

```
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.env
node_modules/
/data/
/models/
```

The leading `/` means "only the top-level `data/` and `models/` folders". A small test folder
called `data/` deeper down can still be saved.

**Step 9 — Copy the architecture document and read the thesis guide.** From `backend/`:

```bash
cp ../../paper/CricketMind_Architecture_v3.pdf ../docs/
```

Then read `thesis/README.md` (10 minutes). It explains where each kind of note goes, and you
will use it every day.

**Step 10 — Make the first commit.** Write today's journal entry (the template is at the top
of `notes/journal.md`), then:

```bash
cd /d/Cricket/cricketmind-v3-work
git add cricketmind-v3
git status
```

Check the green list before you commit. It should include `.gitignore`,
`backend/pyproject.toml`, `backend/uv.lock`, `backend/tests/test_first.py`, `docs/`,
`notes/` and `thesis/`. `.venv` and `__pycache__` must **not** be there.

```bash
git commit -m "day 1: cricketmind-v3 skeleton"
```

**Step 11 — Empty the old copy.** v3 now lives in the worktree, so the old plan, journal and
`thesis/` in the CricShot folder have to go. Two copies would drift apart. Keep the empty
folder itself, because the v3 Claude chat runs from it. First confirm that the commit exists,
then remove the old files:

```bash
git -C /d/Cricket/cricketmind-v3-work log --oneline -1    # shows "day 1: cricketmind-v3 skeleton"
cd /d/Cricket/cricshot-improve/cricketmind-v3
rm -r docs notes thesis
```

**Done when**

- `uv run pytest` shows `1 passed` and `configfile: pyproject.toml`.
- `git -C /d/Cricket/cricketmind-v3-work log --oneline -1` shows the day 1 commit.
- `git -C /d/Cricket/cricshot-improve status` still shows CricShot's own changes, untouched.
  That is the separation working.

**For the paper:** nothing today. The setup is process, not method.

---

#### Day 2 — Type hints and frozen dataclasses

**Goal:** a `Ball` object that compares by its values and cannot be changed by accident.

**Why it matters for v3:** a claim is a fact. Once the Faith Check has checked it, nothing may
change it before it is shown. On Day 10, `Claim` is frozen for exactly this reason.

**Learn**

- **Type hints** (`runs: int`) are notes for you, your editor and ruff. Python does **not**
  check them when the code runs. You will see that in Step 2.
- **`@dataclass`** writes the boring parts for you: `__init__`, a readable print-out, and an
  `==` that compares the values.
- **`frozen=True`** makes every field read-only once the object is made.

**Step 1 — Write the dataclass.** Create `cricketmind-v3/learning/day02_ball.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Ball:
    batter: str
    runs: int
    is_wicket: bool = False


if __name__ == "__main__":
    four = Ball("V Kohli", 4)
    print(four)
    print(four == Ball("V Kohli", 4))  # same values, so equal
    print(Ball("V Kohli", "four"))  # type hints are not checked!
    four.runs = 6  # frozen, so this line crashes
```

**Step 2 — Run it and read every line it prints.**

```bash
cd /d/Cricket/cricketmind-v3-work/cricketmind-v3/backend
uv run python ../learning/day02_ball.py
```

You should see:

1. the ball, printed;
2. `True`;
3. a ball with `runs='four'`, because Python accepted a word as the runs;
4. finally, `FrozenInstanceError: cannot assign to field 'runs'`.

Line 3 is why Day 3 exists.

**Step 3 — Keep pytest inside `learning/`.** Create `learning/pytest.ini` with one line:

```ini
[pytest]
```

An empty settings file still tells pytest to stop looking here. Without it, pytest climbs up
to CricShot's `pytest.ini` again, for the same reason as Day 1, Step 6.

**Step 4 — Write the tests.** Create `learning/test_day02_ball.py`:

```python
import dataclasses

import pytest
from day02_ball import Ball


def test_equal_balls_compare_equal():
    first = Ball("V Kohli", 4)
    second = Ball("V Kohli", 4)
    assert first == second


def test_different_balls_are_not_equal():
    assert Ball("V Kohli", 4) != Ball("V Kohli", 6)


def test_wicket_defaults_to_false():
    assert Ball("V Kohli", 0).is_wicket is False


def test_frozen_ball_cannot_change():
    ball = Ball("V Kohli", 4)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ball.runs = 6
```

`with pytest.raises(...)` means "this block must crash with exactly this error; if it doesn't
crash, fail the test".

**Step 5 — Run them.**

```bash
uv run pytest ../learning
uv run ruff check ../learning
```

You should see `4 passed`, then `All checks passed!`

**Step 6 — Break it on purpose.** Delete `frozen=True`, keeping `@dataclass`, and run the
tests again. `test_frozen_ball_cannot_change` fails, because the ball could now be changed.
Put it back.

**Done when:** 4 tests pass, and you can say in one sentence why a checked fact should be
frozen.

**Finish the day:** journal entry, then:

```bash
cd /d/Cricket/cricketmind-v3-work
git add cricketmind-v3 && git commit -m "day 2: frozen Ball dataclass"
```

---

#### Day 3 — Pydantic: checking data at the door

**Goal:** a `QueryIn` model that rejects bad input with a clear message.

**Why it matters for v3:** every question from the website arrives as JSON text. Pydantic
checks its shape before any of your code runs. On Day 5, FastAPI uses it for you
automatically. From Day 75, every request and response of the real API is a Pydantic model.

**Learn**

| | dataclass (Day 2) | Pydantic model (today) |
|---|---|---|
| Checks the types? | No, it trusts you | Yes, every time |
| Use it for | data made inside your code | data that comes from outside: users, JSON, the LLM |
| Where it goes in v3 | `domain/`, which is pure Python (the Day 8 rule) | the API edge, and the LLM's JSON replies |

- `Field(min_length=5)` adds a rule to a field.
- `str | None = None` means "optional; `None` when it is missing".
- When a rule fails, Pydantic raises a `ValidationError` that names the field, the rule and
  the bad value.

**Step 1 — Install Pydantic.**

```bash
cd /d/Cricket/cricketmind-v3-work/cricketmind-v3/backend
uv add pydantic
```

You should see `pydantic` listed under `dependencies` in `pyproject.toml`.

**Step 2 — Write the model.** Create `learning/day03_query.py`:

```python
from pydantic import BaseModel, Field


class QueryIn(BaseModel):
    text: str = Field(min_length=5)
    player: str | None = None


if __name__ == "__main__":
    print(QueryIn(text="Kohli vs pace?", player="V Kohli"))
    print(QueryIn.model_validate_json('{"text": "Who bowls best at the death?"}'))
    QueryIn(text="hi")  # too short, so this line crashes
```

**Step 3 — Run it and read the error.**

```bash
uv run python ../learning/day03_query.py
```

You should see two good queries, then:

```
1 validation error for QueryIn
text
  String should have at least 5 characters [type=string_too_short, input_value='hi', input_type=str]
```

Find the three parts: **which field** (`text`), **which rule** (at least 5 characters), and
**what was sent** (`input_value='hi'`).

**Step 4 — Write the tests.** Create `learning/test_day03_query.py`:

```python
import pytest
from day03_query import QueryIn
from pydantic import ValidationError


def test_good_query_is_accepted():
    query = QueryIn(text="Kohli vs pace?", player="V Kohli")
    assert query.player == "V Kohli"


def test_player_is_optional():
    query = QueryIn(text="Who bowls best at the death?")
    assert query.player is None


def test_short_text_is_rejected():
    with pytest.raises(ValidationError) as error:
        QueryIn(text="hi")
    assert error.value.errors()[0]["type"] == "string_too_short"


def test_missing_text_is_rejected():
    with pytest.raises(ValidationError):
        QueryIn(player="V Kohli")


def test_number_is_not_text():
    with pytest.raises(ValidationError):
        QueryIn(text=12345)


def test_from_json():
    query = QueryIn.model_validate_json('{"text": "Kohli vs pace?"}')
    assert query.text == "Kohli vs pace?"
```

Compare `test_number_is_not_text` with Day 2, Step 2. The dataclass took `"four"` as runs;
Pydantic refuses `12345` as text.

**Step 5 — Run them.**

```bash
uv run pytest ../learning
uv run ruff check ../learning
```

You should see `10 passed` (4 from Day 2 and 6 new).

**Done when:** the 6 new tests pass, and the journal explains the three parts of a Pydantic
error in your own words.

**Finish the day:** journal entry, then:

```bash
cd /d/Cricket/cricketmind-v3-work
git add cricketmind-v3 && git commit -m "day 3: QueryIn with pydantic checks"
```

---

#### Day 4 — Clean architecture, small: the strike-rate toy

**Goal:** a strike-rate calculator in four small pieces (a domain function, a port, a use case
and a fake), tested without reading any file.

**Why it matters for v3:** this is the shape of the whole system. On Day 39 the real engine
uses a `DeliveryRepository` port exactly like today's `BallRepository`, with DuckDB behind it
instead of a list.

**Learn:** re-read Section 6 (the restaurant, and "A port, a fake and a real adapter").
Then learn the four pieces for today:

| Piece | In the restaurant | Today's file | Its rule |
|---|---|---|---|
| domain function | the chef's recipe | `toy/domain.py` | data in, answer out; no files, no web, no database |
| port | the promise "a supplier who can give me balls" | `toy/ports.py` | says *what*, never *how* |
| use case | the kitchen manager | `toy/use_cases.py` | asks the port, then calls the domain |
| fake | a pretend supplier with hand-made balls | `toy/fakes.py` | used only by tests |

A port is written with `typing.Protocol`. It means "any object with a `balls_for(batter)`
method will do". The use case never knows whether it got a fake, a CSV reader or DuckDB.

**Step 1 — Make a package for the toy.**

```bash
cd /d/Cricket/cricketmind-v3-work/cricketmind-v3/learning
mkdir toy
touch toy/__init__.py
```

The empty `__init__.py` makes `toy` a package, so a test can write `from toy.domain import
Ball`.

**Step 2 — Write the domain.** Create `learning/toy/domain.py`. This `Ball` has `is_wide`
instead of `is_wicket`, because a wide is not a ball faced:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Ball:
    batter: str
    runs: int
    is_wide: bool = False


def strike_rate(balls: list[Ball]) -> float | None:
    runs = 0
    faced = 0
    for ball in balls:
        if ball.is_wide:
            continue  # a wide is not a ball faced
        faced += 1
        runs += ball.runs
    if faced == 0:
        return None
    return round(100.0 * runs / faced, 1)
```

Work one out by hand before you run anything: 4, 1, a wide, 6 makes 11 runs from 3 balls
faced, so 100 × 11 / 3 = 366.7.

**Step 3 — Write the port.** Create `learning/toy/ports.py`:

```python
from typing import Protocol

from toy.domain import Ball


class BallRepository(Protocol):
    def balls_for(self, batter: str) -> list[Ball]: ...
```

The `...` is the whole body. A port has no code, only the promise.

**Step 4 — Write the use case.** Create `learning/toy/use_cases.py`:

```python
from toy.domain import strike_rate
from toy.ports import BallRepository


class GetStrikeRate:
    def __init__(self, repo: BallRepository):
        self.repo = repo

    def run(self, batter: str) -> float | None:
        balls = self.repo.balls_for(batter)
        return strike_rate(balls)
```

The use case is handed its repository from outside, in `__init__`; it never creates one.
That is **dependency injection**, and tomorrow FastAPI does it for you.

**Step 5 — Write the fake.** Create `learning/toy/fakes.py`:

```python
from toy.domain import Ball


class FakeBallRepository:
    def __init__(self, balls: list[Ball]):
        self.balls = balls
        self.asked = []  # remembers who we were asked about

    def balls_for(self, batter: str) -> list[Ball]:
        self.asked.append(batter)
        found = []
        for ball in self.balls:
            if ball.batter == batter:
                found.append(ball)
        return found
```

`asked` lets a test check that the use case really asked the repository.

**Step 6 — Write the tests.** Create `learning/test_day04_toy.py`:

```python
from toy.domain import Ball, strike_rate
from toy.fakes import FakeBallRepository
from toy.use_cases import GetStrikeRate

KOHLI_BALLS = [
    Ball("V Kohli", 4),
    Ball("V Kohli", 1),
    Ball("V Kohli", 0, is_wide=True),
    Ball("V Kohli", 6),
]


def test_strike_rate_counts_runs_per_ball():
    assert strike_rate([Ball("V Kohli", 4), Ball("V Kohli", 0)]) == 200.0


def test_wide_is_not_a_ball_faced():
    assert strike_rate(KOHLI_BALLS) == 366.7


def test_no_balls_means_no_strike_rate():
    assert strike_rate([]) is None


def test_use_case_asks_the_repository():
    repo = FakeBallRepository(KOHLI_BALLS)
    use_case = GetStrikeRate(repo)
    assert use_case.run("V Kohli") == 366.7
    assert repo.asked == ["V Kohli"]


def test_unknown_batter_gets_none():
    use_case = GetStrikeRate(FakeBallRepository(KOHLI_BALLS))
    assert use_case.run("RG Sharma") is None
```

**Step 7 — Run them.**

```bash
cd /d/Cricket/cricketmind-v3-work/cricketmind-v3/backend
uv run pytest ../learning
uv run ruff check ../learning
```

You should see `15 passed`.

**Step 8 — Draw the arrows.** In the journal, draw which file imports which:

```
use_cases.py ──> ports.py ──> domain.py
use_cases.py ───────────────> domain.py
fakes.py ───────────────────> domain.py
```

Every arrow ends at `domain.py`, and `domain.py` imports nothing of yours. That is what
"arrows point inward" means.

**Done when:** the 5 new tests pass, none of them reads a file, and `domain.py` imports only
`dataclasses`.

**Finish the day:** journal entry, then:

```bash
cd /d/Cricket/cricketmind-v3-work
git add cricketmind-v3 && git commit -m "day 4: strike-rate toy with a port, a use case and a fake"
```

---

#### Day 5 — FastAPI and `Depends`: the toy goes on the web

**Goal:** `GET /strike-rate/{batter}` answers from a CSV file in the browser, and from the
fake in tests.

**Why it matters for v3:** the website talks to v3 through endpoints like this one. `Depends`
is how the real app gets DuckDB while the tests get a fake: the same use case, two suppliers.
Day 77 builds this wiring for the real system.

**Learn**

- An **endpoint** is a function plus a URL. `{batter}` in the URL becomes the `batter`
  argument.
- **`Depends(get_repo)`** says "before you call me, call `get_repo()` and hand me what it
  returns".
- `Annotated[BallRepository, Depends(get_repo)]` reads as "a `BallRepository`, which FastAPI
  gets from `get_repo`".
- **`app.dependency_overrides[get_repo] = use_fake`** swaps the supplier for one test.
- **`TestClient`** is a pretend browser. It calls your app directly, so no server is needed.
- `HTTPException(status_code=404)` is the polite way to say "not found".

**Step 1 — Install FastAPI.**

```bash
cd /d/Cricket/cricketmind-v3-work/cricketmind-v3/backend
uv add "fastapi[standard]"
```

`[standard]` also brings the web server (uvicorn), `httpx` (which `TestClient` needs) and the
`fastapi` command.

**Step 2 — Make a tiny data file.** Create `learning/toy/balls.csv`:

```
batter,runs,is_wide
V Kohli,4,0
V Kohli,1,0
V Kohli,0,1
V Kohli,6,0
RG Sharma,0,0
RG Sharma,2,0
```

**Step 3 — Write the real adapter.** It keeps the same promise as the fake, with a different
supplier. Create `learning/toy/csv_repo.py`:

```python
import csv
from pathlib import Path

from toy.domain import Ball


class CsvBallRepository:
    def __init__(self, path: Path):
        self.path = path

    def balls_for(self, batter: str) -> list[Ball]:
        found = []
        with open(self.path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["batter"] != batter:
                    continue
                ball = Ball(row["batter"], int(row["runs"]), row["is_wide"] == "1")
                found.append(ball)
        return found
```

Everything in a CSV is text. That is why you convert it yourself with `int(...)` and
`== "1"`.

**Step 4 — Write the endpoint.** Create `learning/toy/api.py`:

```python
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException

from toy.csv_repo import CsvBallRepository
from toy.ports import BallRepository
from toy.use_cases import GetStrikeRate

CSV_PATH = Path(__file__).parent / "balls.csv"

app = FastAPI(title="Toy strike rate")


def get_repo() -> BallRepository:
    return CsvBallRepository(CSV_PATH)


@app.get("/strike-rate/{batter}")
def read_strike_rate(batter: str, repo: Annotated[BallRepository, Depends(get_repo)]):
    value = GetStrikeRate(repo).run(batter)
    if value is None:
        raise HTTPException(status_code=404, detail="No balls found for " + batter)
    return {"batter": batter, "strike_rate": value}
```

**Step 5 — Run the server and use it.**

```bash
uv run fastapi dev ../learning/toy/api.py
```

Open these in your browser:

- http://127.0.0.1:8000/strike-rate/V%20Kohli shows `{"batter":"V Kohli","strike_rate":366.7}`.
- http://127.0.0.1:8000/strike-rate/Nobody shows `{"detail":"No balls found for Nobody"}`
  (status 404).
- http://127.0.0.1:8000/docs shows the documentation FastAPI wrote for you. Click the
  endpoint, then **Try it out**, type `RG Sharma`, and click **Execute**. You should get
  `100.0`.

Stop the server with **Ctrl+C**. It does not reload when you edit the toy files, so restart it
after every change.

**Step 6 — Write the tests.** Create `learning/test_day05_api.py`:

```python
from fastapi.testclient import TestClient
from toy.api import CSV_PATH, app, get_repo
from toy.csv_repo import CsvBallRepository
from toy.domain import Ball
from toy.fakes import FakeBallRepository


def test_endpoint_with_the_fake():
    fake = FakeBallRepository([Ball("V Kohli", 4), Ball("V Kohli", 2)])

    def use_fake():
        return fake

    app.dependency_overrides[get_repo] = use_fake
    client = TestClient(app)
    response = client.get("/strike-rate/V Kohli")
    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"batter": "V Kohli", "strike_rate": 300.0}
    assert fake.asked == ["V Kohli"]


def test_unknown_batter_is_404():
    def use_empty_fake():
        return FakeBallRepository([])

    app.dependency_overrides[get_repo] = use_empty_fake
    client = TestClient(app)
    response = client.get("/strike-rate/Nobody")
    app.dependency_overrides.clear()

    assert response.status_code == 404


def test_csv_repository_reads_the_file():
    repo = CsvBallRepository(CSV_PATH)
    balls = repo.balls_for("RG Sharma")
    assert balls == [Ball("RG Sharma", 0), Ball("RG Sharma", 2)]
```

The first two tests never touch the CSV. The third tests the real adapter on its own: a
small *integration* test (Section 7).

**Step 7 — Run them.**

```bash
uv run pytest ../learning
uv run ruff check ../learning
```

You should see `18 passed`.

**Done when:** the test with the fake swapped in passes, and you have used `/docs` at least
once.

**Finish the day:** journal entry, then:

```bash
cd /d/Cricket/cricketmind-v3-work
git add cricketmind-v3 && git commit -m "day 5: toy endpoint with Depends and a CSV adapter"
```

---

#### Day 6 — LangGraph 1: state, nodes, edges

**Goal:** the strike-rate calculation as a two-node graph: `count_runs` →
`compute_strike_rate`.

**Why it matters for v3:** the whole question-answering pipeline (N3 → N6 → … → N13) is one
LangGraph graph, built from Day 53. Today you build the smallest graph that still has all
three parts.

**Learn**

- **State:** the shared notebook. A `TypedDict` lists every key it may hold.
- **Node:** a plain function. It reads the state and returns **only the keys it changed**, and
  LangGraph writes those into the notebook.
- **Edge:** "after this node, run that one". `START` and `END` are the doors in and out.
- `compile()` turns the drawing into something you can run. `invoke()` runs it and returns
  the final state. `stream()` gives one update per node, which is how the website will show
  live progress (Day 80).
- A node's name is its function's name (`"count_runs"` for `count_runs`), so the drawing and
  the code always match.

**Step 1 — Install LangGraph.**

```bash
cd /d/Cricket/cricketmind-v3-work/cricketmind-v3/backend
uv add langgraph
```

**Step 2 — Write the graph.** Create `learning/toy/graph.py`:

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from toy.domain import Ball


class ToyState(TypedDict, total=False):
    balls: list[Ball]
    runs: int
    faced: int
    strike_rate: float | None


def count_runs(state: ToyState) -> dict:
    runs = 0
    faced = 0
    for ball in state["balls"]:
        if ball.is_wide:
            continue
        faced += 1
        runs += ball.runs
    return {"runs": runs, "faced": faced}


def compute_strike_rate(state: ToyState) -> dict:
    if state["faced"] == 0:
        return {"strike_rate": None}
    return {"strike_rate": round(100.0 * state["runs"] / state["faced"], 1)}


def build_graph():
    graph = StateGraph(ToyState)
    graph.add_node("count_runs", count_runs)
    graph.add_node("compute_strike_rate", compute_strike_rate)
    graph.add_edge(START, "count_runs")
    graph.add_edge("count_runs", "compute_strike_rate")
    graph.add_edge("compute_strike_rate", END)
    return graph.compile()
```

`total=False` means "the keys may be missing at the start". At the start the notebook has
only `balls`; the nodes fill in the rest.

**Step 3 — See it run.** Create `learning/day06_draw.py`:

```python
from toy.domain import Ball
from toy.graph import build_graph

app = build_graph()
print(app.get_graph().draw_mermaid())

balls = [Ball("V Kohli", 4), Ball("V Kohli", 0, is_wide=True), Ball("V Kohli", 2)]
for step in app.stream({"balls": balls}):
    print(step)
```

```bash
uv run python ../learning/day06_draw.py
```

You should see a drawing written as text (paste it into https://mermaid.live to see boxes and
arrows), then one line per node:

```
{'count_runs': {'runs': 6, 'faced': 2}}
{'compute_strike_rate': {'strike_rate': 300.0}}
```

Each node returned only what it changed.

**Step 4 — Write the tests.** Create `learning/test_day06_graph.py`:

```python
from toy.domain import Ball
from toy.graph import build_graph, count_runs

BALLS = [Ball("V Kohli", 4), Ball("V Kohli", 0, is_wide=True), Ball("V Kohli", 2)]


def test_graph_runs_end_to_end():
    app = build_graph()
    final = app.invoke({"balls": BALLS})
    assert final["runs"] == 6
    assert final["faced"] == 2
    assert final["strike_rate"] == 300.0


def test_a_node_is_just_a_function():
    assert count_runs({"balls": BALLS}) == {"runs": 6, "faced": 2}


def test_stream_shows_each_node_and_what_it_changed():
    app = build_graph()
    steps = list(app.stream({"balls": BALLS}))
    assert steps == [
        {"count_runs": {"runs": 6, "faced": 2}},
        {"compute_strike_rate": {"strike_rate": 300.0}},
    ]
```

The second test needs no graph at all. A node is just a function, so you can test it on its
own.

**Step 5 — Run them.**

```bash
uv run pytest ../learning
uv run ruff check ../learning
```

You should see `21 passed`.

**Step 6 — Break it on purpose.** In `count_runs`, change `"runs": runs` to `"runz": runs`
and run the tests. LangGraph silently drops a key that the state does not list, so the
mistake only shows up one node later, as `KeyError: 'runs'` in `compute_strike_rate`. The
lesson: the state is a contract, and tests are what catch a typo in it. Put it back.

**Done when:** the 3 new tests pass, and the journal has one line each for state, node and
edge.

**Finish the day:** journal entry, then:

```bash
cd /d/Cricket/cricketmind-v3-work
git add cricketmind-v3 && git commit -m "day 6: first langgraph, two nodes"
```

---

#### Day 7 — LangGraph 2: branches, a bounded loop and parallel nodes

**Goal:** (a) a retry loop that can never run a third time; (b) two nodes that run side by
side, and a third node that waits for both.

**Why it matters for v3:** the Faith Check sends a failing agent back to rewrite, at most
twice (Day 70). If that loop had no limit, one stubborn agent would run forever. The agents
also run in parallel and are then collected (Day 69).

**Learn**

- **Conditional edge:** after a node, a *router* function looks at the state and returns the
  **name** of the next node.
- **Bounded loop:** keep a counter (`tries`) in the state. The router checks it before it
  sends the work round again.
- LangGraph also has a safety net, `recursion_limit` (25 steps by default), which stops a
  runaway graph with an error. Never rely on it. Your own limit must stop the loop first, with
  a proper answer.
- **Fan-out:** two edges leaving the same place run both nodes in the same step. **Join:**
  `add_edge(["a", "b"], "c")` makes `c` wait for both.
- **Reducer:** when two parallel nodes write the same key, LangGraph needs to know how to
  combine them. `Annotated[list[str], operator.add]` means "join the lists".

**The pretend agent:** the state carries a `script`, the answer the agent gives on try 1,
try 2 and so on. It is the same trick as the fake LLM in Section 7. `"right"` passes the
check, and anything else fails it.

**Step 1 — Write the retry loop.** Create `learning/toy/retry_graph.py`:

```python
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

MAX_TRIES = 2


class RetryState(TypedDict, total=False):
    script: list[str]  # what the pretend agent says on each try
    tries: int
    answer: str
    ok: bool
    status: str


def write_answer(state: RetryState) -> dict:
    tries = state["tries"] + 1
    answer = state["script"][tries - 1]
    return {"tries": tries, "answer": answer}


def check_answer(state: RetryState) -> dict:
    return {"ok": state["answer"] == "right"}


def route_after_check(state: RetryState) -> str:
    if state["ok"]:
        return "accept"
    if state["tries"] >= MAX_TRIES:
        return "withhold"
    return "write_answer"


def accept(state: RetryState) -> dict:
    return {"status": "accepted"}


def withhold(state: RetryState) -> dict:
    return {"status": "withheld"}


def build_retry_graph():
    graph = StateGraph(RetryState)
    graph.add_node("write_answer", write_answer)
    graph.add_node("check_answer", check_answer)
    graph.add_node("accept", accept)
    graph.add_node("withhold", withhold)
    graph.add_edge(START, "write_answer")
    graph.add_edge("write_answer", "check_answer")
    graph.add_conditional_edges(
        "check_answer", route_after_check, ["accept", "withhold", "write_answer"]
    )
    graph.add_edge("accept", END)
    graph.add_edge("withhold", END)
    return graph.compile()
```

Read `route_after_check` out loud: "if it's right, accept it; if we have already tried twice,
withhold it; otherwise, write again."

**Step 2 — Write the fan-out.** Create `learning/toy/fanout_graph.py`:

```python
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from toy.domain import Ball


class FanState(TypedDict, total=False):
    balls: list[Ball]
    notes: Annotated[list[str], operator.add]  # parallel writes get joined, not overwritten
    summary: str


def count_boundaries(state: FanState) -> dict:
    count = 0
    for ball in state["balls"]:
        if ball.runs == 4 or ball.runs == 6:
            count += 1
    return {"notes": ["boundaries: " + str(count)]}


def count_dots(state: FanState) -> dict:
    count = 0
    for ball in state["balls"]:
        if ball.runs == 0 and not ball.is_wide:
            count += 1
    return {"notes": ["dots: " + str(count)]}


def summarise(state: FanState) -> dict:
    return {"summary": ", ".join(sorted(state["notes"]))}


def build_fanout_graph():
    graph = StateGraph(FanState)
    graph.add_node("count_boundaries", count_boundaries)
    graph.add_node("count_dots", count_dots)
    graph.add_node("summarise", summarise)
    graph.add_edge(START, "count_boundaries")
    graph.add_edge(START, "count_dots")
    graph.add_edge(["count_boundaries", "count_dots"], "summarise")
    graph.add_edge("summarise", END)
    return graph.compile()
```

**Step 3 — Write the tests.** Create `learning/test_day07_loop.py`:

```python
from toy.domain import Ball
from toy.fanout_graph import build_fanout_graph
from toy.retry_graph import build_retry_graph


def run_with_script(script):
    app = build_retry_graph()
    return app.invoke({"script": script, "tries": 0})


def test_right_first_time():
    final = run_with_script(["right"])
    assert final["status"] == "accepted"
    assert final["tries"] == 1


def test_wrong_once_is_fixed():
    final = run_with_script(["wrong", "right"])
    assert final["status"] == "accepted"
    assert final["tries"] == 2


def test_loop_never_runs_a_third_time():
    final = run_with_script(["wrong", "wrong", "right"])
    assert final["status"] == "withheld"
    assert final["tries"] == 2


def test_write_answer_runs_at_most_twice():
    app = build_retry_graph()
    writes = 0
    for step in app.stream({"script": ["wrong", "wrong", "wrong"], "tries": 0}):
        if "write_answer" in step:
            writes += 1
    assert writes == 2


def test_fan_out_runs_both_and_joins():
    balls = [Ball("V Kohli", 4), Ball("V Kohli", 0), Ball("V Kohli", 6), Ball("V Kohli", 0)]
    final = build_fanout_graph().invoke({"balls": balls})
    assert len(final["notes"]) == 2
    assert final["summary"] == "boundaries: 2, dots: 2"
```

`test_loop_never_runs_a_third_time` is the key test. The script's third answer is `"right"`,
so the only way to pass is to stop after two tries.

**Step 4 — Run them.**

```bash
cd /d/Cricket/cricketmind-v3-work/cricketmind-v3/backend
uv run pytest ../learning
uv run ruff check ../learning
```

You should see `26 passed`.

**Step 5 — Break it twice.**

- In `route_after_check`, change `>=` to `>`. Two tests fail, because the loop ran a third
  time. Put it back.
- In `fanout_graph.py`, change the `notes` line to `notes: list[str]`. The fan-out test fails
  with `InvalidUpdateError: At key 'notes': Can receive only one value per step`. Combining
  those two values is the reducer's job. Put it back.

**Done when:** the 5 new tests pass, and the journal says why the limit belongs in your
router and not in `recursion_limit`.

**Finish the day:** journal entry, then:

```bash
cd /d/Cricket/cricketmind-v3-work
git add cricketmind-v3 && git commit -m "day 7: bounded retry loop and fan-out"
```

---

#### Day 8 — The real skeleton, guarded by a test

**Goal:** read the v3 architecture once, then create the five real layers, plus a test that
fails the moment an import points the wrong way.

**Why it matters for v3:** from Day 9, every file goes into one of these layers. A wrong
import is easy to make and hard to undo months later. This test catches it the same minute.

**Learn (about 2 hours): read the v3 architecture, slowly.** Open
`docs/CricketMind_Architecture_v3.pdf`.

| Read carefully | Skim |
|---|---|
| What Changes in Version 3, and Why | Node-by-Node Changes |
| Design Invariants | Control Loops and Bounds |
| System Architecture Diagram | Evaluation Plan; Build Plan |
| The Claim Contract | Risks and Fallbacks |
| Faith Check v3 | Positioning against Related Work |
| What This Document Does Not Claim | |

Then close the PDF and write **10 lines in the journal, in your own words**:

1. What went wrong in v2 that v3 fixes.
2. What a claim is, and why it is data rather than a sentence.
3. What M, D and P mean.
4. What "subject-keyed" means (use the Kohli and Rohit example).
5. The question each check, C1 to C6, asks.
6. One kind of fabrication (F1 to F7) that v2 could not catch.
7. Which engine counts the past, and which one simulates the future.
8. What an invariant is, with one example.
9. One thing the document says it does **not** claim.
10. The part you understand least. That is what to ask about.

**Step 1 — Create the five layers.**

```bash
cd /d/Cricket/cricketmind-v3-work/cricketmind-v3/backend/src/cricketmind_v3
mkdir domain application infrastructure pipeline api
touch domain/__init__.py application/__init__.py infrastructure/__init__.py pipeline/__init__.py api/__init__.py
```

| Folder | In the restaurant | What it will hold | What it may import |
|---|---|---|---|
| `domain/` | the chef | claims, the registry, wording bands, checks C1–C6 | the standard library only |
| `application/` | the kitchen manager | ports, use cases, engines, the graph | `domain/`, plus libraries such as LangGraph |
| `infrastructure/` | the suppliers | DuckDB, Ollama, video, classifier, pose, SQLite | `application/` and `domain/` |
| `pipeline/` | the night shift | Cricsheet ingest, attributes, the shot bridge | the three layers above |
| `api/` | the waiter | the FastAPI app, routers, schemas | everything above |

**Step 2 — Check that the package imports.**

```bash
cd /d/Cricket/cricketmind-v3-work/cricketmind-v3/backend
uv run python -c "import cricketmind_v3.domain; print('ok')"
```

You should see `ok`.

**Step 3 — Set up the test folders.**

```bash
mkdir tests/unit
touch tests/__init__.py tests/unit/__init__.py
```

The empty `__init__.py` files let two test files share a name in different folders later
(`unit/test_claims.py` and `graph/test_claims.py`). They also let tests import shared helpers
as `from tests.fakes import …`. The `integration/`, `graph/` and `api/` folders are made when
their first test arrives (Days 39, 53 and 76).

**Step 4 — Write the architecture test.** Create `backend/tests/unit/test_architecture.py`:

```python
import ast
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "cricketmind_v3"


def imports_in(py_file: Path) -> list[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                names.append("(relative import)")  # we only use absolute imports
            else:
                names.append(node.module)
    return names


def imports_in_layer(layer: str) -> list[tuple[str, str]]:
    found = []
    for py_file in sorted((PACKAGE / layer).rglob("*.py")):
        for name in imports_in(py_file):
            found.append((py_file.name, name))
    return found


def test_domain_uses_only_the_standard_library():
    bad = []
    for file_name, name in imports_in_layer("domain"):
        top = name.split(".")[0]
        if top in sys.stdlib_module_names:
            continue
        if name.startswith("cricketmind_v3.domain"):
            continue
        bad.append(file_name + " imports " + name)
    assert bad == []


def test_application_never_imports_outer_layers():
    outer = ["cricketmind_v3.infrastructure", "cricketmind_v3.pipeline", "cricketmind_v3.api"]
    bad = []
    for file_name, name in imports_in_layer("application"):
        if name == "(relative import)":
            bad.append(file_name + " uses a relative import")
            continue
        for layer in outer:
            if name.startswith(layer):
                bad.append(file_name + " imports " + name)
    assert bad == []


def test_every_layer_exists():
    for layer in ["domain", "application", "infrastructure", "pipeline", "api"]:
        assert (PACKAGE / layer / "__init__.py").exists(), layer
```

How it works: `ast` reads each `.py` file as a tree, *without running it*, and lists its
imports. The tests compare that list with the table in Step 1.

It also enforces one house rule: **absolute imports only**. Write
`from cricketmind_v3.domain.claims import Claim`, never `from .claims import Claim`, so every
import says plainly which layer it comes from.

**Step 5 — Run it.**

```bash
uv run pytest
uv run ruff check .
```

You should see `4 passed` (the 3 new tests plus `test_first.py`).

**Step 6 — Prove it works by breaking it twice.** A test you have never seen fail proves
nothing.

- Create `src/cricketmind_v3/domain/oops.py` containing the line `import fastapi`. Run
  `uv run pytest`. It fails with `['oops.py imports fastapi']`. Delete the file.
- Create `src/cricketmind_v3/application/oops.py` containing the line
  `from cricketmind_v3.infrastructure import duckdb_repo`. Run `uv run pytest`. It fails.
  Delete the file.
- Run `uv run pytest` once more. You should see `4 passed` again.

**Done when:** `uv run pytest` shows `4 passed`, both broken versions failed and then passed
again once you deleted them, and the 10-line journal entry is written.

**For the paper:** one decision row in `thesis/decisions.md`. D-014: the layer rule is
enforced by `tests/unit/test_architecture.py`, and `domain/` is standard-library only.
Paper section: Implementation.

**Finish the day:** journal entry, then:

```bash
cd /d/Cricket/cricketmind-v3-work
git add cricketmind-v3 && git commit -m "day 8: five layers and the architecture test"
```

#### End of Phase 0 — check before Day 9

From `backend/`:

| Command | You should see |
|---|---|
| `uv run pytest` | `4 passed` |
| `uv run pytest ../learning` | `26 passed` |
| `uv run ruff check . ../learning` | `All checks passed!` |
| `git -C /d/Cricket/cricketmind-v3-work log --oneline -8` | your eight day commits |

If all four match, the workshop is ready. Phase 1 builds the first real code, in `domain/`.

---

### Phase 1 — Claims (Days 9–15)

**In simple words:** a claim is one fact an agent wants to say, written as **data**, not as a
sentence. Data can be checked; sentences can only be read.

**Example:**

```python
claim = Claim(
    id="c1",
    kind=Provenance.DESCRIPTIVE,
    metric="strike_rate",
    subject=Subject(player="Kohli", shot="Cover Drive",
                    phase="powerplay", bowler_type="pace"),
    value=142.0,
    n=50,
)
template = "Against pace in the powerplay, Kohli's cover drive returns {c1}."
```

The renderer turns that into *"…returns a strike rate of 142.0 [D, 50 balls]."*

| Day | Learn | Build | Done when |
|---|---|---|---|
| 9 | Enums: a fixed list of choices prevents typos | `domain/provenance.py` (MEASURED, DESCRIPTIVE, PREDICTIVE); `domain/subject.py` | Tests: an unknown provenance is impossible; equal subjects compare equal |
| 10 | Rules an object must obey | `domain/claims.py`: `Claim`; a predictive claim needs an interval; a measured claim needs a clip id | Test: each broken rule raises an error |
| 11 | Finding `{c1}` in text with plain string methods | `domain/findings.py`: `Finding` (agent, claims, template); list its references | Tests: references found; a reference to a missing claim is rejected |
| 12 | Formatting numbers for people | `domain/render.py`: claim → "a strike rate of 142.0 [D, 50 balls]" | One test per metric type, including intervals |
| 13 | A registry: one place that lists what is allowed | `domain/registry.py`: `ClaimType` (name, provenance, source, required subject keys) and the 9 types of Table 4 in the v3 doc | Tests: lookup works; an unknown type returns nothing |
| 14 | Failing when the system is built, not while it runs | Registry validation: every claim type an agent declares exists and maps to an engine metric | Test: a hollow claim type ("the pitch is holding pace") fails validation |
| 15 | Review day | Rename, add docstrings, tidy | All tests green; you can explain Claim, Finding and Registry to a friend |

---

### Phase 2 — Faith Check v3 (Days 16–26)

**In simple words:** the Faith Check is the editor. It runs six checks, and each check asks one
question:

| Check | The question it asks |
|---|---|
| C1 | Is this agent allowed to say this kind of thing? |
| C2 | If I ask the engine myself, for exactly this subject, do I get the same number? |
| C3 | Is the evidence big enough to say this? |
| C4 | Do the words match the uncertainty? |
| C5 | Is there any number in the sentence that did not come from a claim? |
| C6 | Does the sentence talk about the same player and shot as the claim? |

**Example:** an agent writes *"Rohit's cover drive returns {c1}"*, but `c1` is about Kohli. C6
says: *"the sentence names Rohit, the claim is about Kohli"*. The finding goes back to the
agent with that reason.

| Day | Learn | Build | Done when |
|---|---|---|---|
| 16 | Words for probability (Kent's scale) | `domain/wording.py`: the five bands, `band_for(p)`, the hedge rule | Tests at the edges: 0.13, 0.40, 0.60, 0.87 |
| 17 | A pipeline of checks | `domain/faith/gate.py`: `Failure` (agent, claim, check, reason); `run_checks()` runs C1→C6 and collects failures | Test with two dummy checks |
| 18 | — | C1: schema, registry, agent permission | Tests: unknown type; agent not allowed; no source |
| 19 | Ports again: ask an engine without knowing which engine | `application/ports.py`: `FactSource`; `FakeFactSource`; C2 re-query with rounding tolerance | Tests: wrong value (F1) and wrong subject (F2) rejected |
| 20 | Why small samples lie: flip a coin 6 times | C3: evidence floor (30 balls), small-sample qualifier, generalising words ("struggles", "always") | Test: "struggles against spin" on 6 balls rejected; with "on only 6 balls" accepted |
| 21 | What an interval means, in one picture | C4 part 1: wording must fit the band; a hedge is needed if the interval crosses bands | Test: p = 0.405 with "will probably get him out" rejected |
| 22 | "No detectable difference" | C4 part 2: a comparison whose interval contains 0 must say so | Test: 142 vs 112 with interval [−16, +76] |
| 23 | Scanning text for digits | C5: no orphan numbers (allowlist: T20, U-19) | Test: "about 150" in a template rejected |
| 24 | A gazetteer: a list of known names | C6: names in a sentence must match the claim's subject | Test: the Rohit/Kohli swap rejected |
| 25 | Testing by breaking things on purpose; the results ledger | Fabrication harness: build the F1–F7 versions of a correct finding; a "v2 mode" gate that checks values only. Results go to `thesis/results/`, with one row per run in `ledger.jsonl` | Table: v2 mode catches F1 only; v3 catches F1–F7 (on fake data); E-004 verified |
| 26 | Review day — **M1** | Tidy, then a journal teach-back | **M1:** the editor works on fake data |

---

### Phase 3 — Real cricket data (Days 27–37)

**In simple words:** Cricsheet gives every ball of thousands of matches as free JSON files. You
turn them into **one table with one row per ball.**

**Example:**

```
JSON:  {"batter": "V Kohli", "bowler": "JJ Bumrah",
        "runs": {"batter": 4, "extras": 0, "total": 4}}

row:   match_id=1426279, innings=1, over=3, ball=2,
       batter="V Kohli", bowler="JJ Bumrah",
       runs_batter=4, is_wide=False, phase="powerplay"
```

| Day | Learn | Build | Done when |
|---|---|---|---|
| 27 | Reading Cricsheet JSON by hand | Download 5 matches; write notes on `info`, `innings`, `overs`, `deliveries`, `extras`, `wickets`, `review` | Journal: a hand-drawn diagram of one delivery |
| 28 | SQL basics with DuckDB: create, insert, select, group by | `learning/day28_duckdb.py` exercises | Tests on a 10-row table |
| 29 | Designing a table (a schema) | `pipeline/schema.py`: the delivery row; the phase rule per format | Test: a hand-made delivery becomes the right row |
| 30 | Wides and no-balls (why ball numbers get tricky) | `pipeline/parse_match.py`: one match → rows, with plain loops | Tests on 2 fixture matches with a wide, a no-ball and a wicket |
| 31 | Resumable jobs: a manifest plus atomic writes | `pipeline/ingest.py`: all files → DuckDB, skipping finished files | Test: running twice adds nothing; stopping and restarting works |
| 32 | Checking data against itself | Checks: innings totals equal the sum of the balls; match counts; empty fields | Report printed; tests on fixtures |
| 33 | Matching names (aliases) | The player table from the Cricsheet register; name search for N3 | Test: "Kohli" finds "V Kohli" |
| 34 | Where batting hand and bowling type come from | Route A: export from the `cricketdata` package (needs R, once). Route B: hand-label the players in your evaluation set | Coverage report: the share of balls with a bowler type |
| 35 | — | Finish the attribute join (a spillover day if Day 34 runs long) | Test: the join keeps every delivery |
| 36 | Derived tables | The venue table; balls bowled per bowler per date; the reviews table | Tests on fixtures |
| 37 | Review day | Row counts and file size into the journal | The numbers are written down |

---

### Phase 4 — Engines (Days 38–52)

**In simple words:** engines are the only calculators allowed to produce numbers. **N8 counts
the past. N9 simulates the future and says how unsure it is.**

**Example: bootstrap.** Is 142 versus 112 a real difference? Pretend to replay each shot's 50
balls thousands of times, picking balls at random (repeats allowed), and watch how much the
difference jumps around. If the range includes 0, you cannot say one shot is better.

**Example: the Beta distribution.** 4 outs in 50 balls. "8% per ball" is a guess, not a fact.
The Beta distribution is a curve showing how believable each possible rate is. A curve built
from 50 balls is wide; one from 5,000 balls is narrow.

| Day | Learn | Build | Done when |
|---|---|---|---|
| 38 | Rates and sample size: simulate coin flips | `learning/day38_coins.py` | Journal: why 2 balls tell you nothing |
| 39 | The repository pattern with real SQL | `DeliveryRepository` port, DuckDB adapter, fake; M1: strike rate, dismissal rate, n | Tests on the fake and on a tiny DuckDB file |
| 40 | Bootstrap (resampling) | M1: a difference with a 95% interval, fixed seed | Test reproduces [−16, +76] on the synthetic example |
| 41 | — | M2: venue profile (first-innings average, chasing wins, pace and spin wicket share) | Tests |
| 42 | The workload ratio, and why scientists argue about it | M3: balls bowled in 7 and 28 days, the ratio, a flag (never a probability) | Test with a made-up bowler calendar |
| 43 | — | M4: DRS reviews, counts and overturn rate | Tests |
| 44 | Connecting the editor to the real engine | The real `FactSource`: registry metric → engine function | C2 test against the tiny DuckDB |
| 45 | Monte Carlo, and the Beta distribution | `learning/day45_beta.py`: draw rates, plot the curve | Journal: the curve for 4 outs in 50 balls |
| 46 | — | N9.T1: chance of being out within k balls, with estimation uncertainty | Test: median ≈ 0.405, interval ≈ [0.155, 0.697] (seeded) |
| 47 | Shrinkage: pulling small samples toward the average | A population prior from the data | Test: 1 out in 2 balls is not a 50% risk |
| 48 | — | N9.T2 part 1: head-to-head rates with shrinkage; baseline and counterfactual | Tests |
| 49 | — | N9.T2 part 2: the simulated difference, its interval, "no detectable difference" | Tests |
| 50 | Calibration (the weather-forecast example); the Brier score | `learning/day50_calibration.py` | Journal: what "70% means 70%" means |
| 51 | — | Calibration run: fit on older seasons, test on the newest | Reliability plot and Brier score saved |
| 52 | Review day — **M2** | The engine report | **M2:** engines on real data |

---

### Phase 5 — The graph (Days 53–74)

**In simple words:** LangGraph is a **flowchart that runs**. The **state** is a shared notebook
that every node reads and writes. You build the whole flowchart with stubs first, then make
each node real, one at a time.

**Example:** the question *"Kohli vs pace in the powerplay — cover drive or pull?"* flows:
N3 parses it → N6 makes the Shot DNA → N7a fetches the balls → N8 and N9 compute → N10 picks
Tactical Analysis and Personal Performance → each agent writes claims → Faith Check → N12
writes the report → N13 shows it.

| Day | Learn | Build | Done when |
|---|---|---|---|
| 53 | State and reducers | `application/graph/state.py`; the full skeleton with every node as a stub (node name = function name) | Test: the graph compiles and runs end to end |
| 54 | — | The entry branch (clip or text only); N3 with a simple rule-based parser first | Tests for both paths |
| 55 | What a local LLM is; JSON-schema output | `LLM` port, `OllamaLLM` adapter, `FakeLLM` | Test with `FakeLLM`; a manual run with qwen2.5:7b |
| 56 | — | N3 for real: structured parse → Pydantic; an unknown player gets a clear status | Tests with the fake; 5 manual queries |
| 57 | — | N6 Shot DNA (stub perception); N7a subject-keyed retrieval; N7b context | Tests |
| 58 | — | N8 and N9 as graph nodes; the sanctioned fact surface | Tests |
| 59 | Capabilities: what can this question support? | Capability filter: posture needs a clip, matchups need two bowlers, venue claims need a venue | Tests |
| 60 | — | N10 router part 1: keyword rules, capability filter, cap of 4, default pair | Tests |
| 61 | — | N10 router part 2: the LLM proposes, using an evidence summary; rules as the fallback | Tests with a scripted `FakeLLM` |
| 62 | Writing good prompts | Agent spec: stakeholder, allowed claim types, voice; one file per agent in `prompts/` | All 12 prompt files written |
| 63 | — | Agent factory: LLM → `Finding` (claims + template) under a JSON schema | Tests with scripted findings |
| 64 | — | The first real agent: Tactical Analysis (N11b), with the real LLM | 5 real findings pass C1–C6; journal notes |
| 65 | — | N11a Club Operations, N11c Personal Performance, N11d Global Scouting | Tests |
| 66 | — | N11e Fan Concierge, N11f Stadium Operations (uses M2), N11g Bowling Performance (new, uses T2) | Tests |
| 67 | — | N11h Content Production, N11i Match Officials (uses M4), N11j Talent Development | Tests |
| 68 | — | N11k Injury Management (M3 flag), N11l Player Representation | The registry check passes for all 12 |
| 69 | Fan-out and collecting results | Dispatch to the chosen agents; collect their findings | Tests |
| 70 | — | The Faith gate node; regenerate only the failing agents, with reasons; budget of 2 | Tests: lie-once is fixed; lie-always is withheld |
| 71 | — | The withhold path; N12 synthesis (claim references only) with a fallback; N13 provenance tags | Tests |
| 72 | — | End-to-end tests: 6 text-only scenarios with fakes | All pass |
| 73 | — | 10 runs with the real LLM; fix prompts | Journal: what failed and why |
| 74 | Counting cost | Count steps and agent calls; write the worst-case bound; test it | **M3:** text answers work in the terminal |

---

### Phase 6 — The web API (Days 75–86)

**In simple words:** the API is the **waiter** between the website and the kitchen.

**Example conversation:**

```
website  →  POST /api/queries  {"text": "Kohli vs pace: cover drive or pull?"}
API      ←  {"job_id": "q_17"}
website  →  GET  /api/queries/q_17/events     (stays open)
API      ←  "N3 done" … "N8 done" … "Faith Check: 1 agent regenerated" … "done"
website  →  GET  /api/queries/q_17
API      ←  { the full checked report }
```

| Day | Learn | Build | Done when |
|---|---|---|---|
| 75 | Contract first: decide the shapes before the code | Pydantic request and response schemas | Visible on the `/docs` page |
| 76 | The app factory; settings from `.env`; logging | `api/main.py`; `GET /api/health` | Test: health returns ok |
| 77 | The composition root: wire ports to adapters in one place | `infrastructure/container.py` | Test: fakes swapped in for tests |
| 78 | — | `POST /api/queries` for text questions, synchronous first | Test with fakes |
| 79 | Background jobs: why slow work must not block a request | A job store; POST returns a job id; `GET /api/queries/{id}` | Tests |
| 80 | Server-Sent Events: one-way live updates | `GET /api/queries/{id}/events` streams node progress | Stream test |
| 81 | Saving data with SQLite and SQLModel | A reports table; a history endpoint | Tests |
| 82 | Uploading files safely | `POST /api/clips` with type and size limits | Tests: too big and wrong type rejected |
| 83 | — | `GET /api/players?q=koh` for autocomplete | Tests |
| 84 | Errors people can understand | One error format for 404, 422 and 500 | Tests |
| 85 | Security basics: CORS, limits, secrets | CORS for the dev site; `.env` never committed | Tests |
| 86 | Review day | Export `openapi.json`; try every endpoint by hand | Journal |

---

### Phase 7 — The website (Days 87–111)

**In simple words:** React builds pages from small reusable components, like LEGO bricks.
TypeScript tells you about mistakes before you run the page.

**Example component:**

```tsx
type Props = { kind: "M" | "D" | "P" };

// M = measured, D = descriptive, P = predictive
export function ProvenanceBadge({ kind }: Props) {
  return <span className="rounded-full border px-2 text-xs">{kind}</span>;
}
```

| Day | Learn | Build | Done when |
|---|---|---|---|
| 87 | JavaScript for Python people: let/const, functions, arrays, objects, async/await | Small exercises in `frontend/learning/` | They run with Node |
| 88 | TypeScript basics: types, unions, optional fields | Typed versions of yesterday's exercises | `tsc` reports no errors |
| 89 | fetch and promises | A script that calls `/api/health` | It prints ok |
| 90 | React 1: components and props | `ProvenanceBadge` and `ClaimLine` in a playground | They render |
| 91 | React 2: state, events, lists | A toy question form | Typing updates the screen |
| 92 | React 3: effects and loading states; why TanStack Query exists | Fetch health with a spinner | Spinner, then "ok" |
| 93 | — | Vite + React + TypeScript + Tailwind + shadcn/ui + React Router + linter | The home page renders |
| 94 | Types generated from the API | `openapi-typescript` + `openapi-fetch` client; TanStack Query provider | The health check uses the typed client |
| 95 | — | Layout and pages: Ask, Live, Report, History, How it works | Navigation works |
| 96 | Forms | Ask page part 1: the question and player autocomplete | Validation messages show |
| 97 | Uploads with progress | Ask page part 2: optional clip upload | The progress bar moves |
| 98 | — | Submit → job id → the Live page | It navigates with the id |
| 99 | EventSource (the browser side of SSE) | The Live page: a node timeline showing retries and regenerations | The timeline fills in live |
| 100 | — | Report page part 1: findings per agent; claims with M/D/P badges, sample size, intervals | A saved report renders |
| 101 | — | Report page part 2: withheld findings, and which check failed | "Why?" expands |
| 102 | — | The History page | The list opens a report |
| 103 | — | The How-it-works page: the v3 diagram, explained simply | It renders |
| 104 | — | Errors, empty states, skeletons while the LLM is slow | Each state can be seen |
| 105 | Testing components: Vitest + Testing Library | Tests for the badge, the form and the report | Green |
| 106 | Mocking | A Live-timeline test with a fake EventSource | Green |
| 107 | End-to-end tests with Playwright | Ask → report, against the backend in fake mode | Green |
| 108 | Accessibility and small screens | Labels, keyboard use, contrast, phone layout | Checklist done |
| 109 | — | Polish; production build | `npm run build` works |
| 110 | Review day — **M4** | Give yourself a demo | **M4:** the website answers text questions |
| 111 | — | A spillover day for frontend learning | — |

---

### Phase 8 — Video: classifier and pose (Days 112–127)

**In simple words:** the video path turns a clip into *"this is a Pull, 92% sure"* and measures
the batter's body.

**Example:** a clip → drop repeated frames → crop around the batter → 15 frames →
classifier → "Pull, 0.92". Pose → dots on the joints → "weight transfer 0.31, on the back foot".

**When:** this phase comes late on purpose. The classifier comes from CricShot, and by then
its final model will be ready. Until it is, a stub stands in for it, and nothing else waits.

| Day | Learn | Build | Done when |
|---|---|---|---|
| 112 | A contract between two projects | In the CricShot repo: export the final classifier (TorchScript), `classes.json`, a model card, and the frame numbers it picks for the 20 parity clips. Copy them into CricketMind's `models/` with a checksum | Checksum matches; the model card states the exact input, step by step (see the note below) |
| 113 | — | The device helper: XPU → CUDA → CPU | Runs on both machines |
| 114 | Reading video frames | N1: decode the clip and pick frames exactly as the model card says | Test on a tiny clip |
| 115 | — | The striker crop, following the model card | The crops look right on 5 clips |
| 116 | Parity tests: same input, same output | Compare your frames and crops with CricShot's processed clips on the 20 parity clips | The same frame numbers are picked, and the pixel difference is under the agreed limit |
| 117 | — | The classifier adapter: label and confidence | Stub test, plus one real run |
| 118 | — | The confidence gate; retry up to 2 times; the low-confidence exit | Tests |
| 119 | — | The video path wired into the graph and the API | End to end with a real clip |
| 120 | Pose keypoints: 17 dots on the joints | Run pose on one frame and draw it | Picture saved |
| 121 | — | N2b on all 15 frames; pick the striker; handle missing joints | Tests |
| 122 | Turning dots into measurements | Weight transfer, stride length, head over front knee | Tests on hand-made keypoints |
| 123 | — | Posture validation run on the CricShot10k test clips, resumable | All clips done |
| 124 | ROC-AUC, in simple words | AUC per metric; keep a metric only if AUC ≥ 0.75 | Report written; registry updated |
| 125 | — | Measured posture into N6; N11c and N11j cite it | Tests |
| 126 | — | The website shows the clip, the predicted shot, its confidence and the posture | Tests |
| 127 | Review day — **M5** | Journal | **M5:** the video path works |

**Which frames is part of the contract.** Never assume "15 evenly spaced frames from the raw
clip". On 2026-10-16, CricShot's pipeline was: read the clip as RGB → drop repeated frames
(some clips have copies inserted by frame-rate conversion) → find the batter → crop, keeping
only the camera shot with the most detections → only then take 15 evenly spaced frames from
what is left. The final model may want a different count too (video models often take 16).
Get any step wrong and nothing crashes: the classifier just sees inputs it never trained on,
and answers worse. CricShot learned this the hard way: the authors' own model, fed crops made a
different way, lost more than half its accuracy. So N1 follows the model card, never the
numbers in this plan, and Day 116 checks the frame numbers exactly, because a frame or two off
can hide inside a pixel tolerance.

The website shows a prediction for a clip, but never the classifier's accuracy.

---

### Phase 9 — The commentary-to-shot bridge (Days 128–142)

**In simple words:** Cricsheet knows what happened on each ball, but not which shot was played.
Commentary text often says it. You teach the computer to read it.

**Example:** *"Bumrah to Kohli, FOUR, drives through cover"* → **Cover Drive**.

**A go/no-go gate comes first.** If no commentary corpus you are allowed to use exists, or if
the labels turn out too unreliable, you take the fallback path on Day 138. That is still a
result you can report.

| Day | Learn | Build | Done when |
|---|---|---|---|
| 128 | Licences, and go/no-go decisions | Find a commentary corpus you may use, with match/over/ball keys; read the licence; check the columns | GO or NO-GO written in the journal. NO-GO → Day 138 |
| 129 | — | Keyword and synonym rules for the 15 shots ("cuts" → Square Cut, "reverse-sweeps" → Reverse Sweep) | Tests |
| 130 | — | The rules labeller on the whole corpus | Coverage numbers in the journal |
| 131 | — | An LLM labeller for the lines the rules miss: 15 classes or "none", temperature 0; resumable | `FakeLLM` tests |
| 132 | Sampling fairly | A tiny labelling tool; a stratified sample of 500 lines | The tool works |
| 133 | — | Hand-label 250 lines | Saved |
| 134 | — | Hand-label 250 more | 500 saved |
| 135 | Cohen's kappa: agreement beyond luck | κ and per-class F1 | Decision: is κ ≥ 0.61? |
| 136 | — | Join labels to deliveries (match, innings, over, ball; careful with wides and no-balls) | Join-rate report; tests |
| 137 | — | Shot labels, with confidence, into the delivery table; N7a uses them | Tests |
| 138 | — | The fallback path: a hand-labelled evaluation set only | Tests |
| 139 | — | Re-run the engines with shot labels; spot-check | Journal |
| 140 | Review day — **M6** | The bridge report (a positive or negative result) | **M6** |
| 141–142 | — | Spillover for labelling and join problems | — |

---

### Phase 10 — Evaluation for the thesis (Days 143–154)

**In simple words:** an exam for your system. Each experiment asks one question and has a
pass rule written down *before* you run it.

| Day | Learn | Build | Done when |
|---|---|---|---|
| 143 | Designing experiments | The evaluation plan: question, method, pass rule for each | Written |
| 144 | — | Fabrication containment F1–F7: v2 mode vs v3 | Catch-rate table |
| 145 | — | False rejections: how often correct findings are wrongly stopped | Rate and examples |
| 146 | — | The cost of containment: vary how often agents lie | Plot |
| 147 | Writing good test questions | Routing set: the first 50 labelled questions | Saved |
| 148 | — | Routing set: 50 more, then check the labels | 100 questions |
| 149 | — | Keyword router vs LLM router, 3 runs each | Accuracy and stability |
| 150 | — | Engine validation: calibration plots and spot checks | Figures |
| 151 | — | Measured cost vs the formula, across scenarios | Table |
| 152 | — | All figures and tables in one style | `docs/results/` |
| 153 | Honest limits | Threats to validity; what not to claim | Written |
| 154 | Review day — **M7** | Journal | **M7:** thesis results ready |

---

### Phase 11 — Ship it (Days 155–162)

**In simple words:** a Docker container is a shipping box for software. Everything the app
needs goes inside, so it runs the same on any computer.

| Day | Learn | Build | Done when |
|---|---|---|---|
| 155 | Containers, simply; install Docker Desktop | Dockerfiles for the backend and the frontend | Both images build |
| 156 | — | `docker-compose.yml`; Ollama stays on the host | One command starts everything |
| 157 | — | Production settings; smoke test from a clean start | It works |
| 158 | Writing for other people | README (a 10-step setup), architecture notes, decision records | A friend can run it |
| 159 | — | The demo script: a text question, a clip question, and the Faith Check catching a planted lie | Rehearsed |
| 160 | — | Record the demo video | Saved |
| 161 | — | Final checklist; tag v1.0 | **M8:** shipped |
| 162 | — | Buffer | — |

---

### Phase 12 — Write the paper (Days 163–177)

**In simple words:** by now the paper's pieces already exist — decisions, numbers, evidence,
limitations, references and rough drafts. This phase assembles them and polishes. The
introduction and abstract are written last, once you know exactly what you found.

| Day | Learn | Build | Done when |
|---|---|---|---|
| 163 | How papers in your target venue are written | Choose the venue and its template; read 3 recent papers from it; set word budgets in `outline.md` | Venue, template and budgets decided |
| 164 | — | Final check of figures and tables (all rebuilt by script); `evidence.md` and `numbers.md` complete | Every planned claim is verified or dropped |
| 165 | Writing a method section | Method part 1: provenance classes, the claim contract, the registry | Draft done |
| 166 | — | Method part 2: Faith Check C1–C6, the engines and their uncertainty, the evidence pipeline, perception (no accuracy) | Draft done |
| 167 | — | Implementation: clean architecture, the API, the website | Draft done |
| 168 | — | Evaluation setup: questions, data, metrics, pass rules | Draft done |
| 169 | Writing results: say what you measured, not what you hoped | Results part 1: containment, false rejections, cost of containment | Draft done |
| 170 | — | Results part 2: calibration, posture, bridge, routing, cost bounds; discussion | Draft done |
| 171 | Positioning against related work | Related work from `related_work.md` | Draft done |
| 172 | — | Introduction and abstract (written last, on purpose) | Draft done |
| 173 | — | Limitations (from `limitations.md`), data licences and ethics, conclusion | Full draft done |
| 174 | Be the paper's Faith Check | Check every number against `numbers.md` and every claim against `evidence.md`; no classifier accuracy anywhere | Zero unsupported numbers or claims |
| 175 | — | Revise after your supervisor's feedback | Feedback addressed |
| 176 | — | Polish: language, references, formatting, page limit | Fits the template |
| 177 | — | Final read-through; a submission-ready PDF; tag the commit | **M9:** the paper is ready |

---

## 12. When you get stuck

1. **The 30-minute rule.** Stuck for 30 minutes? Stop and write the problem down.
2. **Make it small.** Reproduce the error in a 10-line file in `learning/`.
3. **Read the error from the bottom up.** The last line says what went wrong; the lines above
   say where.
4. **Ask with everything attached:** the error, the smallest code that shows it, and what you
   already tried.
5. **A slow day is still a day.** If a step takes two sessions, that is what the spillover
   and buffer days are for.

Working with Claude: at the start of a day, ask *"explain Day N simply"*. Build it yourself,
then ask for a review of your code and tests. Commit once the "Done when" check passes.

---

## 13. Glossary

| Word | Meaning, simply |
|---|---|
| **Agent** | A part of the system that writes for one audience (coaches, fans…). It may only quote checked facts. |
| **API / endpoint** | A door into the backend. `POST /api/queries` is one door. |
| **Adapter** | Code that implements a port using a real tool (DuckDB, Ollama). |
| **AUC** | How well a number separates two groups. 0.5 is a coin flip; 1.0 is perfect. |
| **Beta distribution** | A curve of how believable each possible rate is, given what you have seen. |
| **Bootstrap** | Replaying your data many times by random picking, to see how much a result could change. |
| **Brier score** | A score for probability forecasts. Lower is better. |
| **Calibration** | When the system says 70%, it happens about 70% of the time. |
| **Claim** | One fact as data: what, about whom, the value, and how sure. |
| **Cohen's kappa (κ)** | How much two labellers agree beyond what luck would give. |
| **Component** | A reusable piece of a web page. |
| **Dependency injection** | Handing a function the tools it needs, instead of letting it build them, so tests can hand it fakes. |
| **Evidence map** | `thesis/evidence.md`: every claim the paper makes, and the file that proves it. |
| **Fake** | A pretend version of a tool, used in tests. |
| **Fan-out** | Running several nodes or agents at the same time. |
| **Gazetteer** | A list of known names: players, shots, venues. |
| **Ledger** | A file that gains one row per experiment run, written automatically, so no result is ever typed by hand. |
| **Monte Carlo** | Answering a question by simulating it many times with random numbers. |
| **Orphan number** | A number in a sentence that did not come from a checked claim. |
| **Port** | A promise about what a tool can do, written without naming the tool. |
| **Provenance** | Where a number came from: measured (M), descriptive (D) or predictive (P). |
| **Registry** | The list of claim types the system allows, and which engine backs each one. |
| **Shrinkage** | Pulling a small sample's result toward the average, so 1 out in 2 balls is not read as a 50% risk. |
| **SSE (Server-Sent Events)** | The server pushing live updates to the browser over one open connection. |
| **State (LangGraph)** | The shared notebook that every node reads and writes. |
| **Stub** | A placeholder that returns a fixed answer until the real part is built. |
