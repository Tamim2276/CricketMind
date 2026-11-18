# Cricket MLOps: project brief

A starting brief for a new project, written 2026-11-18. It is for **you** (Tamim) and for **the
new Claude chat** that will help you build it. Read it together in the first session.

The task came from your supervisor: read an MLOps paper, then build that kind of framework for
a part of cricket you choose. The paper is GreenAirOps (Section 3). The part of cricket chosen
here is **live T20 win probability** (Section 4).

---

## 0. How to start

**Steps 1 and 2 are done (2026-11-18):** the branch `cricket-mlops` exists at
`D:\Cricket\cricket-mlops-work\`, with no commits yet, holding only this file and the paper.
Do not run them again. Only step 3 is left.

1. **Make the new branch** as its own empty folder, so nothing from CricShot or CricketMind
   comes with it. Run this yourself (git 2.53 can do this):

   ```bash
   cd /d/Cricket/cricshot-improve
   git worktree add --orphan -b cricket-mlops ../cricket-mlops-work
   ```

   This gives `D:\Cricket\cricket-mlops-work\` on a new branch `cricket-mlops`, with no files.
   If git complains, show the message to the new chat before trying anything else.

2. **Move two files into it:**
   - this file, to `D:\Cricket\cricket-mlops-work\docs\PROJECT_BRIEF.md`
   - the paper, from `D:\Cricket\cricketmind-v3-work\cricketmind-v3\mlops.pdf` to
     `D:\Cricket\cricket-mlops-work\docs\papers\greenairops.pdf`. It is sitting untracked in
     the thesis folder now, and the next `git add cricketmind-v3` would commit it there.

3. **Open the new chat in that folder** and send it this as the first message:

   > Read `docs/PROJECT_BRIEF.md` and `docs/papers/greenairops.pdf`. Then, before any code:
   > write `CLAUDE.md` from Section 2, and turn Section 8 into `docs/BUILD_PLAN.md`, day by
   > day, for me to read. Ask me about anything in Section 10 you need settled first.

---

## 1. The project in one paragraph

A system that gives the chance each side wins a T20 match, updated after every ball, built the
way production machine learning is built: versioned data, tracked experiments, a model that
must pass a test before it is used, a web service in a container, automatic tests on every
change, monitoring, and retraining when the game changes. The research part is not "we built a
pipeline": it is **a measured answer to whether the MLOps parts help**. The game really did
change (the IPL's Impact Player rule, 2023, raised scoring), so we replay the past seasons in
order and compare a model that is never retrained with one retrained on a schedule and one
retrained when drift is detected.

---

## 2. Rules for the chat (turn this section into `CLAUDE.md`)

**Who the user is.** A CSE student at MIST, Bangladesh. A beginner, who learns by building the
real project and wants to understand every piece of its code.

**How to work.**
- Explain in very simple language, short, with an example.
- Claude writes the code one small piece at a time, and explains every line in simple words
  before it runs. The user reads, asks, then says "next".
- No practice exercises: each idea is taught on the real code the first time it is needed.
- The user runs commits, installs, downloads and long runs; Claude gives the exact commands.
  Claude may write project files, run tests and do quick read-only checks.
- After every finished step, hand over a commit command. No `Co-Authored-By` or any other
  attribution line in commits.
- Commit only this folder's work, path-limited. Never `git add -A`. Never switch branches.
- Do every plan step. Never suggest skipping or deferring one to save effort.
- Power cuts happen (load shedding): anything long must be resumable, and save as it goes.
- One GPU user at a time: CricShot may be training. Tree models run on the CPU, so this project
  should rarely need the GPU. If it does, code runs unchanged on Intel Arc (XPU), then CUDA,
  then CPU.

**Code style.**
- Plain, readable Python: loops and if/else, not comprehensions, lambdas or clever one-liners.
- Comments very short and human. Long explanations go in chat, not in the code.
- Tests with pytest; lint and format with ruff; the environment with uv (Python 3.12).

**Keep the projects apart.**
- Never edit, import from or read the code of **CricketMind v3** (branch `cricketmind-v3`,
  `D:\Cricket\cricketmind-v3-work`) or **CricShot** (`D:\Cricket\cricshot-improve`, branch
  `langGraph`). Each has its own chat.
- Download your own copy of the cricket data (Section 5). Do not read CricketMind's database.
- CricketMind's results never appear in this project's paper, and this project's results never
  appear in CricketMind's. CricShot's classifier accuracy appears in neither.

**Keep the record as you go.** The same habit as CricketMind:
- `notes/journal.md`: one entry per day.
- `docs/decisions.md`: each choice, why, and what else was considered.
- `docs/numbers.md`: no number in the paper without a row here saying how it was made.
- `docs/limitations.md`: each weakness, the day it is noticed.
- Any test rule (what counts as better) is written down **before** the run it judges.

---

## 3. The paper we are following

Goyal, Sharma, Hotchandani, Mahale, Gaikwad and Shahade (2026). *GreenAirOps: production-ready
MLOps for real-time air quality index prediction.* Air Quality, Atmosphere & Health 19:174.
https://doi.org/10.1007/s11869-026-02068-4. Code: https://github.com/vaibhav7766/mlops-climate.
Open access, CC BY-NC-ND 4.0 (cite it; do not copy its text or figures).

**What it builds**, as one loop:

```
fetch data (APIs, retries, cache) -> validate -> clean and make features -> store (DVC)
  -> train (Random Forest + XGBoost) -> evaluate -> passes the thresholds?
        no  -> log the failure, alert
        yes -> register (MLflow) -> build Docker image -> push to registry -> deploy
              -> serve (FastAPI) -> collect metrics -> watch for drift -> retrain?
```

| Part | Tool in the paper |
|---|---|
| Data and model versions | DVC (remote on S3) |
| Experiments, model registry | MLflow (Staging, Production, Archived) |
| Packaging | Docker (multi-stage), Docker Compose |
| Automation | GitHub Actions: lint, tests, build, security scan, retraining job |
| Serving | FastAPI: `POST /predict`, `GET /health` |
| Cloud | AWS (ECR, EC2) |
| Splitting | 80/20 by time, rolling-window cross-validation (no future data in training) |

**Where it is weak, and where this project should be stronger:**
- Its accuracy is poor: MAPE 236.87%, maximum error 337 (its Table 2).
- It **describes** drift monitoring, retraining and rollback, but never **measures** whether
  they help. There is no experiment with and without them.
- Its "validation" is a few example cities.
- So: copy its pipeline shape, and add the experiment it lacks.

---

## 4. What to build: live T20 win probability

**The model's question.** At this ball of this match, what is the chance the chasing side wins?

**Start small, then grow.**
1. **Version 1: the second innings only** (the chase). The state is simple and complete: the
   target, runs still needed, balls left, wickets in hand, and the run rates these give.
2. **Version 2: the first innings too**, which needs an idea of a par score, so it is harder.

**Which matches.** Men's T20 matches, with the IPL as the main case because of the 2023 rule
change. Leave out, at first, and write each down as a limitation: matches with no result, ties
decided by a Super Over, and matches whose target was changed by rain (DLS), since their target
is not the plain first-innings score plus one.

**What a feature may be.** Only what is known at that ball. Never the final score, the result,
or anything from later in the match or a later match. An "era" or "season" feature is left out
on purpose: the study is about whether retraining keeps up with change, and such a feature
would hide it.

**Models.** In this order, each judged the same way:
1. Logistic regression: the simple baseline.
2. Random Forest and XGBoost (or LightGBM), as in the paper.
3. A calibrated version of the best one (Platt or isotonic), fitted on a held-out later season.

**How to judge a probability.** Accuracy alone is not enough: a model that says 90% must be
right about 90% of the time.
- Brier score and log loss: the main numbers.
- A reliability plot, and the calibration error by season.
- Accuracy at 50%: reported, never the main number.

**How to split the data.** By season, in time order, never by random balls. Balls of one match
are near copies of each other, so a random split leaks the answer: one public project reports
random-split accuracy of 0.781 against 0.731 when split by time.

---

## 5. The data

- **Source:** Cricsheet (https://cricsheet.org), ball by ball. Licensed under the Open Data
  Commons Attribution License, so it must be credited. Some matches are withheld at the source:
  write that down when you load it.
- **A pinned snapshot:** the user downloads it. Record the download date and the file's
  SHA-256 in `docs/numbers.md`, and version it with DVC.
- **New data later:** Cricsheet adds matches as they are played. The ingestion step fetches
  only what is new, which is the cricket version of the paper's API polling, and the trigger
  for retraining.

**What is known so far, from the web, not yet checked on our data.** Treat every number here as
a lead to check, never as a result:
- IPL runs per innings rose from 155.5 (2022) to 166.7 (2023), the Impact Player rule's first
  season.
- Several public projects report that win-probability models trained before 2023 under-rate
  chasing sides afterwards, one by about seven percentage points. These are self-reported.
- One preprint (arXiv 2608.14696) reports that an era-adjusted model improved the Brier score
  from 0.1637 to 0.1489 on held-out seasons.

---

## 6. The research question and the experiment

**The question.** When the game changes, does an MLOps pipeline that detects drift and retrains
keep a win-probability model honest better than a model left alone or retrained on a timetable?
How quickly does it notice, and at what cost?

**The season replay.** Pretend the system went live years ago, then feed it the matches in the
order they were played:

| Policy | What it does |
|---|---|
| A. Never retrain | Trained once on the early seasons, then left alone |
| B. On a timetable | Retrained after every season (or every month) |
| C. On drift | Retrained when monitoring says the data or the errors have changed |
| D. Recency-weighted (optional) | Like B, with recent seasons counted more |

**What to measure, season by season:**
- The Brier score and the calibration error.
- How many matches after the 2023 rule change each policy took to react.
- How many retrains each policy needed (the cost), and how many it rolled back.

**The rules, written down before the replay runs:** the training start, the drift tests and
their thresholds, what counts as "better" for promotion, and how a difference is called real.

**Why cricket suits this better than air quality.** The answer to every prediction (who won)
is known within hours, so performance can be monitored directly, not only the inputs.

**Report whatever happens.** Drift outside the IPL may be small, and a timetable may do as well
as drift-triggered retraining. Either is a real result.

---

## 7. The pipeline and the tools

```
Cricsheet download (pinned, then new matches only)
  -> validate (schema and cricket rules; bad files logged, never silently dropped)
  -> features: one row per ball of a chase, saved as Parquet
  -> DVC tracks data and every stage (dvc.yaml, params.yaml)
  -> train, logged in MLflow (parameters, metrics, plots, the model)
  -> evaluate -> the gate: better Brier, calibration no worse, than the model in use?
        no  -> record why, keep the current model
        yes -> register as the new "champion" in MLflow
  -> FastAPI service, in Docker: POST /predict, GET /health, GET /model
  -> every prediction logged -> monitoring: input drift, and Brier once the result is known
  -> drift? -> retrain (back to the top); a worse model in use? -> roll back
```

| Need | Tool | Why |
|---|---|---|
| Environment | uv, Python 3.12 | Exact versions, fast installs (as in CricketMind) |
| Data storage | DuckDB and Parquet | Fast on one laptop; no server to run |
| Data checks | pandera, or plain checks first | Catch bad files before they reach a model |
| Data and pipeline versions | DVC, with a local remote first | Any past model can be rebuilt exactly |
| Experiments and registry | MLflow | The paper's choice; champion and challenger aliases |
| Models | scikit-learn, XGBoost (or LightGBM) | The paper's models, and a simple baseline |
| Drift reports | Evidently, or own tests (PSI, KS) first | The beginner learns what drift is before a library hides it |
| Service | FastAPI | The paper's choice |
| Packaging | Docker, Docker Compose | **Not installed yet**: the user installs Docker Desktop |
| Automation | GitHub Actions | The repository is on GitHub (`Tamim2276/CricketMind`); runs lint, tests, a tiny pipeline run, the image build |
| Cloud | Local first; AWS or another later, only if the supervisor wants it | Costs money; the paper's AWS part is optional for the research question |

**A folder layout to start from** (the new chat may change it, and should say why):

```
cricket-mlops-work/
  CLAUDE.md
  docs/            PROJECT_BRIEF.md, BUILD_PLAN.md, decisions.md, numbers.md,
                   limitations.md, papers/greenairops.pdf
  notes/journal.md
  data/            (DVC-tracked, never in git)
  src/cricket_mlops/
    ingest/        download, validate, load
    features/      one row per ball
    train/         models, the gate, the registry
    serve/         the FastAPI app
    monitor/       drift tests, performance by season
    replay/        the season replay (the experiment)
  tests/
  dvc.yaml  params.yaml  Dockerfile  docker-compose.yml  .github/workflows/
```

---

## 8. A first plan, by phase (to become `docs/BUILD_PLAN.md`)

One "day" is one session of about 4-5 hours. The estimates are rough: the new chat should check
them with the user.

| Phase | What is built | Done when | Days |
|---|---|---|---|
| 0 | The branch, uv, ruff, pytest, `CLAUDE.md`, the plan, the record files | A first test passes | 2 |
| 1 | Download a pinned Cricsheet snapshot; load men's T20 matches; validation checks; DVC | Loading twice adds nothing; bad files are listed | 5 |
| 2 | Features for every ball of a chase; the season split; logistic regression; Brier, log loss, reliability plot | A baseline with honest numbers, split by time | 5 |
| 3 | MLflow tracking; Random Forest and XGBoost; calibration; `params.yaml`; the DVC pipeline | `dvc repro` rebuilds the whole model | 5 |
| 4 | The gate and the MLflow registry (champion and challenger) | A worse model is refused, and the refusal recorded | 2 |
| 5 | FastAPI service; Docker; Docker Compose with MLflow | `docker compose up`, then a prediction from the container | 4 |
| 6 | GitHub Actions: lint, tests, a tiny pipeline on fixture data, the image build | A pull request runs it all | 3 |
| 7 | Monitoring: prediction logs, input drift (PSI or KS), Brier once results are known | A drift report for a known change | 4 |
| 8 | The retraining loop and rollback, triggered by new matches or drift | New data in, a new champion or a recorded refusal out | 3 |
| 9 | The season replay: the rules written first, then policies A, B, C (and D) | Results by season, with uncertainty | 5 |
| 10 | Review: every number in `numbers.md`, figures rebuilt by script | Nothing unsupported | 2 |
| 11 | The write-up (format set by the supervisor) | A draft | 5 |
| | **Total** | | **about 45** |

**Milestones:** a baseline with honest numbers (end of Phase 2); a model rebuilt by one command
(Phase 3); a prediction from a container (Phase 5); the replay's results (Phase 9).

---

## 9. What "good" looks like at the end

- Anyone can rebuild any past model from the repository, DVC and one command.
- A worse model can never replace a better one without the reason being recorded.
- The service answers with the model version that made each prediction.
- The season replay says, with numbers and their uncertainty, whether monitoring and retraining
  helped, and how fast drift was noticed.
- Every number in the write-up has a row in `docs/numbers.md`.

---

## 10. To settle with the supervisor (before Phase 1)

1. Is this a separate paper, a course project, or a chapter of the thesis? It decides the venue
   and how much of the record-keeping is needed.
2. Is live win probability the right choice? Alternatives considered: a first-innings score
   predictor (simpler, a weaker story); dismissal risk (overlaps with CricketMind's thesis); the
   CricShot video model (belongs to the CricShot project).
3. Must it run in a cloud (the paper uses AWS)? If so, whose account and what budget?
4. Men's T20 only, or women's too? Only the IPL, or all leagues?
5. Is the GitHub repository private or public? GitHub Actions minutes depend on it.

---

## 11. Sources found so far

- The paper: https://doi.org/10.1007/s11869-026-02068-4 and https://github.com/vaibhav7766/mlops-climate
- Cricsheet: https://cricsheet.org
- Win-probability projects on Cricsheet (self-reported numbers, to be checked):
  - https://github.com/Pinkesh2905/GameCastAI
  - https://github.com/Seshasai-Tunuguntla/chaseline
  - https://github.com/Devarsh-04/cricket-analytics
  - https://github.com/dani8055-eng/cricket-winprob
  - https://github.com/chandr-csgp/IPL-WInProbability_Project
  - https://github.com/Kushgirap/bbl-win-probability
  - https://github.com/PushkarSingh-tech/CricLeverage
- The Calibration-Leverage Tradeoff in Exactly Solvable Win-Probability Models (arXiv):
  https://arxiv.org/pdf/2608.14696
- The Impact Player rule and scoring:
  - https://www.espncricinfo.com/ci/content/story/1436641.html
  - https://africa.espn.com/cricket/story/_/id/36223170/ipl-2023-how-much-impact-impact-player-rule-had
  - https://www.cricketmachan.com/analysis/has-the-impact-player-rule-tilted-ipl-in-favor-of-batsmen/
- Drift and retraining:
  - https://arxiv.org/pdf/1907.08120
  - https://learn.aisingapore.org/2024/10/evaluating-model-retraining-strategies-by-reinhard-sellmair-oct-2024/
- General MLOps frameworks (background):
  - https://www.ijert.org/mlops-changes-and-tools-adapting-machine-learning-workflows-for-enhanced-efficiency-and-scalability
  - https://www.sciencepublishinggroup.com/article/10.11648/j.ajai.20250902.29

**Not yet done:** a proper literature search. The web search on 2026-11-18 found no paper that
joins cricket with a full MLOps pipeline and a measured drift study, but a web search is not a
literature review. Phase 0 should include one before the gap is claimed.
