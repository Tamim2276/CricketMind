# The 10-day plan

The whole project in 10 days. Nothing is dropped: every step of [the full plan](BUILD_PLAN.md)
is here, grouped into 10 days. That plan's "Day 1" to "Day 56" are called **steps 1–56** here;
open it when you want the detail of a step.

Written 2026-11-20.

## Read this first: how 56 steps fit into 10 days

The full plan was 56 sessions of 4–5 hours, about 250 hours. Ten long days of 10–12 hours are
about 120 hours. It only fits because of four changes in how we work:

1. **I write a whole file at a time**, then explain the lines that matter most. You ask about
   any other line. Before, it was one small piece at a time, every line explained. (This
   changes a rule in `CLAUDE.md`, so I'll ask you before I change it there.)
2. **Long runs go overnight**, and they resume after a power cut.
3. **The paper is written as we go**, not all at the end.
4. **You commit after every block**, not only at night, so a power cut costs at most one block.

Even so, the plan is tight. Day 10's draft will be a first draft. If a day spills over, the
next one starts where it stopped. No step is skipped.

Each day has three blocks of about 3–4 hours, with breaks. Each day ends with an entry in
`notes/journal.md` and a commit command from me.

---

## Before the days: jobs for you, and answers from your supervisor

| By | What | Why |
|---|---|---|
| Day 2 | Your supervisor agrees that live win probability is the right choice | Everything after Day 1 is built on it |
| Day 4 | Docker Desktop installed (large download, needs WSL 2, restarts the computer) | Step 22 builds the first container |
| Day 6 | Your supervisor names the venue (length, format, deadline, are authors hidden?) | Step 52 writes the outline for it |
| Day 7 | An AWS account, with your supervisor's answer on whose it is and the budget | Steps 34–36 deploy to AWS |
| Day 8 | Decide where the retraining timer lives (see "Still open" in the full plan) | Step 43 builds the loop |

---

## The folder structure

Made on Day 1, filled in day by day. The brief allowed changes to its layout as long as each
change is explained. Ours adds `app/`, `retrain/`, `deploy/`, `paper/` and `scripts/`, with the
reasons in the table below the tree.

```
cricket-mlops-work/
  CLAUDE.md                  rules for Claude
  README.md                  how to rebuild any model; credits (Day 10)
  LICENSE                    the code's licence (Day 10)
  pyproject.toml             the project, its packages, ruff and pytest settings
  uv.lock                    exact package versions (made by uv)
  .python-version            3.12
  .gitignore                 what git never stores (data, models, caches)
  params.yaml                every setting: seasons, model settings, seeds, thresholds (Day 4)
  dvc.yaml  dvc.lock         the pipeline stages, and their fingerprints (Day 4)
  Dockerfile  .dockerignore  the API service's image (Day 4)
  docker-compose.yml         the service + MLflow + the Streamlit page (Day 5)
  .github/workflows/
    ci.yml                   lint, tests, tiny pipeline, image build, security scan (Day 6)
    deploy.yml               push the image to AWS ECR (Day 7)
  docs/
    PROJECT_BRIEF.md  BUILD_PLAN.md  BUILD_PLAN_10_DAYS.md
    decisions.md  numbers.md  limitations.md      the record (Day 1)
    literature.md                                  the literature search (Day 1)
    replay_rules.md                                the experiment's rules, written first (Day 9)
    papers/greenairops.pdf
  notes/
    journal.md               one entry per day
  data/                      tracked by DVC, never by git
    raw/                     the Cricsheet downloads, untouched (Day 1)
    interim/cricket.duckdb   matches and balls (Day 2)
    reports/bad_files.csv    every file that failed a check, and why (Day 2)
    features/chase_balls.parquet   one row per ball of a chase (Day 2)
    predictions/             the service's prediction log (Day 5)
    replay/                  the replay's saved progress and results (Day 9)
  src/cricket_mlops/
    config.py                reads params.yaml (Day 4)
    ingest/                  read_match.py, load.py, validate.py (Day 2); download.py (Day 8)
    features/                chase.py (Day 1–2), split.py (Day 2)
    train/                   models.py, metrics.py, calibrate.py, train.py (Day 3);
                             gate.py, registry.py (Day 4); ensemble.py (Day 5)
    serve/                   app.py, schemas.py (Day 4); prediction_log.py, trust.py (Day 5);
                             what_if.py (Day 6)
    monitor/                 drift.py (Day 7); performance.py, warnings.py, report.py (Day 8)
    retrain/                 loop.py, rollback.py (Day 8)
    replay/                  engine.py, policies.py (Day 9); results.py (Day 10)
  app/
    streamlit_app.py         the page (Day 5)
    sample_matches.json      a few real matches for the page (Day 5)
    Dockerfile               the page's image (Day 5)
  deploy/aws/
    deploy.sh                run any image tag on EC2, which also does rollbacks (Day 7)
    teardown.md              how to stop and delete everything, to stop costs (Day 7)
  paper/
    outline.md  draft.md  figures/     written as we go (Days 6, 9, 10)
  scripts/
    make_figures.py          every figure in the paper, rebuilt by one command (Day 10)
  tests/
    fixtures/                a few real match files, the same as app/sample_matches (Day 6)
    test_*.py                one test file per code file
```

| Change from the brief | Why |
|---|---|
| `app/` beside `src/` | The Streamlit page is a separate program with its own container; it only calls the API |
| `retrain/` | The loop joins ingest, train and serve, so it belongs to none of them |
| `deploy/aws/` | The AWS scripts are not Python, and not part of the package |
| `paper/` | The paper is written as we go, next to the numbers it uses |
| `scripts/` | Figures are rebuilt by one command, not made by hand |

These go in `docs/decisions.md` on Day 1.

---

## Day 1 — Setup, the record, the literature, the data (steps 1–4)

**Block 1: setup (steps 1–2)**
1. The record files: `notes/journal.md`, `docs/decisions.md`, `docs/numbers.md`,
   `docs/limitations.md`. The first decisions: the Section 10 answers, the 10-day pace, the
   folder layout.
2. Python 3.12 and the project. You run:
   ```
   uv python install 3.12
   uv init --package --name cricket-mlops --python 3.12
   uv add --dev pytest ruff
   ```
3. I set up ruff and pytest in `pyproject.toml`, and `.gitignore` (`.venv/`, `data/`,
   `mlruns/`, `mlflow.db`, the cache folders).
4. The empty folders from the tree above, each Python folder with an `__init__.py`.
5. The first real code: `features/chase.py` with `runs_needed()` and `balls_left()`, and
   `tests/test_chase.py`.
6. Checks, then the first push. You run:
   ```
   uv run pytest
   uv run ruff check .
   uv run ruff format --check .
   git push -u origin cricket-mlops
   ```

**Block 2: the literature search (step 3)**
- `docs/literature.md`: Google Scholar, Semantic Scholar, IEEE Xplore, ACM, arXiv. The search
  words and dates written down, a short note on each relevant paper, the gap checked, and two or
  three venues for your supervisor. These notes later become the paper's related work.

**Block 3: the data (step 4)**
- We pick the men's T20 files on cricsheet.org together. You download them into `data/raw/`,
  then fingerprint each file:
  ```
  Get-FileHash data\raw\* -Algorithm SHA256
  ```
- The date, file names and fingerprints go in `docs/numbers.md`. The Cricsheet credit goes in
  the README, and the matches withheld at the source go in `docs/limitations.md`.

**Done when:** the first test passes, lint is clean, the branch is on GitHub,
`docs/literature.md` exists, and the snapshot's fingerprints are recorded.

---

## Day 2 — Load, check, version, features (steps 5–10)

You install: `uv add pandas duckdb pyarrow dvc`

**Block 1: reading and storing (steps 5–6)**
- `ingest/read_match.py`: one match file → its balls. Keeps men's T20 only, and sets aside
  no-result matches, Super Over ties and DLS targets, with counts in `docs/limitations.md`.
- `ingest/load.py`: all files → `data/interim/cricket.duckdb`. Loading twice adds nothing.
- Tests: `test_read_match.py`, `test_load.py` (load twice, same row count).

**Block 2: checks and DVC (steps 7–8)**
- `ingest/validate.py`: file shape, then cricket rules (≤ 20 overs, ≤ 10 wickets, runs add
  up, target = first innings + 1). Bad files go in `data/reports/bad_files.csv` with a reason.
  Then we decide on pandera and record why.
- DVC, with the store in a folder outside the repo. You run:
  ```
  uv run dvc init
  uv run dvc remote add -d localstore D:/Cricket/dvc-store-mlops
  uv run dvc add data/raw
  uv run dvc push
  ```

**Block 3: features and the split (steps 9–10)**
- `features/chase.py`: one row per ball of a chase → `data/features/chase_balls.parquet`, plus
  a leakage test. There is no season or era feature.
- The split rule goes in `docs/decisions.md` first. Then `features/split.py`, with a test that
  no test season comes before a training season.

**Done when:** loading twice adds nothing, bad files are listed, and the leakage and split tests
pass.

---

## Day 3 — The baseline, MLflow, the models (steps 11–16)

You install: `uv add scikit-learn xgboost matplotlib mlflow`

**Block 1: the baseline and its numbers (steps 11–13)**
- First, "what counts as better" goes in `docs/decisions.md`, before any number is seen.
- `train/models.py`: logistic regression. `train/metrics.py`: Brier, log loss, accuracy at
  50%, the reliability plot, calibration error by season, a bootstrap over whole matches.
- The random split is run once, to show how much it flatters. Every number gets a row in
  `docs/numbers.md`.

**Block 2: MLflow, Random Forest, XGBoost (steps 14–15)**
- `train/train.py` logs each run to MLflow. To see the runs:
  ```
  uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
  ```
  then open http://127.0.0.1:5000.
- Random Forest and XGBoost, on the same split and judged the same way. CPU only.

**Block 3: calibration (step 16)**
- `train/calibrate.py`: Platt and isotonic, fitted on the held-out later season. Reliability
  plots before and after, logged.

**Done when (milestone):** a baseline with honest numbers, split by time, and all three models
side by side in MLflow.

---

## Day 4 — One command to rebuild, the gate, the API, Docker (steps 17–22)

Docker Desktop must be installed before today. You install: `uv add fastapi uvicorn pyyaml`
and `uv add --dev httpx`

**Block 1: the pipeline (steps 17–18)**
- `params.yaml` (every setting), `config.py` (reads it), `dvc.yaml` (load → check → features →
  train → evaluate). You run, twice, and we compare:
  ```
  uv run dvc repro --force
  ```

**Block 2: the gate (steps 19–20)**
- The gate rule goes in `docs/decisions.md` first. `train/registry.py`: "champion" and
  "challenger" aliases in MLflow. `train/gate.py`: a worse model is refused, with the reason
  recorded. A test proves it.

**Block 3: the API and the first container (steps 21–22)**
- `serve/app.py` and `serve/schemas.py`: `POST /predict`, `GET /health`, `GET /model`, with
  input checks and tests. To try it:
  ```
  uv run uvicorn cricket_mlops.serve.app:app --reload
  ```
  then open http://127.0.0.1:8000/docs.
- `Dockerfile` (multi-stage) and `.dockerignore`. You run:
  ```
  docker build -t cricket-mlops-api .
  docker run -p 8000:8000 cricket-mlops-api
  ```

**Done when (milestone):** `dvc repro` rebuilds the model with the same numbers, a worse model
is refused, and a prediction comes from the container.

---

## Day 5 — Compose, the Streamlit page, the range and the warning (steps 23–28)

You install: `uv add streamlit requests`

**Block 1: the model inside the service, and Compose (steps 23–24)**
- How the service gets its model is decided and recorded. `/health` says OK only after the
  model is loaded. `serve/prediction_log.py` logs every prediction to `data/predictions/`.
- `docker-compose.yml`: the service and an MLflow server. You run: `docker compose up --build`

**Block 2: the Streamlit page (steps 25–26)**
- `app/streamlit_app.py`: sliders → the chance and the model version. Then a past match from
  `app/sample_matches.json`, drawn ball by ball, with wickets marked.
- `app/Dockerfile`, and the page as a third service in Compose. To try it on its own:
  ```
  uv run streamlit run app/streamlit_app.py
  ```
  then open http://localhost:8501.

**Block 3: trust (steps 27–28)**
- `train/ensemble.py`: several model copies on resampled matches, saved with the model.
  `/predict` returns a range.
- The warning's cut-off goes in `docs/decisions.md` first, from seasons before 2023 only. Then
  `serve/trust.py`: a summary of the training balls, and the "unusual state" warning, with a
  test.

**Done when (milestone):** `docker compose up` brings up all three services, the past-match line
draws, and the range and the warning show on the page.

---

## Day 6 — What-ifs, GitHub Actions, the outline (steps 29–33, 52)

**Block 1: what-ifs (steps 29–30)**
- `serve/what_if.py` and `POST /what-if`: "what it takes" (the fewest runs off the next over to
  become favourite) and "how much the next ball matters" (the chance after a dot, one, two,
  four, six or wicket). Both are tested on states checked by hand, and both are shown on the
  page.

**Block 2: CI (steps 31–33)**
- `tests/fixtures/`: the same few matches as the page uses, credited to Cricsheet.
- `.github/workflows/ci.yml`: ruff, pytest, the tiny pipeline on the fixtures, the Docker
  build, and `pip-audit`. Then the real pull request test. You run:
  ```
  git push origin HEAD:ci-check
  ```
  then on GitHub you open a pull request from `ci-check` into `cricket-mlops`, wait for green,
  close it, and delete the branch:
  ```
  git push origin --delete ci-check
  ```

**Block 3: the paper's outline (step 52, moved earlier)**
- `paper/outline.md`, following the venue's rules from your supervisor, with the list of
  figures and tables.

**Done when:** both what-ifs show on the page, a pull request runs all the checks, and the
outline is written.

---

## Day 7 — AWS, the prediction log, input drift (steps 34–38)

The AWS account and the budget must be settled before today.

**Block 1: AWS, safely (step 34)**
- A budget alarm first. A separate login for daily work, never root. An ECR repository. A role
  GitHub can use through OIDC, so no AWS key is stored anywhere.
  `.github/workflows/deploy.yml` pushes the image, tagged with the git commit.

**Block 2: EC2, deploy, rollback (steps 35–36)**
- A small EC2 instance, with only the API's and the page's ports open. `deploy/aws/deploy.sh`
  runs any tag, so rolling back means running the previous tag. `deploy/aws/teardown.md`
  covers stopping and deleting.
- The S3 store for DVC. You run:
  ```
  uv add "dvc[s3]"
  uv run dvc remote add s3store s3://<bucket-name>/dvc
  ```
- The cost so far goes in `docs/numbers.md`. The server is stopped before you sleep.

**Block 3: monitoring starts (steps 37–38)**
- The prediction log, joined to results once matches end.
- The drift thresholds go in `docs/decisions.md` first, from seasons before 2023 only. Then
  `monitor/drift.py`: PSI and KS, tested on made-up data.

**Done when (milestone):** a prediction comes from AWS, a rollback works, the server is stopped,
and the drift tests pass.

---

## Day 8 — Monitoring, retraining, rollback (steps 39–44)

You install: `uv add evidently`

**Block 1: the rest of monitoring (steps 39–41)**
- `monitor/performance.py`: Brier and calibration over a rolling window.
  `monitor/warnings.py`: the warning rate over time. `monitor/report.py`: the IPL 2022 → 2023
  report, with the pre-set thresholds and the warning rate, checked against Evidently.

**Block 2: new matches and the loop (steps 42–43)**
- `ingest/download.py`: only matches we don't have yet, checked, then a new DVC version.
  Running it twice brings nothing new the second time.
- `retrain/loop.py`: new data or drift → retrain → gate → a new champion or a recorded refusal.
  The timer is set where we decided it lives.

**Block 3: rollback (step 44)**
- `retrain/rollback.py`: a worse champion moves back, and the previous image is redeployed,
  with the reason recorded. A test forces a rollback.

**Done when:** the drift report for 2022 → 2023 exists, new data gives a champion or a recorded
refusal, and a forced rollback is recorded.

---

## Day 9 — The season replay (steps 45–48, 53)

**Block 1: the rules, committed first (step 45)**
- `docs/replay_rules.md`: the training start, policies A–D, the drift tests and thresholds,
  the promotion rule, how reacting is counted, how the warning rate is judged, the comparison
  league, and how a difference is called real. **You commit it before any replay code is
  written**, so git shows the order.

**Block 2: the engine, and starting the runs (steps 46–48)**
- `replay/engine.py` and `replay/policies.py`: matches in date order, saved after every
  season to `data/replay/`. A short test replay over two seasons.
- Then you start the runs: all four policies on the IPL, then on the comparison league. Each
  resumes after a cut. I give you the exact commands on the day.

**Block 3: the paper's system section (step 53), while the runs go**
- `paper/draft.md`: the pipeline in our own words. GreenAirOps is cited, never copied.

**Overnight:** the runs finish.

**Done when:** the rules are committed before the code, and all runs have finished by the
morning.

---

## Day 10 — Results, review, the draft (steps 49–51, 54–56)

**Block 1: results (step 49)**
- `replay/results.py`: by policy and season, the Brier score, calibration error, matches taken
  to react, retrains and rollbacks, and whether warnings came before the Brier score got worse.
  Each comes with a bootstrap interval. Whatever the results show is reported.
- `scripts/make_figures.py` rebuilds every figure. You run:
  ```
  uv run python scripts/make_figures.py
  ```

**Block 2: review (steps 50–51)**
- Every row in `docs/numbers.md` traced to its run or script, and a rebuild from clean.
- `docs/limitations.md` reviewed. The brief's web numbers are either confirmed on our data or
  dropped.
- `README.md` (how to rebuild any model; credits to Cricsheet and GreenAirOps), and a
  `LICENSE`.

**Block 3: the draft (steps 54–56)**
- The experiment and results, the introduction, related work (from `docs/literature.md`), and
  limitations. Every number is checked against `docs/numbers.md`, and a reproducibility
  statement is added. Then the draft goes to your supervisor.

**Done when (milestone):** a first draft, with every number traceable.
