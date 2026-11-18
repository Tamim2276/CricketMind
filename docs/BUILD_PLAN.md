# Build plan

Live T20 win probability, built as an MLOps system, plus a season replay that measures whether
drift monitoring and retraining help.

Made on 2026-11-18 from Section 8 of [the brief](PROJECT_BRIEF.md), after settling Section 10.
**One day = one session of about 4–5 hours.** The days are a guide, not a deadline: if a day runs
over, the next session starts where it stopped. The order is the plan, and no step is skipped.

---

## What we settled on Day 1

| Question (brief, Section 10) | Answer | What it changes |
|---|---|---|
| What is this project? | A separate paper | A literature search in Phase 0; Phase 11 is a paper draft |
| Must it run in a cloud? | Yes, AWS | A new phase, 6b (3 days) |
| Which matches? | Men's T20, all leagues; the IPL is the main case | Phase 1 loads all leagues; Phase 9 adds a comparison league |
| Is the repo private or public? | Public | No limit on GitHub Actions minutes. No secret ever goes in the repo |
| Is win probability the right choice? | Not asked yet | The plan assumes yes (see "Still open") |

## What changed from the brief's plan

1. **Phase 0 gets a third day** for the literature search. A paper can't claim "no one has done
   this" without one.
2. **New Phase 6b: AWS** (3 days), between GitHub Actions and monitoring.
3. **Phase 7 sets the drift thresholds from seasons before 2023 only**, and writes them down
   before looking at 2022 → 2023. Otherwise the replay would be tuned on the very answer it is
   meant to find.
4. **Phase 9 runs policy D too** (the brief marks it optional). It is the same code, run once
   more.
5. **New Phase 5b: a basic Streamlit page** (2 days), so the system has a face you can show.
   It only calls the FastAPI service; it never loads a model itself.
6. **New Phase 5c: trust and what-ifs** (4 days). With every chance, the service also gives a
   range and an "unusual state" warning; and it answers two what-if questions: what it takes to
   become favourite, and how much the next ball matters.
7. **Phase 7 gets a fifth day** for the warning rate over time, and **Phase 9 measures** whether
   warnings rose in 2023 before the Brier score got worse. If they did, the system can notice a
   change without waiting for results.
8. **Total: about 56 days**, not 45.

---

## The phases at a glance

| Phase | What gets built | Days | Done when |
|---|---|---|---|
| 0 | Setup, the record files, the literature search | 1–3 | A first test passes; the gap is checked |
| 1 | Cricsheet data, loaded, checked, and in DVC | 4–8 | Loading twice adds nothing; bad files are listed |
| 2 | Features per ball; the season split; a baseline; honest metrics | 9–13 | A baseline with honest numbers, split by time |
| 3 | MLflow; Random Forest and XGBoost; calibration; the DVC pipeline | 14–18 | `dvc repro` rebuilds the whole model |
| 4 | The gate and the model registry | 19–20 | A worse model is refused, and the refusal recorded |
| 5 | FastAPI service; Docker; Docker Compose | 21–24 | `docker compose up`, then a prediction from the container |
| 5b | A basic Streamlit page that calls the service | 25–26 | A past match's win-probability line, drawn live in the browser |
| 5c | Trust and what-ifs: a range, an "unusual state" warning, two what-if answers | 27–30 | All four show on the page for any match state |
| 6 | GitHub Actions | 31–33 | A pull request runs it all |
| 6b | AWS: the image registry (ECR) and a server (EC2) | 34–36 | A prediction from AWS; a rollback to the previous image |
| 7 | Monitoring: prediction logs, drift, the warning rate, Brier once results are known | 37–41 | A drift report for a known change |
| 8 | The retraining loop and rollback | 42–44 | New data in, a new champion or a recorded refusal out |
| 9 | The season replay (the experiment) | 45–49 | Results by season, with uncertainty |
| 10 | Review | 50–51 | Nothing unsupported |
| 11 | The paper | 52–56 | A draft for the supervisor |

**Milestones:** a baseline with honest numbers (Day 13) · a model rebuilt by one command
(Day 18) · a prediction from a container (Day 24) · the Streamlit page working (Day 26) · trust
and what-ifs on the page (Day 30) · a prediction from AWS (Day 36) · the replay's results
(Day 49) · the draft (Day 56).

## Jobs for you outside a session

Big downloads and sign-ups. Do them when the power is steady.

- **Before Day 22:** install Docker Desktop. It is a large download, it needs WSL 2, and the
  computer restarts.
- **Before Day 34:** create an AWS account, once your supervisor has said whose account it is and
  what the budget is.

---

## Day by day

Each day says what we build, what you learn, and how we know it is done. Every day ends with an
entry in `notes/journal.md` and a commit command from me.

### Phase 0 — Setup (Days 1–3)

**Day 1 (2026-11-18): the rules and the plan**
- Build: `CLAUDE.md`, this plan, and the four record files (`notes/journal.md`,
  `docs/decisions.md`, `docs/numbers.md`, `docs/limitations.md`), with today's answers as the
  first decisions.
- Learn: what the paper builds, where it is weak, and what our experiment adds.
- Done when: you have read this plan and said what to change.

**Day 2: the Python project**
- Build: a uv project on Python 3.12; ruff and pytest set up; the folder layout; a `.gitignore`
  (data and model files stay out of git); the first real function and its test. Then the first
  push of this branch to GitHub.
- Learn: what uv, a virtual environment, `pyproject.toml`, a linter and a test are.
- Done when: `uv run pytest` passes and `uv run ruff check .` is clean.

**Day 3: the literature search**
- Build: `docs/literature.md`. Searches in Google Scholar, Semantic Scholar, IEEE Xplore, the
  ACM Digital Library and arXiv, with the search words and dates written down, and a short note
  on each relevant paper. The brief's leads (the arXiv preprint, the public projects' numbers)
  are checked as leads only. A short list of two or three venues to discuss with your
  supervisor.
- Learn: how to record a literature search so someone else could repeat it.
- Done when: the gap is either confirmed or rewritten, with sources.

### Phase 1 — The data (Days 4–8)

**Day 4: Cricsheet and the snapshot**
- Build: we pick the men's T20 files on cricsheet.org together, and you download them. Then the
  date, the file names and each file's SHA-256 go in `docs/numbers.md`; the credit to Cricsheet
  (its licence asks for one); a note of the matches withheld at the source.
- Learn: what one match file looks like (match info, innings, overs, balls), and what a SHA-256
  fingerprint is.
- Done when: the snapshot's fingerprint is recorded.

**Day 5: reading matches**
- Build: `ingest/` code that reads one match file, then all of them, and keeps men's T20 only.
  No-result matches, Super Over ties and rain-changed (DLS) targets are set aside, and their
  counts go in `docs/limitations.md`.
- Learn: JSON; reading files in a loop; why a match is never dropped silently.
- Done when: tests pass on a few real match files.

**Day 6: storing them**
- Build: DuckDB tables for matches and balls, saved as Parquet. Loading the same files again
  adds nothing.
- Learn: what a database table and a Parquet file are.
- Done when: a test loads the same files twice, and the row count doesn't change.

**Day 7: checking them**
- Build: validation checks. First the shape of each file, then cricket rules: no more than
  20 overs or 10 wickets, runs that add up, a target equal to the first-innings score plus one.
  Bad files go into a report, each with its reason.
- Learn: data validation. Plain checks first; then we decide whether pandera is worth adding,
  and record why in `docs/decisions.md`.
- Done when: the report lists every file that failed, and why.

**Day 8: DVC**
- Build: DVC set up, with a local remote folder outside the repo. The raw snapshot and the
  loaded data are tracked and pushed to it.
- Learn: why data doesn't go in git, and how DVC keeps a small fingerprint in git instead.
- Done when: Phase 1's two checks pass: loading twice adds nothing, and bad files are listed.

### Phase 2 — Features and a baseline (Days 9–13)

**Day 9: one row per ball**
- Build: `features/`. For every ball of a chase: the target, runs needed, balls left, wickets in
  hand, the current and required run rates, and the label (did the chasing side win?). Saved as
  Parquet.
- Learn: what a feature and a label are. Leakage: using something not yet known at that ball.
  A test guards against it.
- Done when: the leakage test passes, and no "season" or "era" feature exists.

**Day 10: the season split**
- Build: the split rule goes in `docs/decisions.md` first: which seasons train, which one
  calibrates, which ones test, all in time order. Then the code.
- Learn: why balls of one match are near copies of each other, so a random split leaks the
  answer.
- Done when: a test checks that no test season comes before a training season.

**Day 11: logistic regression**
- Build: the first model, a logistic regression baseline (scikit-learn).
- Learn: what it does: a weighted sum of the features, turned into a chance between 0 and 1.
- Done when: it predicts on the test seasons.

**Day 12: judging a probability**
- Build: Brier score, log loss, accuracy at 50%, a reliability plot, and calibration error by
  season. What counts as "better" is written down before any number is seen.
- Learn: why a model that says 90% must be right about 90% of the time.
- Done when: all the metrics and the plot come from one script.

**Day 13: honest numbers**
- Build: uncertainty by resampling whole matches (a bootstrap). The random split is run once,
  to show how much it flatters the model. Every number goes in `docs/numbers.md`.
- Learn: what a confidence interval says, and why we resample matches, not balls.
- Done when (**milestone**): a baseline with honest numbers, split by time.

### Phase 3 — Tracking and a rebuildable model (Days 14–18)

**Day 14: MLflow**
- Build: MLflow tracking on your laptop. The baseline is re-run and logged: its settings,
  metrics, plots and the model.
- Learn: what an experiment run is, and why we log numbers instead of copying them by hand.
- Done when: the baseline run shows up in the MLflow web page.

**Day 15: Random Forest and XGBoost**
- Build: both models, on the same split, judged the same way, logged in MLflow. CPU only.
- Learn: what a decision tree, a forest of trees and boosting are.
- Done when: all three models sit side by side in MLflow.

**Day 16: calibration**
- Build: Platt and isotonic calibration of the best model, fitted on the held-out later season.
- Learn: why a model can put matches in the right order and still be too sure of itself, and
  how calibration fixes that.
- Done when: the reliability plots, before and after, are logged.

**Day 17: params.yaml and dvc.yaml**
- Build: every setting in `params.yaml`. The stages (load, check, features, train, evaluate) in
  `dvc.yaml`.
- Learn: a pipeline as a chain of stages, each with its inputs and outputs.
- Done when: `dvc repro` runs the whole chain.

**Day 18: rebuilt by one command**
- Build: fixed random seeds. Run `dvc repro` from clean twice, and compare the results.
- Learn: what makes a result repeatable.
- Done when (**milestone**): `dvc repro` rebuilds the whole model, with the same numbers.

### Phase 4 — The gate (Days 19–20)

**Day 19: the rule and the registry**
- Build: the gate rule, written down first. A new model (the "challenger") replaces the one in
  use (the "champion") only if its Brier score is better and its calibration is no worse. Then
  the MLflow model registry, with "champion" and "challenger" as aliases.
- Learn: model versions. The paper uses MLflow's "Staging / Production" stages; newer MLflow
  replaced them with aliases.
- Done when: the current best model is the registered champion.

**Day 20: refusing a worse model**
- Build: the gate in code. Each refusal is recorded with its reason.
- Learn: how an automatic test can stop a bad release.
- Done when: a deliberately worse model is refused, and the refusal recorded.

### Phase 5 — The service (Days 21–24)

**Day 21: FastAPI**
- Build: `serve/` with `POST /predict` (the match state in; the chance and the model version
  out), `GET /health` and `GET /model`. Input checks, for example wickets between 0 and 10.
  Tests.
- Learn: what a web API, JSON input and a status code are.
- Done when: the tests call all three endpoints.

**Day 22: Docker**
- Build: a multi-stage Dockerfile. Build the image and run it.
- Learn: what an image and a container are, and why two stages make the image small.
- Done when: a prediction comes from the container.

**Day 23: the model inside the service**
- Build: decide how the service gets its model (built into the image, as the paper does, or
  loaded from MLflow at start) and record why. `/health` says OK only after the model is loaded.
  Every prediction is logged, for Phase 7.
- Learn: start-up, health checks and logs.
- Done when: the prediction log grows as you call the service.

**Day 24: Docker Compose**
- Build: `docker-compose.yml` with the service and an MLflow server.
- Learn: running several containers together.
- Done when (**milestone**): `docker compose up`, then a prediction from the container.

### Phase 5b — A basic Streamlit page (Days 25–26)

Streamlit turns a short Python script into a web page: no HTML, no JavaScript. The page is a
separate app. It sends each match state to the FastAPI service and shows what comes back, so
every chance it shows comes with the model version that made it.

**Day 25: the form**
- Build: `app/` with one Streamlit script. Sliders for target, runs needed, balls left and
  wickets in hand; a button; the answer shown as a percentage for each side, with the model
  version from `GET /model`. If the service isn't running, the page says so in plain words.
  The service's address is read from a setting, not typed into the code.
- Learn: how Streamlit reruns the script from top to bottom each time you touch something, and
  how one program calls another over HTTP.
- Done when: you move a slider and the chance changes.

**Day 26: a past match, ball by ball**
- Build: a small sample of real matches saved as one file (credited to Cricsheet; the same
  matches become the CI fixtures on Day 32). You pick a match; the page asks the service for the
  chance at every ball of the chase and draws it as a line, with wickets marked. A small
  Dockerfile for the page, and a third service in `docker-compose.yml`.
- Learn: why the line should fall when a wicket goes, and what a flat or jumpy line would tell
  us about the model.
- Done when (**milestone**): `docker compose up` brings up the service, MLflow and the page, and
  the line for a past match draws in your browser.

### Phase 5c — Trust and what-ifs (Days 27–30)

Four extras with every chance. The service works them out in tested Python; the Streamlit page
only shows them. For example:

> Chasing side: **62%** (55–69%) · ⚠ Few past matches looked like this, so trust it less.
> To become favourite: **12 off the next over**. Next ball: six 71% · dot 58% · wicket 48%.

**Day 27: the range**
- Build: several copies of the model, each trained on a different resample of the training
  matches, saved with the model in MLflow so they travel with every version. `/predict` returns
  the range of their answers next to the chance.
- Learn: why models trained on slightly different data disagree, and why they disagree most
  where data is thin.
- Done when: the range shows on the page. Whether it really is wider for rare match states is
  checked and recorded, whatever the answer.

**Day 28: the "unusual state" warning**
- Build: a summary of the training balls (how many fell in each box of runs needed, balls left
  and wickets in hand), saved with the model. For each prediction, the service counts how many
  training balls looked like this one; too few, and it adds a warning. The cut-off comes from
  seasons before 2023 only, and goes in `docs/decisions.md` before it is used.
- Learn: what "out of distribution" means: asking a model about something it has rarely seen.
- Done when: a test shows a rare made-up state gets a warning and a common one doesn't.

**Day 29: "what it takes"**
- Build: a `POST /what-if` endpoint, part one: the fewest runs off the next over that make the
  chasing side favourite (50% or more), found by asking the model "what if" for each possible
  total. If the side is already favourite, how few runs keep it there. Shown on the page.
- Learn: asking a model hypothetical questions; why a tree model can wobble (more runs don't
  always give a higher chance), and what we do about it.
- Done when: tests pass on a few match states checked by hand.

**Day 30: "how much the next ball matters"**
- Build: `POST /what-if`, part two: the chance after each possible next ball (dot, one, two,
  four, six, wicket), and the gap between the best and the worst. Shown on the page as small
  bars. The idea exists already (baseball's "leverage", and the arXiv preprint in the brief),
  so the paper cites it rather than claiming it.
- Learn: why a ball in the last over usually matters more than one in the fifth.
- Done when (**milestone**): the range, the warning and both what-ifs show on the page for any
  match state.

### Phase 6 — GitHub Actions (Days 31–33)

**Day 31: lint and tests on GitHub**
- Build: a workflow that runs ruff and pytest on every push to `cricket-mlops` and every pull
  request into it.
- Learn: what CI (continuous integration) is, and how to read a failed run.
- Done when: a push shows a green tick on GitHub.

**Day 32: a tiny pipeline in CI**
- Build: a handful of real match files kept as test fixtures (credited to Cricsheet). CI runs
  load → check → features → train → gate on them in a minute or two.
- Learn: why CI uses small fixture data, not the full snapshot.
- Done when: the tiny pipeline passes in CI.

**Day 33: the image build, and a pull request**
- Build: CI builds the Docker image and runs a security scan (for example `pip-audit` on the
  packages). Then a real pull request: you push a short-lived branch without switching
  (`git push origin HEAD:ci-check`) and open a PR from it into `cricket-mlops` on GitHub.
- Learn: pull requests. Ours can't go into `main`, because `main` has a completely different
  history from this branch.
- Done when: a pull request runs it all.

### Phase 6b — AWS (Days 34–36)

Needed before Day 34: an AWS account, and your supervisor's answer on whose account it is and
what the budget is.

**Day 34: AWS, safely, and the image registry**
- Build: a budget alarm first. A separate login for daily work (never the root account). An ECR
  repository. GitHub Actions pushes the image there, tagged with the git commit. It signs in
  with a short-lived token (OIDC), so no AWS key is ever stored in the public repo.
- Learn: what IAM, a region, ECR and OIDC are.
- Done when: the image appears in ECR after a push.

**Day 35: a server on EC2**
- Build: a small EC2 instance that pulls the image and runs the service, with only the service's
  port open (and the Streamlit page's port, so the demo works there too). An S3 remote added for
  DVC, as the paper has.
- Learn: what a cloud server is, and what a security group does.
- Done when: a prediction comes from AWS, over the internet.

**Day 36: deploy, roll back, switch off**
- Build: a deploy script that runs any image tag, so rolling back means running the previous
  tag. Steps to stop and to tear down. The cost so far recorded in `docs/numbers.md`.
- Learn: rollback by image tag, and how cloud cost builds up.
- Done when (**milestone**): a prediction from AWS, and a rollback to the previous image that
  works. The server is stopped whenever it isn't in use.

### Phase 7 — Monitoring (Days 37–41)

**Day 37: the prediction log**
- Build: each prediction stored with its input, output, range, warning, model version and time.
  Results are joined in once the match ends.
- Learn: why cricket suits monitoring: the true answer is known within hours.
- Done when: a query shows predictions next to their results.

**Day 38: input drift, our own way**
- Build: PSI and the KS test, written by us, on features like the required run rate. The
  thresholds come from seasons before 2023 only, and are written down before any 2023 data is
  looked at.
- Learn: drift, and what PSI and KS measure, each with a small example.
- Done when: the tests give the expected answers on made-up data.

**Day 39: performance drift**
- Build: the Brier score and calibration error over a rolling window of finished matches.
- Learn: input drift versus performance drift, and why we want both.
- Done when: a chart shows Brier match by match.

**Day 40: the warning rate over time**
- Build: over a rolling window of matches, the share of predictions that carried a warning and
  the average width of the range, drawn next to input drift and the Brier score.
- Learn: a warning on single balls versus a drift test on a whole season, and which might see a
  change sooner.
- Done when: a chart shows the warning rate match by match.

**Day 41: a drift report for a known change**
- Build: a report for IPL 2022 → 2023 (the Impact Player rule), using the pre-set thresholds,
  with the warning rate beside it. The same comparison in Evidently, to check our numbers
  against a library.
- Learn: how to read a drift report.
- Done when: a drift report for a known change.

### Phase 8 — Retraining and rollback (Days 42–44)

**Day 42: new matches only**
- Build: ingestion that fetches from Cricsheet only the matches we don't have yet, checks them,
  and makes a new DVC data version.
- Learn: this is the cricket version of the paper's API polling.
- Done when: running it twice brings nothing new the second time.

**Day 43: the loop**
- Build: new data or drift → retrain → gate → a new champion or a recorded refusal. Where the
  timer lives is settled first (see "Still open").
- Learn: how the pieces from Phases 1–7 join into one loop.
- Done when: new data in, a new champion or a recorded refusal out.

**Day 44: rollback**
- Build: if the champion does worse on finished matches than the one before it, the alias moves
  back and the previous image is redeployed, with the reason recorded.
- Learn: rollback as a normal, tested step, not an emergency.
- Done when: a test forces a rollback, and the rollback is recorded.

### Phase 9 — The season replay (Days 45–49)

**Day 45: the rules, written first**
- Build: `docs/replay_rules.md`: the training start, policies A–D, the drift tests and their
  thresholds, the promotion rule, how "reacting" to 2023 is counted, how we judge whether the
  warning rate rose before the Brier score got worse, the comparison league and why it was
  chosen, and how a difference is called real. Committed before any replay code runs, so git
  shows the order.
- Learn: writing the test before the run, and why it makes the result believable.
- Done when: the rules are committed.

**Day 46: the replay engine**
- Build: `replay/`. Matches are fed in date order; each policy predicts, sees the results, and
  decides whether to retrain. It saves after every season, so a power cut loses little.
- Learn: how to act out years of a live system on a laptop.
- Done when: a short replay over two seasons gives the expected output.

**Day 47: policies A and B on the IPL**
- Build: "never retrain" and "retrain on a timetable", over the full IPL. You run it, and it
  resumes after a cut.
- Done when: both runs finish and are logged.

**Day 48: policies C and D, then the comparison league**
- Build: "retrain on drift" and "recency-weighted" on the IPL. Then all four policies on the
  comparison league, with the same rules.
- Done when: all the runs finish and are logged.

**Day 49: results**
- Build: for each policy, season by season: Brier score, calibration error, matches taken to
  react after 2023, retrains and rollbacks, each with a bootstrap interval. Then how many
  matches into 2023 the warning rate rose, against how many the Brier score took. Figures made
  by script.
- Learn: how to say a difference is real, or that it isn't.
- Done when (**milestone**): results by season, with uncertainty. Whatever they show is
  reported.

### Phase 10 — Review (Days 50–51)

**Day 50: every number traced**
- Build: each number in `docs/numbers.md` checked against the run or script that made it. All
  figures rebuilt by one script. A rebuild from clean.
- Done when: nothing is unsupported.

**Day 51: limitations and the repo**
- Build: `docs/limitations.md` reviewed. The brief's web leads either confirmed on our data or
  dropped. A README on how to rebuild any model. Credits (Cricsheet, the GreenAirOps paper). A
  licence chosen for the code.
- Done when: a stranger could rebuild a model from the README.

### Phase 11 — The paper (Days 52–56)

**Day 52: the venue and the outline**
- Build: the venue's rules (length, format, deadline, and whether reviewers must not know who
  the authors are). An outline, and the list of figures and tables.
- Done when: your supervisor agrees with the outline.

**Day 53: the system**
- Write: the pipeline and how it is built, in our own words and figures. The GreenAirOps paper
  is cited, never copied.

**Day 54: the experiment and the results**
- Write: the replay's design and results. Every number comes from `docs/numbers.md`.

**Day 55: introduction, related work, limitations**
- Write: from `docs/literature.md` (updated) and `docs/limitations.md`.

**Day 56: the full draft**
- Build: every number checked once more against `docs/numbers.md`. A reproducibility statement.
  The draft goes to your supervisor.
- Done when: a draft.

---

## Still open

| Question | Needed by | Note |
|---|---|---|
| Has your supervisor agreed that live win probability is the right choice? (Section 10, Q2) | Day 4 | The plan assumes yes. |
| Which venue? Its length, format and deadline | A short list by Day 3; the choice by Day 52 | A deadline may change the pace. |
| Does the venue hide the authors from reviewers? | Before the paper links the repo | A public repo with your name on it would break that. |
| Whose AWS account, and what budget? | Day 34 | The budget alarm comes first. |
| Where does the retraining timer live? | Day 43 | GitHub runs timed workflows only from the repo's default branch, `main`, and ours isn't `main`. Options: (a) give this project its own GitHub repo; (b) retrain whenever new data is pushed; (c) a timer on your laptop or on AWS. **I suggest (a):** timed workflows then just work, and the paper's code link shows only this project. |
