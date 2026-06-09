# Phase 2 — remaining work (deferred)

This document collects the open work items that came out of Tasks
27–35. Each item has a brief description, why it matters, a rough
estimate of the time/effort, and a pointer to the closest existing
code or doc.

## 1. ResultScout live wiring for UCL 25/26 + WC 26 finals

**Why:** The agent's full pitch is "log a prediction, wait for the
result, self-evaluate." Right now `ResultScout` is implemented and
unit-tested, but it's never been pointed at a live result feed.
Until we run it on the two real finals (UCL 25/26 + WC 26), the
"self-evaluation" half of the agent is unproven on the live data
that motivated the project.

**What needs to happen:**
- Add an RSS / feed-poll poller to `src/soccer_agent/result_scout.py`
  (or a new `live_results.py`) that hits UEFA.com / FIFA.com / an
  external result API on a schedule.
- For the UCL 25/26 final, the input is straightforward: we know
  the date, the teams, and the venue (Puskás Aréna, Budapest).
- For the WC 26 final (MetLife Stadium, 19 July 2026), the input
  is not yet known — we won't know who's playing until the
  semi-finals. The scout needs to discover the match from the
  bracket and self-bind.
- Schedule predictions: at kickoff -24h, kickoff -2h, and kickoff
  -10min. Each one logs a separate prediction row. The latest one
  is what `ResultScout` self-evaluates against.
- Document the contract in `docs/result_scout.md`.

**Effort:** ~1 day once the data source is picked.

**Files:** `src/soccer_agent/result_scout.py`, `docs/result_scout.md`.

## 2. Per-competition LOO cross-validation (honest test)

**Why:** Task 35 wired per-comp calibrators into `predict()`, but
the empirical question "does per-comp beat global out-of-sample?"
was *not* answered. At n=20–29 per comp the in-sample Brier is
zero (overfit) and we have no way to tell whether per-comp is
actually generalizing. The honest test is leave-one-out within
each competition: fit on n-1 cases, predict the held-out one,
compute the Brier. Repeat. If the per-comp LOO Brier beats global
LOO Brier across competitions, keep the per-comp. Otherwise drop
it back to global-only.

**What needs to happen:**
- Add `fit_and_score_per_comp_loo(samples, *, key, min_n) -> dict`
  to `src/soccer_agent/eval/calibration.py`.
- Run it on the 106-case DB and write a small markdown report.
- Either ship per-comp (if the LOO test passes) or document the
  finding and switch back to global-only in `predict()`.

**Effort:** ~3 hours, mostly the LOO loop.

**Files:** `src/soccer_agent/eval/calibration.py`, a new
`docs/per_comp_loo.md`.

## 3. Re-evaluate per-comp vs global at n≥50 per comp

**Why:** Even if the LOO test (item 2) gives a result, it's
inherently underpowered at n=20–29 per comp. The right thing to
do is expand the eval set (ingest 25/26 + 26/27 seasons once those
are in the football-data.co.uk archive) and re-run.

**What needs to happen:**
- Wait for 25/26 + 26/27 seasons to populate football-data.co.uk.
- Re-run `scripts/ingest_football_data.py` with the new CSVs.
- Re-fit, re-test, re-document.

**Effort:** ~1 hour, gated on external data.

## 4. Investigate LaLiga's higher raw Brier

**Why:** Per-competition Brier breakdown (Task 35) showed LaLiga at
~0.052 raw vs ~0.020–0.030 for the other leagues. That's ~2× worse.
The cause could be:
- the numeric reasoner's Elo model is over-fit to EPL
- the LaLiga fixtures in the eval set are systematically harder
  (e.g. mid-table vs top-table)
- the data ingestion mapped some team names wrong and we're
  silently making bad predictions on the wrong matches
- the bookmaker odds we used for the form/odds signal were
  noisier for LaLiga

**What needs to happen:**
- Pick the 5 worst LaLiga predictions and trace each one back to
  the underlying fixtures and tool calls.
- If it's a data bug (likely), fix the team-name mapping.
- If it's a model bias, add a LaLiga-specific Elo blend or weight.

**Effort:** ~2 hours, mostly investigation.

## 5. Multi-class calibration (3-way H/D/A)

**Why:** Right now we reduce "did the pick match the actual?" to a
1D probability. That works for the Brier score, but it discards
information: a confident "draw" pick and a confident "home" pick
that both win are treated identically. The right thing is to fit a
3-way calibrator on the H/D/A probability vector. Calibrated draw
probabilities especially would be useful — the market is bad at
pricing draws, and a calibrated draw signal is a competitive edge.

**What needs to happen:**
- Add a `multiclass_isotonic` or Dirichlet-calibration module to
  `src/soccer_agent/calibration.py`.
- Update `predict()` to apply the 3-way calibrator instead of the
  1D one.
- Re-measure Brier and ECE.

**Effort:** ~1 day. Several off-the-shelf implementations exist
(`netcal`, `mapie`).

## 6. Online / rolling recalibration

**Why:** The calibrators are re-fitted in batch from a static
`predictions` DB. As the agent sees more matches, its confidence
distribution will drift (the LLM reasoner improves, the tool
fixture quality changes, the data sources change). A 90-day
rolling recalibration cron would keep the calibrators fresh
without manual intervention.

**What needs to happen:**
- Add a `--window-days 90` flag to
  `python -m soccer_agent.eval.calibration`.
- Wire it into a cron job (`hermes cron create`).
- Add a calibration-staleness tile to the dashboard.

**Effort:** ~3 hours, mostly the cron wiring.

**Files:** `src/soccer_agent/eval/calibration.py`, dashboard.

## 7. Public deployment

**Why:** The dashboard and CLI are localhost-only. To check on a
prediction from a phone during a live match, the user needs a
public URL. A small VPS (fly.io, Railway, Hetzner) running the
FastAPI app behind a `caddy` reverse proxy is the cheapest
deployable shape.

**What needs to happen:**
- Write a `Dockerfile` (one FastAPI process, mounted SQLite).
- Write a `fly.toml` (or equivalent).
- Decide on auth — at minimum, a bearer token in the URL. For
  multi-user, OIDC or basic auth.
- Update the README's "Deployment" section.

**Effort:** ~1 day, mostly infra decisions.

## Priority order

If we have a free week:

1. **Item 1** (ResultScout) — proves the agent end-to-end on real
   data, which is the whole point.
2. **Item 4** (LaLiga investigation) — likely a 2-hour data bug
   that, if fixed, improves accuracy for free.
3. **Item 2** (Per-comp LOO) — settles an open question.
4. **Item 7** (Deployment) — makes the agent usable in the wild.
5. Items 3, 5, 6 — follow-ups once we have live data flowing.
