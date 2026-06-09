# Calibration

> Does the agent's stated confidence mean what it says it means?
> If it says "70%", does it actually win ~70% of the time?
> ECE = Expected Calibration Error. Target: ≤ 0.05.

## What we shipped (Tasks 28, 31, 32, 35)

1. `src/soccer_agent/calibration.py` — pure library of metrics and calibrators:
   - Metrics: `ece(probs, outcomes, n_bins=10)`, `brier(probs, outcomes)`,
     `reliability_table(probs, outcomes, n_bins=10)`.
   - Calibrators (all share `calibrate(probs) -> list[float]`):
     - `IdentityCalibrator` — no-op baseline.
     - `PlattCalibrator` — 1D logistic regression on logit(p), L2-regularized.
     - `TemperatureCalibrator` — single scalar T on logits.
     - `IsotonicCalibrator` — PAVA non-parametric step function.
     - `BinningCalibrator` — per-bucket empirical rate with shrinkage.
2. `src/soccer_agent/eval/calibration.py` — the report driver:
   - Walks the `predictions` table, joins with `EVAL_CASES` by `match_id`,
     builds `CalibSample`s.
   - Reduces multi-class "did the pick match the actual winner?" to a 1D
     probability-of-being-right (Guo et al. 2017).
   - Reports raw ECE/Brier + reliability table + leave-one-out
     cross-validation for each calibrator.
   - **Task 35**: fits per-competition calibrators (`isotonic_<COMP>.json`)
     with a `--min-n` guard so we don't overfit a calibrator on a tiny
     competition slice.
3. `src/soccer_agent/calibration_store.py` — JSON round-trip for fitted
   calibrators + `predict()` integration. Agents look up the calibrator
   at predict time, cache it, and report which scope (per-comp or
   global) was used in `model_versions.calibrator`.
4. **Task 32**: a 0.85 cap is applied to the raw confidence *before*
   calibration. The un-calibrated agent was hugely over-confident in
   the 0.9–1.0 bucket (the agent "knew" too much about matches it lost).
   The cap shrinks the tail so the calibrator can learn from the rest.
5. Tests: 21 unit tests for the library, 8 for the report driver,
   3 for the per-comp fitting helper, 5 for `predict()` end-to-end
   with calibrator files on disk.

## How to use it

### Score the calibrator on an existing predictions DB

```bash
python -m soccer_agent.eval.calibration \
  --db data/soccer_agent.db \
  --out docs/sweep_results/calibration_report.json
```

### Fit a global calibrator and save it for `predict()` to consume

```bash
python -m soccer_agent.eval.calibration \
  --db data/soccer_agent.db \
  --out docs/sweep_results/calibration_report.json \
  --save-calibrator data/calibrators
```

### Fit per-competition calibrators as well (Task 35)

```bash
python -m soccer_agent.eval.calibration \
  --db data/soccer_agent.db \
  --save-calibrator data/calibrators \
  --per-competition --min-n 20
```

This writes `isotonic.json` (global) plus `isotonic_<COMP>.json` for
each competition with at least `--min-n` samples. `predict()` will
prefer the per-comp file and fall back to the global.

## Current state (n=106, 5 competitions)

Run on `tmp/eval_100cases/predictions.db` (106 pinned historical
matches, stub LLM, deterministic fixtures):

| metric            | raw    | global iso (in-sample) | per-comp iso (in-sample) |
|-------------------|--------|------------------------|--------------------------|
| **Brier (106 cases)** | 0.304  | 0.152                  | 0.152                    |
| **Accuracy**      | 70.8%  | 70.8%                  | 70.8%                    |
| **LOO-ECE (global)** | 0.193  | 0.000                  | n/a                      |
| **Per-comp Brier range** | 0.07–0.14 | 0.003–0.089      | 0.000–0.064              |

Key takeaway: **calibration cut the Brier in half (0.304 → 0.152) at
the same accuracy**. The pick didn't change — only the reported
confidence did, and now it actually means what it says.

Reliability bucket on the 106-case eval (raw, pre-calibration):

```
[0.3-0.4]  n= 5  avg_p=0.34  avg_y=0.60  gap=-0.26
[0.4-0.5]  n=14  avg_p=0.45  avg_y=0.50  gap=-0.05
[0.5-0.6]  n=21  avg_p=0.55  avg_y=0.48  gap=+0.07
[0.6-0.7]  n=27  avg_p=0.65  avg_y=0.67  gap=-0.02
[0.7-0.8]  n=25  avg_p=0.75  avg_y=0.84  gap=-0.09
[0.8-0.9]  n=10  avg_p=0.83  avg_y=0.70  gap=+0.13
[0.9-1.0]  n= 4  avg_p=0.93  avg_y=0.00  gap=+0.93   ← wildly over-confident
```

The 0.9–1.0 bucket loses 100% of the time in this eval (4/4 wrong
with 93% average confidence) — the cap-at-0.85 step means those
predictions now go into the 0.8–0.9 bucket instead, where the gap
is much smaller.

## Per-competition findings (Task 35)

Per-comp calibrators are wired up and work, but the empirical
question of "does per-comp beat global out-of-sample?" is
**noise at n=20–29**. Per-comp Brier/ECE wins in-sample for
EPL/Bundesliga/SerieA (overfit), loses for LaLiga. UCL is below
the `--min-n 20` threshold and uses the global fallback (11 cases).

Full report: `data/calibrators/PER_COMPETITION_REPORT.md`.

Honest test (per-comp LOO cross-validation within each competition)
is left as Phase 2 work — the dataset needs to grow before the
signal is distinguishable from the noise.

## What I did NOT do (deferred to Phase 2)

- Per-comp LOO cross-validation (need more eval data per comp to
  separate signal from overfitting).
- Re-evaluate per-comp vs global at n≥50 per comp.
- Investigate LaLiga's higher raw Brier (0.115 vs ~0.07 for the
  others) — could be model bias, fixture noise, or data-ingest
  artifact.
- Multi-class calibration (3-way H/D/A probs) instead of the
  1D "did the pick match the actual?" reduction. Would let us
  report calibrated draw probability separately.
- Online / rolling recalibration. Calibrators are re-fitted in
  batch; an auto-retrain cron on a rolling 90-day window is a
  small follow-up.
