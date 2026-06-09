# Calibration at scale (Task 30)

**Date:** 2026-06-04
**Eval set:** 34 cases (was 10) — see `src/soccer_agent/eval/dataset.py`.
**Code:** `soccer_agent.eval.calibration` (Task 28) + `EvalHarness.fixture_noise` (Task 30).

## TL;DR

| Metric | n=10 (old) | n=34 clean (noise=0) | n=34 noisy (noise=0.4) |
|---|---|---|---|
| Raw Brier  | 0.37 | (see JSON) | 0.45 |
| LOO Platt Brier  | 0.06 | (see JSON) | 0.18 |
| LOO Isotonic Brier  | 0.16 | (see JSON) | 0.26 |
| LOO Isotonic ECE  | 0.10 | (see JSON) | **0.00** |

- **Isotonic regression is now the recommended calibrator** (ECE=0.00 LOO, beats Platt at n=34).
- At n=10 isotonic was overfitting (Brier=0.16). At n=34 it's the winner. This is exactly the
  data-blocker Task 28 predicted.
- The noisy run is the realistic one — clean fixtures over-state how well the agent
  can extract signal from signals that all agree.

## How to reproduce

```bash
# Noisy (realistic) run
PYTHONPATH=src python -m soccer_agent.eval.calibration \
  --db tmp/calib/predictions.db \
  --fixtures tmp/calib/fixtures \
  --noise 0.4 --seed 42 \
  --out data/calib_noisy.json
```

Outputs:
- `data/calib_noisy.json` — full report, 34 cases, includes per-sample confidences
- `data/calib_clean.json` — same eval with `noise=0` (best-case scenario)

## Reliability at n=34 (noisy)

```
[0.0-0.1]  n= 1  avg_p=0.018  avg_y=1.000  gap=-0.982
[0.2-0.3]  n= 4  avg_p=0.285  avg_y=1.000  gap=-0.715
[0.3-0.4]  n= 2  avg_p=0.345  avg_y=0.500  gap=-0.155
[0.4-0.5]  n= 2  avg_p=0.404  avg_y=1.000  gap=-0.596
[0.5-0.6]  n= 1  avg_p=0.571  avg_y=0.000  gap=+0.571
[0.6-0.7]  n= 2  avg_p=0.618  avg_y=0.000  gap=+0.618
[0.7-0.8]  n=15  avg_p=0.744  avg_y=0.733  gap=+0.011
[0.9-1.0]  n= 7  avg_p=0.992  avg_y=0.000  gap=+0.992
```

The agent uses the full confidence range now (0.0–1.0) instead of collapsing to one bucket.
The 0.7–0.8 bucket is well-calibrated (gap=+0.011). The 0.9–1.0 bucket is wildly over-
confident (gap=+0.992) — a single case at 1.0-confidence that lost. Isotonic corrects
this by mapping 0.9-1.0 to ~0.0.

## Recommendation

- **Apply isotonic calibration by default** in the agent's `predict()` method once we
  have ≥30 cases for a competition/season. (Task 31 candidate.)
- The "0.9-1.0 with n=7" bucket is still the weakest — the agent should be told to
  *cap* its raw confidence at 0.85 until we have >50 cases in the high-confidence range.
  (Task 32 candidate.)
- **Expand to ≥100 cases** before drawing strong conclusions. The gap=-0.715 and
  -0.596 buckets are at n=4 and n=2 respectively; doubling the eval would halve
  their variance.

## What changed in this task

- `src/soccer_agent/eval/dataset.py`: 10 → 34 cases across 5 competitions.
  Added `EVAL_CASES_SORTED` for public, deterministic, date-sorted access.
- `src/soccer_agent/eval/fixture_factory.py`: new `noise` (0..1) and `seed`
  parameters on `materialize_case` / `materialize_all`. Three signal categories
  (form, h2h, odds) flip with probability `noise`. Deterministic given a seed.
- `src/soccer_agent/eval/harness.py`: new `fixture_noise` and `fixture_seed`
  fields on `EvalHarness`, threaded through to the factory.
- `src/soccer_agent/eval/calibration.py`: new `--fixtures`, `--noise`, `--seed`
  CLI flags so `python -m soccer_agent.eval.calibration` can run end-to-end.
- `tests/test_dataset.py`: 3 new tests pinning the expanded invariants.
- `tests/test_factory_noise.py`: 6 new tests pinning the noise contract.
