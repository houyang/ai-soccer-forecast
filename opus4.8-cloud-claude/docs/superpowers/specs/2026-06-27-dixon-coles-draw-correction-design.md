# Dixon–Coles Draw Correction for the Scoreline Model

**Date:** 2026-06-27
**Status:** Approved (design)

## Goal

Improve prediction accuracy by correcting the independent-Poisson scoreline model's systematic
**underestimation of draws**, measured against the World Cup matches played so far.

## Evidence (backtest of committed predictions vs 27 played matches)

The pre-tournament group-stage predictions (`prediction/worldcup-2026-predictions.json`) were
scored against the actual results now in `data/worldcup-2026.json`:

| Metric | Current model |
|---|---|
| Outcome accuracy (argmax of H/D/A) | 51.9% |
| Brier score (lower better) | 0.577 |
| Log-loss | 0.950 |
| Avg predicted P(draw) | 0.23 |
| Actual draw rate | 0.37 (10/27) |
| Predicted outcome mix | H19 / D0 / A8 |
| Actual outcome mix | H14 / D10 / A3 |

**Root cause:** independent Poisson underestimates draws (a well-documented property). The model
assigns draws only 23% mass vs the 37% observed and never ranks a draw as the single most likely
outcome.

**Metric choice:** argmax outcome accuracy stayed at 51.9% across every variant tried — expected,
since a ~37%-likely draw rarely beats a ~40% favourite. The improvement target is therefore
**probability calibration** (Brier / log-loss), not argmax accuracy.

## Decision

Apply a **Dixon–Coles low-score correction** to the scoreline matrix, controlled by a single
parameter `DRAW_RHO = -0.15`.

Backtest of candidate corrections on the 27 played matches (re-running the model's pre-tournament
λ for each match, leakage-free):

| Variant | Brier | Log-loss | Avg P(draw) |
|---|---|---|---|
| baseline (rho=0) | 0.577 | 0.950 | 0.23 |
| DC rho=−0.10 | 0.570 | 0.931 | 0.25 |
| **DC rho=−0.15 (chosen)** | **0.566** | **0.923** | **0.26** |
| DC rho=−0.20 | 0.563 | 0.915 | 0.27 |

`rho=−0.15` is a conservative point on a monotonic curve: a clear improvement without chasing the
extreme of a 27-game sample.

### Alternatives rejected

- **Flat diagonal draw boost** — similar metric gains but unprincipled; inflates *all* draw
  scores including unrealistic 4-4/5-5.
- **Recalibrating `BASE_MATCH_GOALS` / supremacy** — the tournament is high-scoring (3.26
  goals/game vs the model's 2.60), but scaling goals *up* worsened Brier (more goals → fewer
  draws), and tuning constants on 27 games risks overfitting. Left untouched.

## Design

### Dixon–Coles correction in `predict._scoreline_matrix`

Add a module constant `DRAW_RHO = -0.15`. After building the independent-Poisson matrix, multiply
the four low-score cells by the Dixon–Coles `tau` factors, then renormalize the whole matrix to a
proper probability distribution:

```
tau(0,0) = 1 - lambda_home * lambda_away * rho
tau(0,1) = 1 + lambda_home * rho
tau(1,0) = 1 + lambda_away * rho
tau(1,1) = 1 - rho
```

With `rho = -0.15` (negative), `tau(0,0)` and `tau(1,1)` exceed 1 (boost 0-0 and 1-1) while
`tau(0,1)` and `tau(1,0)` fall below 1 (reduce 1-0 and 0-1) — net draw inflation. Each factor is
floored at a small positive epsilon for safety. `rho = 0` reproduces the current independent
matrix exactly (clean no-op fallback).

`_scoreline_matrix` signature gains an optional `rho: float = DRAW_RHO` parameter so it stays
testable in isolation. All downstream consumers — `_outcome_probs`, `_modal_score`,
`predict_one`, `top_scorelines`, and the adjustment-aware `_predict` — inherit the correction
unchanged because they all read this matrix.

### Scope of change

- `src/soccer/worldcup/predict.py`: add `DRAW_RHO`, the `tau` correction, and renormalization in
  `_scoreline_matrix`. No other model logic changes.
- No new dependencies; pure-function change; no import-time side effects.

## Testing

- **New** (`tests/worldcup/test_predict.py` or a focused test): for equal λ (e.g. 1.3/1.3),
  `_scoreline_matrix` with `DRAW_RHO` yields a higher summed diagonal (P(draw)) than with
  `rho=0`; the matrix sums to 1.0 within tolerance; `rho=0` matches the independent product
  cell-for-cell.
- **Update** existing predict/CLI assertions whose exact probability or modal-score values shift
  because of the correction.
- **Validation gate (not a unit test):** re-run the backtest after the change and confirm Brier
  and log-loss are at least as good as the values in the table above; the change ships only if the
  metrics actually improve.

## Out of scope / follow-ups

- Goal-level recalibration (`BASE_MATCH_GOALS`) — flagged by the 2.60-vs-3.26 gap but left for a
  later pass with more data, since it trades against draw calibration.
- A reusable evaluation module (`soccer.worldcup.evaluate`) to score predictions vs results — the
  backtest here is a throwaway script; promoting it is a possible future task.

## Definition of done

`ruff` + `mypy` + full `pytest` pass; backtest confirms Brier/log-loss improvement; today's
(2026-06-27 US-Pacific) cards regenerated with the improved model in both PDF and JSON, named by
matchup.
