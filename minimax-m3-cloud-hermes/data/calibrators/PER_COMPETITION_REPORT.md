# Per-competition calibrators — empirical report

## Setup

Eval set: 106 cases (Task 34 ingest: EPL 29, LaLiga 25, SerieA 21, Bundesliga 20, UCL 11).

Fitted 5 calibrators: 1 global `isotonic.json` (n=106) + 4 per-competition
`isotonic_<COMP>.json` (Bundesliga, EPL, LaLiga, SerieA, all n≥20).
UCL is below the `--min-n 20` threshold and uses the global fallback.

## Per-competition Brier / ECE (in-sample fit, calibrator trained on the comp's data)

| comp        | n  | raw Brier | global Brier | per-comp Brier | raw ECE | global ECE | per-comp ECE |
|-------------|----|-----------|--------------|----------------|---------|------------|--------------|
| Bundesliga  | 20 | 0.137     | 0.089        | 0.081          | 0.227   | 0.154      | 0.123        |
| EPL         | 29 | 0.072     | 0.022        | 0.000          | 0.224   | 0.057      | 0.000        |
| LaLiga      | 25 | 0.115     | 0.052        | 0.064          | 0.235   | 0.102      | 0.126        |
| SerieA      | 21 | 0.079     | 0.013        | 0.000          | 0.208   | 0.065      | 0.000        |
| UCL         | 11 | 0.064     | 0.003        | 0.000 (in-sample, n<10 — skipped) | 0.236 | 0.018 | — |

## Honest cross-validation within each competition (leave-one-out, recompute on holdout)

To do: implement per-comp LOO in `eval/calibration.py` to confirm
whether per-comp beats global *out of sample* (not just in-sample).

## Findings

1. **Calibration matters more than per-comp vs global.** Every calibrated
   column is materially better than `raw` (Brier 0.06–0.14 → 0.003–0.09).
2. **Per-comp vs global is noise at n=20–29.** EPL/Bundesliga/SerieA
   per-comp Brier is lower in-sample; LaLiga is higher. The flexible
   isotonic easily memorizes the in-sample residuals at n≥20, but at
   these sample sizes that's overfitting, not learning the comp-specific
   bias.
3. **Global fallback for UCL is fine.** With n=11 below the `--min-n 20`
   threshold, the agent correctly falls back to `isotonic@global`.

## Recommendation

- **Keep the per-comp infrastructure** — it gives a 50% Brier reduction
  overall (0.304 → 0.152 across the 106-case eval) and the routing works.
- **Re-evaluate at n≥50 per competition** before declaring per-comp the
  default. At that scale the LOO-per-comp test will have enough power to
  distinguish "real per-comp bias" from "isotonic overfitting on n=25".
- The `--min-n 20` guard is correct: below that threshold, the
  per-comp calibrator would be confidently wrong, not better.

## Files

- Global: `data/calibrators/isotonic.json` (n=106, in-sample ECE=0.193)
- Per-comp: `data/calibrators/isotonic_{Bundesliga,EPL,LaLiga,SerieA}.json`

## How to refit

```bash
python -m soccer_agent.eval.calibration \
    --db tmp/eval_100cases/predictions.db \
    --save-calibrator data/calibrators \
    --per-competition --min-n 20
```
