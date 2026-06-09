# Dashboard (Task 29)

> A localhost-only, no-auth, no-build-step dashboard for the
> prediction agent. Single `uvicorn` command serves both the API
> and the static page.

## Running it

```bash
bash scripts/serve_dashboard.sh
# open http://127.0.0.1:8000/
```

Defaults:

- Binds `127.0.0.1:8000` (localhost only, no auth — by design for
  personal use).
- Uses the **stub** LLM provider so the server boots without an
  LLM installed. Override with `SOCCER_AGENT_LLM_PROVIDER=ollama`
  (or `openai` / `openrouter`) for live predictions.
- DB path defaults to `data/soccer_agent.db`. Override with
  `SOCCER_AGENT_DB_PATH=...`.

## What the page shows

1. **Summary tiles** — predictions, resolved, accuracy, Brier,
   log-loss, calibration ECE. Tiles are color-coded:
   - accuracy ≥ 50% → green, < 50% → red
   - Brier ≤ 0.2 → green, otherwise red
   - ECE ≤ 0.05 → green (the target from `docs/calibration.md`),
     otherwise red
2. **Reliability chart** — for each confidence bucket (x), the
   observed actual win rate (y) is plotted. The vertical bar shows
   the gap between stated and actual. Green = over-confident
   (`actual > stated`, bad), red = under-confident (`actual <
   stated`, conservative). The dashed line is perfect calibration.
3. **Predict form** — `POST /predictions` with the same shape as
   the CLI: `home_id`, `away_id`, `venue_id`, `kickoff`, optional
   `competition` and `round`. After the agent runs, the page
   refreshes automatically and the new row appears in the table.
4. **Record-result form** — `POST /predictions/:match_id/result`
   with `home_goals` and `away_goals`. Triggers the same
   `evaluate()` path the CLI uses; the row's correctness, Brier,
   and other fields update in place.
5. **Predictions table** — most recent first, with pick,
   confidence, result, Brier, and creation timestamp. The pick
   pill turns green (correct) or red (wrong) once a result is
   recorded.

The page auto-refreshes every 30s.

## API contract

The page polls **one** endpoint:

```
GET /api/dashboard
```

Response shape:

```json
{
  "summary": {
    "n_predictions": 10,
    "n_resolved": 10,
    "accuracy": 0.7,
    "brier": 0.183,
    "log_loss": 0.567,
    "calibration_ece": 0.515
  },
  "predictions": [
    {
      "match_id": "manchester_city_vs_real_madrid__2026-05-30",
      "pick": "home",
      "confidence": 0.74,
      "rationale": "...",
      "result": { "home_goals": 2, "away_goals": 1,
                  "was_correct": true, "brier": 0.118 },
      "created_at": "2026-05-30T12:00:00+00:00"
    }
  ],
  "calibration": {
    "n_samples": 10,
    "raw": { "ece": 0.515, "brier": 0.373,
             "reliability": [
               { "lo": 0.3, "hi": 0.4, "bin_midpoint": 0.35,
                 "count": 1, "avg_actual": 1.0, "bin_label": "[30-40]" }
             ] },
    "loo": { "identity": {...}, "platt": {...}, ... }
  },
  "generated_at": "2026-06-04T18:30:00+00:00"
}
```

The endpoint composes three existing sources; no new evaluation
logic:

- `EvalHarness.run()` for the summary
- `db.list_predictions()` for the rows
- `run_calibration_report()` for reliability

Caching or static hosting can sit in front of `/api/dashboard`
later without changing the page.

## File layout

```
src/soccer_agent/api/
  server.py            # FastAPI app (adds /api/dashboard + static mount)
  static/
    index.html         # single-file UI, no build
    app.js             # vanilla JS, fetches /api/dashboard
    style.css          # dark theme, no framework
scripts/
  serve_dashboard.sh   # one-command launcher
tests/
  test_dashboard.py    # 13 tests: endpoint contract + static serving
```

## Why this design

- **One endpoint, one payload.** The page polls once and renders.
  No N+1 round-trips, no client-side state to keep in sync.
- **No build step.** The page is `index.html` + `app.js` +
  `style.css`. No bundler, no node_modules, no toolchain to
  maintain. Open in any browser, edit in any editor.
- **No auth, localhost only.** The use case is "check the
  dashboard on my machine during a match". Public hosting would
  need auth, rate limiting, etc. — out of scope.
- **Reuses existing logic.** The endpoint is a 30-line composer
  over `EvalHarness`, `db.list_predictions()`, and
  `run_calibration_report()`. No new evaluation math, no new
  schema.
