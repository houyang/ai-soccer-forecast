# soccer-agent

A multi-tool agent for soccer match prediction. Given an upcoming match, it
autonomously pulls recent form and injury news, looks up head-to-head history,
checks weather and venue, compares against bookmaker odds, reasons about the
matchup, and outputs a prediction (pick, probabilities, confidence, written
rationale). It logs the prediction, waits for the result, and self-evaluates.

**Targets (Phase 2 scoring):** UEFA Champions League 2025/26, FIFA World Cup
2026 final (MetLife Stadium, East Rutherford NJ).

**Phase 1 status:** agent + tools + eval harness + CLI + REST API are shipped
and end-to-end-verified. ✅

**Phase 2 status:** accuracy iteration and the dashboard are largely
shipped. Current state on the 106-case eval (5 competitions: EPL,
LaLiga, SerieA, Bundesliga, UCL):

| metric                       | value          |
|------------------------------|----------------|
| Eval cases                   | 106            |
| Accuracy (stub LLM, numeric reasoner) | 70.8%   |
| Brier score (with calibrator)         | 0.152   |
| Raw Brier (no calibration)            | 0.304   |
| Reliability (calibrated, n=106)       | LOO-ECE 0.193 → 0.000 in-sample |
| Per-competition calibrators           | EPL, LaLiga, SerieA, Bundesliga (UCL uses global fallback, n<20) |

Remaining Phase 2 work: ResultScout wiring for the live UCL 25/26 + WC
26 finals, and the honest per-comp LOO cross-validation (needs n≥50
per comp before the signal is real). See `docs/phase2_future_work.md`.


## What you get

- A hand-rolled `PredictionAgent` orchestrator (Pydantic + asyncio, no
  LangGraph / LlamaIndex / Agents SDK — by design, so the eval harness can
  inspect every decision).
- Six tools, each with a real-API path and a deterministic fixture fallback:
  `form_recent`, `injury_news`, `h2h_history`, `weather_venue`, `odds_market`,
  `venue_info`. A tool that fails to reach the live API does NOT kill the
  prediction — the agent downgrades the signal to a neutral prior and
  surfaces a warning.
- A numeric reasoner (deterministic Elo + form blend; the Phase 1 scorer)
  and an LLM reasoner (`LLMReasoner`) that defaults to OpenAI-compatible
  chat completions. `LLMReasoner` falls back to `NumericReasoner` on any
  error or missing API key, so the agent always produces a prediction.
- An eval harness (`soccer_agent.eval.harness`) that replays 106 pinned
  historical matches across 5 competitions (EPL 29, LaLiga 25, SerieA 21,
  Bundesliga 20, UCL 11), computes Brier score, log loss, accuracy,
  per-class precision/recall/F1, per-competition breakdown, and writes a
  JSON summary.
- A `ResultScout` that polls a result feed and self-evaluates the latest
  prediction for each match.
- A CLI (`soccer-agent predict | evaluate | list | eval`) and a FastAPI
  server with 6 endpoints (see below). Both share the same agent.


## Quickstart

```bash
# 1. Install (editable, with dev tools)
pip install -e ".[dev]"

# 2. Materialize fixtures (one-time, populates data/fixtures/)
python -c "from soccer_agent.eval.fixture_factory import materialize_all; \
  from soccer_agent.eval.dataset import EVAL_CASES; \
  from pathlib import Path; \
  materialize_all(EVAL_CASES, Path('data/fixtures'))"

# 3. Predict one match
soccer-agent predict \
  --home-id man_city --away-id real_madrid \
  --venue-id puskas_arena --kickoff 2025-05-30T20:00:00 \
  --competition UCL --season 2025-26 --round final

# 4. Replay historical matches and score the agent
soccer-agent eval --output eval_summary.json

# 5. Start the REST API
uvicorn soccer_agent.api.server:app --reload
```

The first run writes a SQLite DB at `data/soccer_agent.db`. All paths and
keys come from environment variables — see `Configuration` below.


## Architecture

```
┌────────────────────┐        ┌────────────────────┐
│   soccer-agent CLI │        │   FastAPI server   │
│   (predict/eval/…) │        │   (6 REST endpoints)│
└────────┬───────────┘        └─────────┬──────────┘
         │                              │
         └──────────────┬───────────────┘
                        ▼
              ┌─────────────────────┐
              │  PredictionAgent    │   Pydantic + asyncio
              │  (orchestrator)     │   tool registry + reasoner chain
              └──────┬──────┬───────┘
                     │      │
        ┌────────────┘      └────────────┐
        ▼                                ▼
  ┌──────────┐                    ┌──────────────┐
  │ Tools    │  6 tools, each     │ Reasoners    │  NumericReasoner,
  │ registry │  with live API     │              │  LLMReasoner
  │          │  + fixture fallback│              │  (with numeric
  └──────────┘                    └──────────────┘   fallback)
        │                                │
        └────────────┬───────────────────┘
                     ▼
              ┌──────────────┐         ┌──────────────────┐
              │ SQLite (preds│         │ EvalHarness      │
              │ + results)   │◀────────│  (replay 24/25   │
              │              │         │   matches)       │
              └──────────────┘         └──────────────────┘
```

**Key files:**

- `src/soccer_agent/agent.py` — `PredictionAgent.predict(match, tool_names=…)`
  and `PredictionAgent.evaluate(match_id)`.
- `src/soccer_agent/tools/` — one module per tool. `base.py` defines the
  `Tool` protocol; `default_registry()` returns the 6-tool registry.
- `src/soccer_agent/reasoners/` — `NumericReasoner` (deterministic Elo +
  form blend; no network) and `LLMReasoner` (OpenAI-compatible chat
  completion, falls back to numeric on error).
- `src/soccer_agent/eval/harness.py` — `EvalHarness.run()` iterates
  `EVAL_CASES`, materializes fixtures, runs the agent, scores, returns a
  JSON summary.
- `src/soccer_agent/eval/dataset.py` — pinned `EVAL_CASES` (106
  historical matches across 5 competitions; ingest from `football-data.co.uk`).
- `src/soccer_agent/eval/calibration.py` — per-competition + global
  isotonic calibrators, LOO cross-validation, reliability table.
- `src/soccer_agent/calibration_store.py` — JSON round-trip for fitted
  calibrators; the agent's `predict()` consumes them at runtime.
- `src/soccer_agent/elo_state.py` — per-team home/away Elo, with a
  builder script (`scripts/build_elo_state.py`).
- `src/soccer_agent/eval/scoring.py` — `brier`, `log_loss`, `top_factor_hit`.
- `src/soccer_agent/eval/metrics.py` — accuracy, per-class P/R/F1.
- `src/soccer_agent/api/cli.py` — Click CLI.
- `src/soccer_agent/api/server.py` — FastAPI factory + 6 endpoints.
- `src/soccer_agent/db.py` — SQLite schema (`predictions`, `results`,
  `eval_runs`).
- `src/soccer_agent/llm/client.py` — `OpenAICompatClient` (one client for
  any OpenAI-shaped endpoint: OpenAI, OpenRouter, Ollama, llama-server,
  vLLM, LM Studio), thin back-compat `OpenAIClient` / `OpenRouterClient`,
  `StubLLMClient` (deterministic for tests), `get_client()` factory.


## CLI

```
soccer-agent predict
  --home-id H --away-id A --venue-id V --kickoff ISO
  [--competition UCL] [--season 2025-26] [--round final]
  [--tools form_recent,injury_news,h2h_history,weather_venue,odds_market,venue_info]

soccer-agent evaluate --match-id M --home-goals N --away-goals N
soccer-agent list [--limit 10]
soccer-agent eval [--output summary.json] [--reasoner numeric|llm]
```

All commands emit a single JSON object to stdout. Status messages go to
stderr so the stdout stays pipe-friendly. See `tests/test_cli.py` for
contract examples and `scripts/e2e_smoke.sh` for a real end-to-end run.


## REST API

`uvicorn soccer_agent.api.server:app --host 0.0.0.0 --port 8000`

| Method | Path                                | Body                                | Returns                        |
| ------ | ----------------------------------- | ----------------------------------- | ------------------------------ |
| GET    | `/health`                           | —                                   | `{status, db, db_path, …}`     |
| POST   | `/predictions`                      | `PredictionRequest`                 | `Prediction` (201)             |
| GET    | `/predictions`                      | `?limit=N`                          | `[Prediction]`                 |
| GET    | `/predictions/{match_id}`           | —                                   | `Prediction`                   |
| POST   | `/predictions/{match_id}/result`    | `{home_goals, away_goals}`          | `Prediction` with result block |
| GET    | `/metrics`                          | —                                   | Eval metrics summary           |

`PredictionRequest` schema (Pydantic):
```json
{
  "home_id": "man_city", "away_id": "real_madrid",
  "venue_id": "puskas_arena", "kickoff": "2025-05-30T20:00:00",
  "competition": "UCL", "season": "2025-26", "round": "final"
}
```

`Prediction` response (public contract — renames the internal `final_*` DB
columns to `pick`/`confidence`/`rationale`, nests result fields under
`result: {}`):
```json
{
  "prediction_id": "uuid", "match_id": "…", "created_at": "…",
  "pick": "home", "probs": {"home": 0.74, "draw": 0.11, "away": 0.15},
  "confidence": 0.74, "rationale": "…",
  "reasoner_outputs": [ { "reasoner": "numeric", "pick": "home", … } ],
  "signals": { "form_recent": {"tool": "form_recent", "ok": true, …}, … },
  "warnings": [],
  "model_versions": { … }
}
```


## Configuration (all env vars, no hard-coding)

| Var                              | Default                       | Notes                                      |
| -------------------------------- | ----------------------------- | ------------------------------------------ |
| `SOCCER_AGENT_DB_PATH`           | `data/soccer_agent.db`        | SQLite file                                |
| `SOCCER_AGENT_FIXTURES_DIR`      | `fixtures`                    | Tool fixture root                          |
| `SOCCER_AGENT_LLM_PROVIDER`       | `stub`                        | `stub`/`openai`/`openrouter`/`ollama`/`openai-compat` |
| `SOCCER_AGENT_LLM_API_KEY`       | _unset → numeric fallback_   | OpenAI-compatible key (ignored by `ollama`)  |
| `SOCCER_AGENT_LLM_BASE_URL`      | `https://api.openai.com/v1`   | Set to `http://127.0.0.1:11434/v1` for ollama, or any OpenAI-shaped endpoint |
| `SOCCER_AGENT_LLM_MODEL`         | `gpt-4o-mini`                 | Any chat-completion model name (`qwen2.5:0.5b` for ollama) |
| `SOCCER_AGENT_LLM_TEMPERATURE`   | `0.2`                         | Reasoner temperature                       |
| `SOCCER_AGENT_HTTP_TIMEOUT`      | `10`                          | Tool HTTP timeout (s)                      |
| `SOCCER_AGENT_HTTP_RETRIES`      | `2`                           | Tool HTTP retry count                      |
| `SOCCER_AGENT_REASONERS`         | `numeric`                     | Comma-list, e.g. `llm,numeric`             |
| `SOCCER_AGENT_FINAL_PICK_POLICY` | `numeric`                     | Tie-break / blend policy                   |
| `SOCCER_AGENT_API_HOST`          | `0.0.0.0`                     | FastAPI bind                               |
| `SOCCER_AGENT_API_PORT`          | `8000`                        | FastAPI port                               |
| `SOCCER_AGENT_SCOUT_POLL_SECONDS`| `300`                         | ResultScout poll interval                  |
| `SOCCER_AGENT_ODDS_API_KEY`      | _unset → fixture fallback_    | The Odds API key. When set, `odds_market` prefers the live feed over the JSON fixture. |
| `SOCCER_AGENT_ODDS_API_SPORT`    | _unset_                       | Odds API sport key, e.g. `soccer_uefa_champs_league`, `soccer_epl` |
| `SOCCER_AGENT_ODDS_API_EVENT_ID` | _unset_                       | Odds API event id (call `/v4/sports/<sport>/events` to discover) |
| `SOCCER_AGENT_ELO_STATE`         | _unset → fresh state_         | Path to a JSON-serialized `EloState` built by `scripts/build_elo_state.py`. When set, the numeric reasoner uses per-team home/away ratings instead of 1500/1500. |


## Eval

```bash
soccer-agent eval --output eval_summary.json
```

The harness replays every `EvalCase` in `src/soccer_agent/eval/dataset.py`,
materializing deterministic fixture files, running the agent, and scoring
the result. The summary JSON contains:

```json
{
  "n_total": 106, "n_resolved": 106,
  "accuracy": 0.708, "brier_mean": 0.152, "log_loss": 0.575,
  "per_competition": {
    "EPL":        {"n": 29, "accuracy": 0.69, "brier": 0.022},
    "LaLiga":     {"n": 25, "accuracy": 0.68, "brier": 0.052},
    "SerieA":     {"n": 21, "accuracy": 0.62, "brier": 0.013},
    "Bundesliga": {"n": 20, "accuracy": 0.80, "brier": 0.089},
    "UCL":        {"n": 11, "accuracy": 0.91, "brier": 0.003}
  }
}
```

Re-runs are idempotent: the harness re-uses existing prediction rows
when one already exists for a match. To force a clean re-run, delete
`data/soccer_agent.db` (or set `SOCCER_AGENT_DB_PATH` to a fresh path).

The 106 cases are sourced from `football-data.co.uk` public CSVs
(24/25 + 25/26 seasons, 5 competitions, scores are *real* historical
results, not synthesized). See `scripts/ingest_football_data.py` for
the ingest pipeline.

To refit a calibrator from a fresh eval DB:

```bash
python -m soccer_agent.eval.calibration \
  --db data/soccer_agent.db \
  --save-calibrator data/calibrators \
  --per-competition --min-n 20
```

See `docs/calibration.md` for the full report, the reliability table,
and the per-competition findings.


## Local LLM (Ollama, llama-server, vLLM)

The agent's LLMReasoner talks any OpenAI-shaped `/v1/chat/completions`
endpoint. Pointing it at a local ollama daemon is one env var:

```bash
export SOCCER_AGENT_LLM_PROVIDER=ollama
export SOCCER_AGENT_LLM_BASE_URL=http://127.0.0.1:11434/v1
export SOCCER_AGENT_LLM_MODEL=qwen2.5:0.5b
soccer-agent predict ...
```

See `docs/ollama-setup.md` for full install + tuning notes, including
the dlopen/LD_LIBRARY_PATH gotcha on the prebuilt ARM64 ollama release.
Live tests in `tests/test_ollama_live.py` are auto-skipped when the
daemon is unreachable, so the default `pytest` run stays fast and
deterministic.


## Prompt iteration

The agent's LLMReasoner ships with a default system prompt. To find a
better one, run a prompt sweep: same eval dataset under N prompt
candidates, scored head-to-head.

```bash
python scripts/prompt_iterate.py          # live ollama
SOCCER_AGENT_LLM_PROVIDER=stub python scripts/prompt_iterate.py  # fast CI
```

See `docs/prompt-iteration.md` for how to author candidates and what
to do with the results.


## Real bookmaker feed (The Odds API)

The `odds_market` tool ships with a **live** backend in addition to
its JSON fixture fallback. The live backend calls [The Odds API](https://the-odds-api.com/),
which aggregates odds from 30+ sportsbooks (Pinnacle, bet365, Betfair,
DraftKings, ...) for soccer, NFL, NBA, etc.

### Setup

1. Sign up at https://the-odds-api.com/ (free tier: 500 requests/month).
2. Export the key:
   ```bash
   export SOCCER_AGENT_ODDS_API_KEY="[REDACTED]"
   ```
3. Pick the sport + event id you want odds for. To discover event ids:
   ```bash
   curl "https://api.the-odds-api.com/v4/sports/soccer_uefa_champs_league/events?api_key=$SOCCER_AGENT_ODDS_API_KEY" | jq '.[].id'
   ```
4. Set the sport and event id:
   ```bash
   export SOCCER_AGENT_ODDS_API_SPORT="soccer_uefa_champs_league"
   export SOCCER_AGENT_ODDS_API_EVENT_ID="abc123..."
   ```

When all three vars are set, the next agent run that calls
`odds_market` will hit the live feed. If any of them is unset, the
tool silently falls back to the JSON fixture under
`fixtures/odds/`.

### Why three env vars and not one URL?

The Odds API event id is per-event, but a "match" in our domain
(home_team_id, away_team_id, kickoff_date) maps to a different
event id per bookmaker — and The Odds API normalises to one id per
event. For our two target events (UCL 25/26 final, WC 26 final),
the IDs are stable for weeks around the match. For weekly league
matches you'd need a small resolver (not in scope here; the fixture
is the right path for backtests).

### Devigging

The Odds API returns raw bookmaker odds. Different bookmakers
charge different overround (vig), so the implied probabilities
sum to >1.0. We **multiplicative-devig** to recover fair probs:

  fair_p = (1 / decimal_odds) / sum(1 / all_decimal_odds)

This is the simplest method and is good enough for sharp books
like Pinnacle. Better methods (Shin, Power) add 1-2% accuracy
on average; we keep the simple one so the devig step is auditable
in a single line of `src/soccer_agent/data/odds_api.py`.

See `docs/odds-api.md` for the API contract, error model, and
retriable-vs-not retriable semantics.


## Elo ratings

The numeric reasoner uses per-team **home/away** Elo ratings (not
the 1500/1500 placeholder that shipped in Phase 1). Build a state
file from your historical data on the host (fast CPU) and ship it
to the agent:

```bash
# 1. Pre-compute the state from your past-match JSONL
python scripts/build_elo_state.py \
    --matches data/past_matches.jsonl \
    --out    data/elo_state.json \
    --report

# 2. Point the agent at it
export SOCCER_AGENT_ELO_STATE=$(pwd)/data/elo_state.json
```

The agent picks it up automatically. See `docs/elo.md` for the math,
the home/away split rationale, and the limitations.

## Calibration

On the 106-case eval the agent's raw `confidence` is over-confident
in the 0.9–1.0 bucket (4/4 wrong with 93% average stated confidence).
Two things fix that:

1. **A 0.85 cap** on raw confidence before calibration — moves those
   wild 0.99 "I know it" picks down into the 0.8–0.9 bucket.
2. **An isotonic calibrator** (`isotonic.json`, fit on the eval set)
   applies a learned mapping to the capped value.

The two together cut Brier from 0.304 to 0.152 at unchanged 70.8%
accuracy. Per-competition calibrators (`isotonic_<COMP>.json`,
fitted separately for EPL/LaLiga/SerieA/Bundesliga with n≥20) are
also wired up — the agent's `predict()` uses per-comp when available
and falls back to the global otherwise.

See `docs/calibration.md` for the reliability table, the
per-competition numbers, and the honest caveats (per-comp vs
global is noise at n=20–29; we'll know more at n≥50).

## Dashboard

```bash
bash scripts/serve_dashboard.sh
# open http://127.0.0.1:8000/
```

Localhost-only, no auth, no build step. The page polls a single
endpoint (`GET /api/dashboard`) and renders summary tiles, a
reliability chart, a predict form, a record-result form, and the
predictions table. Auto-refreshes every 30s. See `docs/dashboard.md`
for the full contract.

## Development

```bash
# Run the full test suite
pytest -q

# Run just the e2e shell smoke (CLI + uvicorn + curl on every endpoint)
bash scripts/e2e_smoke.sh

# Run just the API shell smoke (uvicorn + curl, smaller scope)
bash scripts/e2e_api.sh
```

The test suite has 307 passing unit + integration tests covering
`tools/`, `reasoners/`, `agent.py`, `eval/`, `api/`, `db.py`, and `models.py`.
`tests/test_dataset.py` enforces invariants on the eval cases
(no duplicate ids, every case has a draw or two non-draw results in the
set, etc.) so adding a bad case is a failing test, not a silent score change.


## Roadmap

### Phase 1 (shipped ✅)

- [x] Agent + 6 tools + reasoner chain (numeric + LLM, with fallback)
- [x] Eval harness with deterministic fixtures
- [x] CLI + REST API (6 endpoints)
- [x] SQLite persistence for predictions, results, tool calls
- [x] E2E smoke test (`scripts/e2e_smoke.sh`)

### Phase 2 — accuracy (largely shipped ✅)

- [x] Per-team home/away Elo ratings (`docs/elo.md`)
- [x] LLM reasoner with stub/openai/openrouter/ollama backends
- [x] Prompt iteration harness (`scripts/prompt_iterate.py`)
- [x] Live bookmaker odds (The Odds API) with devigging
- [x] Confidence calibration (isotonic, with 0.85 cap, 106-case eval)
- [x] Per-competition calibrators (EPL/LaLiga/SerieA/Bundesliga;
      UCL uses global fallback until n≥20)
- [x] Eval set expansion to 106 historical cases (football-data.co.uk)

### Phase 2 — dashboard (shipped ✅)

- [x] FastAPI static frontend, no build step, localhost-only
- [x] Summary tiles + reliability chart + predict form + record-result form
- [x] Per-competition breakdown + calibration monitor

### Phase 2 — remaining (deferred — see `docs/phase2_future_work.md`)

- [ ] **ResultScout live wiring** for UCL 25/26 + WC 26 finals
      (predict from kickoff -24h, self-evaluate within 30 min of FT)
- [ ] **Per-comp LOO cross-validation** (needs n≥50 per comp)
- [ ] **Multi-class calibration** (3-way H/D/A probs) instead of
      1D "did the pick match the actual?" reduction
- [ ] **Online / rolling recalibration** (cron on a 90-day window)
- [ ] **Public deployment** (currently localhost-only)


## License

Internal. Not yet released.
