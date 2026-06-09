# Football Match Prediction Agent — Implementation Plan

> **For Hermes:** Execute task-by-task using subagent-driven-development. Each task ships one test, one impl, one commit. Both reviews (spec + quality) must pass before moving on.

> **Status note:** This plan is the **v2 revision** that supersedes the v1 plan referenced in the user's earlier chat (saved at `plans/2026-06-02_144614-soccer-prediction-agent.md` in the host's `.hermes/`). v1 was a Phase-1-only plan with 20 tasks and a 0..100 confidence scale. v2 (this file) explicitly splits the work into Phase 1 and Phase 2, normalizes confidence to 0..1, and adds the two named use cases (UCL 25/26, WC 26 final) as first-class entries in the eval set.

## Goal

Build a multi-tool agent that, given an upcoming match, autonomously gathers context (form, injuries, H2H, weather, venue, bookmaker odds), reasons about the matchup, emits a prediction (pick, confidence, rationale), logs it, waits for the result, and self-evaluates.

**Two immediate use cases** (named throughout the code & eval set):
- **UCL 2025/26** — currently in progress; we'll start tracking from the round-of-16 onward.
- **FIFA World Cup 2026 final** — kickoff 2026-07-19, MetLife Stadium.

**Phase 1 focus:** agent architecture, tool design, eval harness, CLI + API surface.
**Phase 2 focus:** accuracy iteration against the eval set, calibrated confidence, polished UI/dashboard.

## Architecture

The agent has four layers: a thin entry surface (CLI / FastAPI), an orchestrator (PredictionAgent), a tool layer (six tools, each with a real-API path and a fixture fallback), and a reasoner layer (a deterministic numeric reasoner and an optional LLM reasoner). Results come from a ResultScout that polls a feed, and an eval harness replays historical matches to score the reasoners end-to-end.

### Why hybrid reasoning

A closed-form numeric model (Elo + form + H2H + injury impact + weather + market-implied probability) is reproducible, free, and easy to A/B test against. An LLM reasoner reads the same structured context and emits its own pick and rationale. The agent returns both; the eval harness scores both. The numeric model is the baseline we measure Phase 2 accuracy work against.

### Why fixture fallbacks

Every tool has a primary (network) path and a deterministic stub keyed off a JSON file. The stub path is what the eval harness uses for reproducibility. Real provider swap is a single-class change (Odds API, API-Football, etc.). All fixtures live under `fixtures/`.

### Tool contract (the single most important design decision)

Every tool is a `BaseTool[TIn, TOut]` with **strict Pydantic input and output schemas** that double as:
1. The contract the LLM sees when picking tools.
2. The validation boundary that keeps the reasoner honest.
3. The serialization format the eval harness and dashboard consume.

This means swapping a fixture for a live API is "fill in the same Pydantic shape" — and the rest of the system doesn't notice.

## Tech Stack

- Python 3.11, FastAPI, Streamlit, SQLite (stdlib), pytest, ruff.
- HTTP via httpx; LLM via OpenAI-compatible API (default = OpenRouter, swap to Anthropic/OpenAI via env).
- No frontend build step. Streamlit = dashboard. FastAPI = JSON surface. CLI = operator entry.

## Files (target tree)

```
ai-agent-dev/
├── pyproject.toml
├── README.md
├── .env.example
├── src/soccer_agent/
│   ├── __init__.py
│   ├── config.py
│   ├── db.py
│   ├── models.py
│   ├── llm.py                 # OpenRouter adapter
│   ├── agent.py
│   ├── result_scout.py
│   ├── tools/
│   │   ├── base.py            # BaseTool, ToolRegistry
│   │   ├── _fixtures.py
│   │   ├── form_recent.py
│   │   ├── injury_news.py
│   │   ├── h2h_history.py
│   │   ├── weather_venue.py
│   │   ├── odds_market.py
│   │   └── venue_info.py
│   ├── reasoners/
│   │   ├── base.py
│   │   ├── numeric.py
│   │   └── llm.py
│   ├── eval/
│   │   ├── metrics.py
│   │   └── harness.py
│   └── api/
│       ├── app.py
│       └── cli.py
├── fixtures/
│   ├── matches.jsonl
│   ├── form/  injuries/  h2h/  weather/  odds/  venues/
├── dashboard/                  # Streamlit app (Phase 2)
│   └── app.py
├── docs/
│   └── plans/                 # this file lives here
└── tests/
    ├── test_db.py  test_models.py  test_registry.py
    ├── test_fixtures.py
    ├── test_tools_form_recent.py  test_tools_injury_news.py
    ├── test_tools_h2h_history.py  test_tools_weather_venue.py
    ├── test_tools_odds_market.py  test_tools_venue_info.py
    ├── test_reasoners.py
    ├── test_agent.py  test_cli.py
    ├── test_result_scout.py
    ├── test_eval_metrics.py  test_eval_harness.py
    ├── test_api.py
    └── test_e2e.py
```

---

## Core Schemas (Phase 1, Task 2)

These flow through the whole system. Get them right once.

```python
# src/soccer_agent/models.py
from datetime import datetime
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, NonNegativeFloat

class Competition(str, Enum):
    UCL = "UEFA Champions League"
    FIFA_WC = "FIFA World Cup"

class Team(BaseModel):
    name: str; country: str | None = None

class Match(BaseModel):
    match_id: str                # slug, e.g. "ucl-2025-final"
    competition: Competition
    kickoff: datetime
    home: Team
    away: Team
    venue_id: str                # FK to venues
    neutral_venue: bool = False

class Signal(BaseModel):
    """Anything a tool produced — stored verbatim on the Prediction row."""
    tool: str
    ok: bool
    data: dict | None = None
    error: str | None = None
    source: Literal["live", "fixture", "stub"]

class ReasonerOutput(BaseModel):
    name: str
    pick: Literal["home", "draw", "away"]
    probs: dict[str, NonNegativeFloat]    # home/draw/away, sum=1
    confidence: float                      # 0..1 (NOT 0..100 — use fractions throughout)
    rationale: str
    factors: list[str] = []
    warnings: list[str] = []

class Prediction(BaseModel):
    match_id: str
    final_pick: Literal["home", "draw", "away"]
    final_probs: dict[str, NonNegativeFloat]
    final_confidence: float
    final_rationale: str
    signals: list[Signal]
    reasoner_outputs: list[ReasonerOutput]
    warnings: list[str]
    model_version: str
    created_at: datetime

class Result(BaseModel):
    match_id: str
    home_goals: int; away_goals: int
    result: Literal["home", "draw", "away"]    # derived
    settled_at: datetime

class EvalRun(BaseModel):
    run_id: str
    dataset_path: str
    n: int
    metrics: dict                          # per-reasoner + final
    started_at: datetime; finished_at: datetime
    judge: dict | None = None
```

**Critical schema rule:** `confidence` is **always 0..1** (not 0..100). All consumer code must assume fractions. This avoids the v1 plan's confusing 0..100 and matches PyTorch/Brier conventions.

---

## Phase 1 Tasks

Each task is 2–5 minutes of focused work. Every task: write test, run it (fail), write impl, run it (pass), commit.

### Task 1: Project skeleton + pyproject

**Files:** `pyproject.toml`, `src/soccer_agent/__init__.py`, `.gitignore`, `README.md`.

- pyproject: package `soccer-agent`, deps `httpx`, `pydantic>=2`, `pydantic-settings`, `fastapi`, `uvicorn`, `streamlit`, `pytest`, `pytest-asyncio`, `ruff`. Python ≥3.11. Console script `soccer-agent`.
- README: one-paragraph "what this is" and the 5-command quickstart.
- .gitignore: `__pycache__`, `.venv`, `data/*.db`, `.env`, `*.egg-info`, `reports/`.

**Verify:** `pip install -e .` succeeds.
**Commit:** `chore: project skeleton`

---

### Task 2: Config + DB + models

**Files:** `src/soccer_agent/config.py`, `src/soccer_agent/db.py`, `src/soccer_agent/models.py`.

- `config.py` — pydantic-settings: `SOCCER_AGENT_DB_PATH`, `SOCCER_AGENT_LLM_API_KEY`, `SOCCER_AGENT_LLM_BASE_URL` (default `https://openrouter.ai/api/v1`), `SOCCER_AGENT_LLM_MODEL` (default `anthropic/claude-sonnet-4`), `SOCCER_AGENT_LLM_TEMPERATURE` (default `0.2`), `SOCCER_AGENT_HTTP_TIMEOUT` (default `10`), `SOCCER_AGENT_REASONERS` (csv, default `numeric,llm`), `SOCCER_AGENT_USE_FIXTURES` (default `True`).
- `db.py` — stdlib sqlite3, idempotent `init_db()`, context-managed connections, migrations as `CREATE TABLE IF NOT EXISTS`. Tables: `predictions`, `results`, `eval_runs`, `tool_calls`.
- `models.py` — Pydantic v2 schemas above (`Team`, `Match`, `Signal`, `ReasonerOutput`, `Prediction`, `Result`, `EvalRun`, `MatchContext`, `ToolErrorPayload`).

**Tests:** `tests/test_db.py` (init twice idempotent; insert + fetch prediction), `tests/test_models.py` (Pydantic rejects bad confidence >1, missing fields, probs don't sum to 1).
**Verify:** `pytest -q` → 4+ tests pass.
**Commit:** `feat: config, sqlite store, schemas`

---

### Task 3: Tool protocol + registry

**Files:** `src/soccer_agent/tools/base.py`, `src/soccer_agent/tools/registry.py`, `src/soccer_agent/tools/__init__.py`.

- Tool protocol: `name`, `description`, `input_model`, `output_model`, `async run(input) -> output`.
- `ToolError(Exception)` carrying `source`, `message`, `retriable: bool`.
- `ToolRegistry`:
  - `register(tool)`, `get(name)`, `names` list.
  - `async run(name, payload, *, timeout=10, retries=2, use_fixture_if_missing=True) -> ToolOutput` returning discriminated union `{ok: True, data, source: live|fixture}` | `{ok: False, error}`.
  - On tool error, retry retriable errors with exponential backoff, then capture. **Never raise out of `run`.**
  - Logs every call to `tool_calls` table.

**Tests:** `tests/test_registry.py` — success path; tool raising `ToolError` is captured, not raised; timeout triggers retry; unknown tool raises `KeyError`; fixture fallback is used when configured.
**Commit:** `feat: tool protocol and registry`

---

### Task 4: Fixture loader

**Files:** `src/soccer_agent/tools/_fixtures.py`, `fixtures/` tree (empty subdirs created).

- `load_fixture(tool_name, match_id) -> dict` reads `fixtures/{tool_name}/{match_id}.json`; raises `ToolError(source=fixture, message=no fixture for {match_id})` when missing.
- `has_fixture(tool_name, match_id) -> bool` for cheap checks.
- Pure stdlib. No Pydantic here — the tool validates the loaded dict against its own output model.

**Tests:** `tests/test_fixtures.py` — loads existing; errors clearly on missing.
**Commit:** `feat: fixture loader`

---

### Task 5: form_recent tool

**Files:** `src/soccer_agent/tools/form_recent.py`, `fixtures/form/` (seed ≥5 examples).

- Input: `{home_team_id, away_team_id, last_n: int = 5}`.
- Output: per-team `{played, won, drawn, lost, gf, ga, points, last5_form_string}`.
- Primary path: stubbed httpx call (real provider swap in Phase 2).
- Stub path: fixture JSON shaped identically.

**Tests:** stub path returns known values; shape matches `FormOutput`; missing fixture → `ToolError`.
**Commit:** `feat: form_recent tool`

---

### Task 6: injury_news tool

**Files:** `src/soccer_agent/tools/injury_news.py`, `fixtures/injuries/`.

- Input: `{home_team_id, away_team_id, as_of: datetime}`.
- Output: `{home: [InjuryReport], away: [InjuryReport]}` where `InjuryReport = {player, status, reported_at, source}`.
- Helper `impact_score(reports) -> float` (out=1.0, doubt=0.5, questionable=0.2, capped per-team at 2.0).

**Tests:** impact sum; missing fixture → `ToolError`; cap enforced.
**Commit:** `feat: injury_news tool`

---

### Task 7: h2h_history tool

**Files:** `src/soccer_agent/tools/h2h_history.py`, `fixtures/h2h/`.

- Input: `{home_team_id, away_team_id}`.
- Output: `{meetings: [H2HMeeting], home_wins, away_wins, draws, last_meeting, last_winner}`.
- **Lookup key = sorted** `(a, b)` so order does not matter.

**Tests:** order-invariant; counts add up; missing pair → `ToolError`.
**Commit:** `feat: h2h_history tool`

---

### Task 8: weather_venue tool

**Files:** `src/soccer_agent/tools/weather_venue.py`, `fixtures/weather/`.

- Input: `{venue_id, kickoff: datetime, lat, lon}`.
- Output: `{temp_c, precip_mm, wind_kph, conditions, is_dome, playability_risk: low|medium|high}`.
- Primary: Open-Meteo forecast endpoint (free, no key) when kickoff is ≤14d out.
- Risk rule: high if precip >5mm AND outdoor, OR wind >50kph outdoor, OR temp <0°C outdoor; medium if precip >2mm outdoor; else low.

**Tests:** risk thresholds; Open-Meteo URL is built correctly (mock httpx); fixture fallback works; dome flag suppresses risk.
**Commit:** `feat: weather_venue tool with Open-Meteo`

---

### Task 9: odds_market tool

**Files:** `src/soccer_agent/tools/odds_market.py`, `fixtures/odds/`.

- Input: `{home_team_id, away_team_id, match_id}`.
- Output: `{bookmakers: [{name, home, draw, away}], implied_probs: {home, draw, away} (vig-free, normalized), market_consensus_pick}`.
- Parser interface: `OddsParser` protocol; default parser reads fixture. Real provider swap = single-class replacement.

**Tests:** implied probs sum to ~1.0; vig removal is monotonic; missing fixture → `ToolError`.
**Commit:** `feat: odds_market tool`

---

### Task 10: venue_info tool

**Files:** `src/soccer_agent/tools/venue_info.py`, `fixtures/venues/`.

- Input: `{venue_id}`.
- Output: `{name, city, country, capacity, surface, is_neutral, altitude_m}`.
- **Seed at least `metlife_stadium` (WC26 final), `puskas_arena` (UCL final), `old_trafford`, `san_siro`, `wembley`.**

**Tests:** lookup; missing → `ToolError`; neutral flag set for finals.
**Commit:** `feat: venue_info tool`

---

### Task 11: Reasoner protocol + numeric reasoner

**Files:** `src/soccer_agent/reasoners/base.py`, `src/soccer_agent/reasoners/numeric.py`.

- Reasoner protocol: `name`, `async reason(context: MatchContext) -> ReasonerOutput(pick, probs, confidence, rationale, factors)`.
- **Numeric reasoner** (deterministic, the Phase 2 baseline):
  - **Elo baseline:** each team starts at 1500, updated from last 5 results with K=20, home advantage +50. Projected win prob via standard Elo formula.
  - **Form delta:** 1.0/0.5/0.0 weights, z-scored across teams.
  - **H2H prior:** empirical home/draw/away from last 10 meetings, Laplace-smoothed (+1 each).
  - **Injury impact:** subtract `0.04 × impact_score` from team win prob.
  - **Weather:** high risk → +3% draw probability (capped at 50%); medium → +1%.
  - **Odds blend:** `0.6 × numeric + 0.4 × market_implied`.
  - **Final probs:** renormalize; pick = argmax; confidence = `1 − H/H_max` where H is normalized Shannon entropy; rationale = templated top-3 factors by absolute weight.
- **Deterministic. Same inputs → same output. Critical for the eval harness.**

**Tests:** `tests/test_reasoners.py::test_numeric_*` — known-input/known-output; Elo math correct; confidence in [0,1]; draw prob can exceed either side; rationale lists top factors.
**Commit:** `feat: numeric reasoner (Elo + form + H2H + market blend)`

---

### Task 12: LLM reasoner (with stub fallback)

**Files:** `src/soccer_agent/reasoners/llm.py`, `src/soccer_agent/llm.py`.

- `llm.py` — OpenAI-compatible chat completion via httpx (works with OpenRouter, Anthropic, OpenAI, Ollama). Retries (3x, exp backoff) on 429/5xx. JSON-mode toggle for structured outputs.
- `reasoners/llm.py` — system prompt mandates strict JSON: `pick`, `probs`, `confidence`, `rationale`. No markdown, no prose. Retries once on parse failure; second failure → stub.
- **If `SOCCER_AGENT_LLM_API_KEY` unset → stub:** `pick = market_consensus`, `probs = market_implied`, `confidence = max(market)`, `rationale = "LLM disabled; falling back to market consensus."`, with a `warnings: [llm_disabled]` field. This is what runs in CI / on the eval harness by default — zero cost.

**Tests:** stub path returns deterministic output; mocked httpx call returns parsed output; malformed JSON → second attempt + warning.
**Commit:** `feat: llm reasoner with stub fallback`

---

### Task 13: PredictionAgent orchestrator

**Files:** `src/soccer_agent/agent.py`, `tests/test_agent.py`.

- `PredictionAgent(registry, reasoners, db, final_pick_policy=numeric)`.
- `async predict(match: Match) -> Prediction`:
  1. Build `MatchContext(match, signals={})`.
  2. Run each registered tool in parallel via `asyncio.gather(return_exceptions=True)`, collecting `ToolOutputs`. Partial failures become warnings; never abort.
  3. Call each reasoner sequentially. ReasonerOutput failures are warnings too.
  4. Decide `final_pick`:
     - `numeric` (default): use numeric reasoner output.
     - `llm`: use LLM reasoner output if available else numeric.
     - `split`: return all and mark disagreement.
  5. Persist `Prediction` row with all signals, all reasoner outputs, the final pick, the model versions, and the warnings list.

**Tests:** end-to-end with stub reasoners + fixture-only match; partial tool failure surfaces in warnings; `final_pick_policy` switches; both reasoners' outputs persisted.
**Commit:** `feat: PredictionAgent orchestrator`

---

### Task 14: CLI

**Files:** `src/soccer_agent/api/cli.py`, `src/soccer_agent/api/__init__.py`. Console script entry point.

- `soccer-agent predict MATCH_ID [--reasoners numeric,llm] [--db PATH] [--no-llm]` → pretty JSON to stdout.
- `soccer-agent eval [--dataset PATH] [--judge] [--max-n N] [--reasoners ...]` → runs harness, prints summary table.
- `soccer-agent settle` → run one tick of ResultScout.
- `soccer-agent serve [--host 0.0.0.0] [--port 8000]` → uvicorn the FastAPI app.
- `soccer-agent --help` works.

**Tests:** `tests/test_cli.py` — `predict` happy path via subprocess on a fixture match.
**Commit:** `feat: CLI`

---

### Task 15: ResultScout + self-evaluation

**Files:** `src/soccer_agent/result_scout.py`, `tests/test_result_scout.py`.

- `ResultScout(feed: ResultFeed, db, eval_runner)`.
- `ResultFeed` protocol: `async get_result(match_id) -> Result | None`. Default impl: read from `results` table; CSV-based stub for tests.
- `async run_once()`: find predictions where `match.kickoff < now` and no result row yet, ask the feed, persist, and run per-prediction self-eval.
- **Per-prediction self-eval (first-class feature):**
  - `was_final_pick_correct: bool`
  - `brier_contribution: float`
  - `top_cited_factor_hit: bool` — post-hoc, did the top reason cited actually align with the result? (binary, kept simple)
  - `critique: str` — re-invoke the LLM reasoner with the actual outcome appended, ask it to write a 3-sentence self-critique: "Was the confidence calibrated? What did the tool outputs miss?"
- All stored on the `results` row. The critique is the most important field for Phase 2 — it feeds back into prompt iteration.
- `async run_forever(poll_seconds=300)` for production use; `run_once()` for tests and the CLI `settle` command.

**Tests:** scout finds pending predictions, persists results, writes eval rows including critique. Mock the feed.
**Commit:** `feat: ResultScout and self-evaluation loop`

---

### Task 16: Eval dataset seed (UCL 24/25 + the two named use cases)

**Files:** `fixtures/matches.jsonl` and matching files in `fixtures/{form,injuries,h2h,odds,weather,venues}/`.

- One JSON object per line: `{match_id, competition, kickoff, home_team_id, away_team_id, venue_id, result: {home_goals, away_goals}}`.
- **Phase 1 seeds at least 30 historical matches** with known results — UCL 24/25 (group through final) + WC 2022 + a few qualifiers — so the harness can compute non-degenerate calibration bins.
- Add **playground entries** (no result, so the harness skips them):
  - `{match_id: ucl_2025_final, kickoff: 2026-05-30T20:00:00Z, venue_id: puskas_arena, home: TBD, away: TBD}`
  - `{match_id: wc_2026_final, kickoff: 2026-07-19T20:00:00Z, venue_id: metlife_stadium, home: TBD, away: TBD}`
- For each historical match, write the matching tool fixtures with realistic (not necessarily accurate — labelled as synthetic) numbers.

**Verify:** dataset has ≥30 lines; two named use cases are present and skipped by the harness because they have no result.
**Commit:** `data: eval set v2 (30 historical + 2 playground)`

---

### Task 17: Eval metrics

**Files:** `src/soccer_agent/eval/metrics.py`, `src/soccer_agent/eval/__init__.py`, `tests/test_eval_metrics.py`.

- `accuracy(predictions, results, k=1)`: top-1 (and a draw-tolerance variant: if pick is home/away but actual is draw, treat as half-credit).
- `brier(predicted_probs, actual_onehot)`: summed over 3 classes, normalised to [0,2].
- `calibration_bins(predicted_class_probs, actual_class, n_bins=10)`: reliability table.
- `ece(bins)`: weighted average of `|mean_pred − empirical_freq|`, weights = count.
- `log_loss` for completeness.
- **All metrics reasoner-aware:** harness scores numeric, llm, and final separately.
- **Confidence tracked separately from calibration:** `mean_confidence_correct` vs `mean_confidence_wrong`. A model with high Brier can still have a useful "say no" signal.

**Tests:** known toy dataset → known metrics. ECE=0 for perfectly calibrated toy model.
**Commit:** `feat: eval metrics (accuracy, Brier, ECE, calibration bins)`

---

### Task 18: Eval harness

**Files:** `src/soccer_agent/eval/harness.py`, `tests/test_eval_harness.py`.

- `Harness(agent, db)`.
- `async run(dataset_path=fixtures/matches.jsonl, *, reasoner_filter=None, max_n=None, judge=False) -> EvalRun`:
  1. Stream dataset, for each match **with a known result**: build `Match`, call `agent.predict`, record outputs alongside the actual result.
  2. Skip playground entries (no result). This is what lets us name the UCL 25/26 and WC 26 final in the dataset.
  3. After all predictions, compute metrics, persist an `eval_runs` row, return a summary.
- **`--judge`:** sample 10 predictions, have the LLM reasoner rate each rationale on (a) cited dominant signal, (b) internal consistency, (c) calibration vs. confidence. Average score 0-10 reported.
- **Determinism:** the harness pins `temperature=0` and seeds any randomness; this is the source of truth for "is Phase 2 making the model better?"

**Tests:** tiny in-test dataset (3 matches) → metrics computed; persisted; reasoner filter excludes the other reasoner; `--judge` skipped when no LLM key.
**Commit:** `feat: eval harness with reasoner-aware metrics`

---

### Task 19: FastAPI surface

**Files:** `src/soccer_agent/api/app.py`, `tests/test_api.py`.

- Endpoints:
  - `POST /predict` — body: `Match`. Returns `Prediction` JSON. **200 even on partial tool failure** (warnings in body).
  - `POST /eval` — body: `{path?, reasoner_filter?, max_n?, judge?}`. Returns `EvalRun` summary.
  - `POST /settle` — runs one `ResultScout` tick; useful for triggering self-eval on demand.
  - `GET /predictions?limit=50` — list recent predictions, joined with results and critiques.
  - `GET /matches/{match_id}` — full `Prediction` with all `Signal`s, all `ReasonerOutput`s, and the result + critique if settled.
  - `GET /eval-runs?limit=20` — list past eval runs.
  - `GET /stats` — overall accuracy, brier, calibration bins, per-competition breakdown. (Pre-compute at `/eval`-time and cache; don't recompute per request.)
  - `GET /healthz` — `{ok: true, db: true}`.
- Dependency-injected `agent` and `db` (so tests swap in fakes).
- Lifespan handler initializes DB.

**Tests:** TestClient; each endpoint; partial tool failure → 200 with warnings; 404 on unknown route.
**Commit:** `feat: FastAPI surface`

---

### Task 20: End-to-end smoke + README

**Files:** `tests/test_e2e.py`, `README.md` updates.

- **E2E:** spin up agent, predict a fixture match, persist, score against known result, assert **Brier < 1.0** (loose — proves the pipeline wires up).
- **README:** 5-command quickstart, the agent loop in one paragraph, a "what this does not do yet" section (Phase 2 roadmap: real odds, Elo upgrades, dashboard, the two named use cases).
- A **`scripts/predict_named.py`** stub that the Phase 2 work fills in: `python -m soccer_agent.scripts.predict_named ucl_2025_final` and `wc_2026_final`.

**Verify:** `pytest -q` all green; `soccer-agent --help`; `soccer-agent predict ucl_2024_qf_1` returns JSON.
**Commit:** `docs: README and end-to-end smoke test`

---

## Phase 1 Risks & Tradeoffs

- **No real odds API in Phase 1.** The market signal is the single biggest accuracy lever for soccer. Mitigated with (a) a clean parser interface (one-file swap), and (b) blending numeric + market in the reasoner, so we degrade gracefully rather than catastrophically.
- **LLM cost.** Each predict makes one LLM call. With fixtures and the stub fallback, the eval harness is essentially free.
- **Elo is a poor man's rating.** Good enough as a baseline. Phase 2 replaces it with a better prior.
- **UCL 25/26 and WC26 final are *targets*, not the eval set.** Phase 1 evaluates on 24/25 (and 22/23 to get to 30 matches). The dashboard's live-tracking view is what makes the named use cases visible — that's a Phase 2 deliverable.

## Definition of Done (Phase 1)

- All 20 tasks committed, each with passing tests.
- `pytest -q` green; coverage on `tools/`, `reasoners/`, `agent.py`, `eval/` ≥80%.
- `soccer-agent eval fixtures/matches.jsonl` prints accuracy, Brier, and ECE for both reasoners.
- `soccer-agent serve` starts the API; `curl -X POST localhost:8000/predict -d @sample.json` returns a prediction.
- README has the 5-command quickstart and explicitly names the two target use cases.
- `fixtures/matches.jsonl` contains `ucl_2025_final` and `wc_2026_final` as playground entries (skipped by the harness, ready for Phase 2).
- `settle` command produces self-critiques for at least one settled historical match.

---

## Phase 2 — Accuracy Iteration + Dashboard (sketch, to be planned in detail after Phase 1 lands)

### Task 21: Prompt engineering loop
- Write `prompts/baseline.txt` and 3 variants.
- `scripts/run_eval.py --variants baseline,v2,v3` runs all, diffs metrics, picks winner.

### Task 22: Confidence calibration
- Add explicit "calibrate my confidence" instruction to the predict prompt.
- After eval, fit a temperature/scale on the probability vector to minimize Brier on the eval set. Store the scale factor in the row.

### Task 23: Streamlit dashboard
- `dashboard/app.py` with 3 pages:
  1. **Leaderboard** — KPI cards (total preds, accuracy, brier, ECE), calibration chart, recent matches.
  2. **Match detail** — prediction, rationale (markdown-rendered), tool-output evidence (collapsible JSON), self-eval critique if settled.
  3. **Named-use-cases tracker** — pinned cards for `ucl_2025_final` and `wc_2026_final` with countdown, latest prediction, and (after kickoff) the settled result.

### Task 24: Live data adapters (one-class swaps)
- `form_recent_live.py` → API-Football / football-data.org
- `odds_market_live.py` → the-odds-api.com
- `weather_venue_live.py` → already done in Task 8
- `injury_news_live.py` → newsapi.org or Transfermarkt scrape

### Task 25: Named-use-case demo scripts
- `scripts/predict_ucl_final.py` and `scripts/predict_wc_final.py`.
- Run daily via cron; commit markdown reports to `reports/`.

### Phase 2 acceptance
- Brier < 0.20 on the eval set (random ≈ 0.67).
- Accuracy > 55% (random = 33%).
- Dashboard live with both named use cases tracked end-to-end.
