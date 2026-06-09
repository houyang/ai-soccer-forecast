# Soccer Prediction Agent — Phase 1 Design

**Date:** 2026-06-08
**Status:** Approved for planning
**Scope:** Phase 1 only — agent architecture, tool design, and eval harness.
Phase 2 (prediction-accuracy tuning, polished UI/dashboard) is out of scope here and
gets its own spec later.

## Mission

Build a multi-tool agent that, given an upcoming match, autonomously gathers recent form,
injury news, head-to-head history, weather, venue, and bookmaker odds; reasons about the
matchup with a local LLM; and outputs a 1X2 prediction with probabilities, a written
rationale, and a confidence score. Predictions are logged; a separate on-demand `settle`
step fetches finished results and self-evaluates. An offline eval harness scores the agent
against known results and against the bookmaker baseline.

Two driving use cases, shipped as fixture scenarios: **UEFA Champions League 2025/26** and
the **FIFA World Cup 2026 final**.

## Decisions (locked during brainstorming)

1. **Pluggable adapters** — one `Protocol` per tool, with fixture (offline) and HTTP (real)
   implementations swapped at a registry boundary. The agent never knows which is active.
2. **Local Ollama reasoner** behind a swappable `Reasoner` interface; default model
   `gemma4:12b-mlx` via `http://localhost:11434`. A deterministic fake reasoner is the
   default for tests and the offline harness.
3. **On-demand `settle`** — no background process or daemon. Invoked manually or by an
   external scheduler.
4. **Orchestration = Approach C (hybrid)** — a deterministic pipeline builds a typed
   dossier and calls the reasoner once. Tools also live in a registry behind a uniform
   `Tool` view, leaving the door open for a future model-driven tool-selection loop without
   rework.
5. **Prediction content** — 1X2 (Home/Draw/Away) pick + normalized probabilities +
   confidence (0–1) + written rationale. No scoreline, no extra markets in Phase 1.

These choices follow the repository's `../AGENTS.md`: dependency injection for all side
effects, deterministic network-free tests, simple/explicit code, `src/` layout, no invented
endpoints, no background processes.

## Architecture overview

The agent is a `PredictionAgent` that runs a deterministic pipeline over a tool registry,
assembles a typed `MatchDossier`, then calls a `Reasoner` once to produce the prediction.
Three on-demand operations:

- **predict** — build dossier → reason → log a `Prediction`.
- **settle** — for logged predictions, fetch finished results → score + self-critique → log
  an `Evaluation`.
- **eval** — run the agent over fixture scenarios with known results → aggregate metrics
  report, including the bookmaker baseline.

### Data flow

```
predict:  CLI → AppConfig → ToolRegistry(fixture|http) → PredictionAgent
              → build_dossier(match) → reasoner.predict(dossier) → Prediction → store
settle:   CLI → store.pending() → ResultProvider.get_result(match)
              → if finished: score + reasoner.self_evaluate(pred, result) → Evaluation → store
eval:     harness → Scenario(matches + known results) → agent.predict each
              → metrics + bookmaker-implied baseline → EvalReport
```

Two invariants serving Phase-1 testability:
- Providers and the reasoner are swapped only at the registry/config boundary; fixtures for
  tests and the harness, HTTP for real runs.
- Tool failures degrade gracefully — `build_dossier` catches `ToolError` per tool, records
  the missing capability in `dossier.missing`, and continues. One dead provider never kills a
  prediction; the reasoner is told what is missing.

## Module layout (`src/soccer/`)

```text
models.py            # all dataclasses + to_dict/from_dict serialization
config.py            # AppConfig — env reads only at this boundary
tools/
  base.py            # Tool Protocol, ToolError, uniform Tool view types
  form.py            # FormProvider Protocol → Fixture + Http adapters
  injuries.py        # InjuryProvider …
  head_to_head.py    # H2HProvider …
  weather.py         # WeatherProvider …
  venue.py           # VenueProvider …
  odds.py            # OddsProvider …
  results.py         # ResultProvider (used by settle) …
registry.py          # ToolRegistry: capability → provider instance + uniform Tool view
dossier.py           # build_dossier(match, registry) → MatchDossier  (the pipeline)
reasoning/
  base.py            # Reasoner Protocol + ReasonResult
  ollama.py          # OllamaReasoner (HTTP localhost:11434, JSON mode, temp 0, seed)
  fake.py            # DeterministicReasoner (tests + offline harness default)
  prompt.py          # dossier → prompt rendering
agent.py             # PredictionAgent.predict(match) → Prediction
store.py             # PredictionStore: append/load JSONL (predictions/results/evaluations)
settle.py            # settle(store, registry, reasoner) → list[Evaluation]
evaluation.py        # metrics: Brier, log-loss, accuracy, calibration, beat_market, score()
harness.py           # Scenario, EvalReport, run_scenario → metrics + market baseline
cli.py               # argparse: predict | settle | eval | report
__main__.py          # python -m soccer …
```

`tests/` mirrors this structure.

## Data model (`models.py`)

Frozen, fully type-annotated dataclasses. Dossier pieces carry `as_of` and `source` for
provenance. `Outcome` is a 3-class enum (`HOME | DRAW | AWAY`).

```python
MatchRef        # id, competition, home, away, kickoff (UTC), venue_id, season
TeamForm        # team, last_n: list[MatchOutcome], gf, ga, points, streak, as_of, source
InjuryReport    # team, out: list[PlayerStatus], doubtful: list[PlayerStatus], as_of, source
H2HRecord       # home, away, meetings: list[PastMeeting], home_wins, draws, away_wins, source
WeatherReport   # venue_id, kickoff, temp_c, wind_kph, precip_mm, condition, source
VenueInfo       # venue_id, name, city, surface, capacity, altitude_m, home_advantage_hint
OddsSnapshot    # bookmaker, home, draw, away (decimal), implied_probs (normalized), as_of
MatchDossier    # match, form{home,away}, injuries{home,away}, h2h, weather, venue, odds,
                #   missing: list[str]   ← capabilities that failed/returned nothing

Prediction      # id, match_ref, created_at, probs{HOME,DRAW,AWAY}, pick, confidence(0-1),
                #   rationale, market_probs, dossier_digest, reasoner_name
MatchResult     # match_id, home_goals, away_goals, outcome, status, source
Evaluation      # prediction_id, result, correct, brier, log_loss, beat_market,
                #   self_critique, evaluated_at
```

Probabilities are always normalized to sum to 1.0; a validator enforces this on
construction. `id` is a stable hash of match id + `created_at`. Serialization
(`to_dict`/`from_dict`) is plain stdlib — no framework.

## Tool interfaces (`tools/`)

Each capability is a single-method `Protocol`. Methods take match/`as_of` context so
fixtures stay deterministic and HTTP adapters can request point-in-time data.

```python
class FormProvider(Protocol):
    def get_form(self, team: str, as_of: datetime) -> TeamForm: ...
class InjuryProvider(Protocol):
    def get_injuries(self, team: str, as_of: datetime) -> InjuryReport: ...
class H2HProvider(Protocol):
    def get_h2h(self, home: str, away: str) -> H2HRecord: ...
class WeatherProvider(Protocol):
    def get_weather(self, venue_id: str, kickoff: datetime) -> WeatherReport: ...
class VenueProvider(Protocol):
    def get_venue(self, venue_id: str) -> VenueInfo: ...
class OddsProvider(Protocol):
    def get_odds(self, match: MatchRef) -> OddsSnapshot: ...
class ResultProvider(Protocol):
    def get_result(self, match: MatchRef) -> MatchResult | None: ...   # None = not finished
```

Two implementations per Protocol:
- **`Fixture*Provider`** — reads hand-authored JSON fixtures (UCL 2025/26 and WC-2026-final
  scenarios live here). Deterministic, offline; the test and harness default.
- **`Http*Provider`** — real adapter. Phase 1 ships these as thin, config-gated stubs with
  the request/parse wiring sketched and a clear `NotImplementedError` until a concrete API is
  chosen (AGENTS.md: do not invent endpoints). Filling in a real endpoint must not change any
  caller.

`ToolError` is the single exception type providers raise. The `ToolRegistry` maps capability
→ instance and exposes each as a uniform `Tool` (name, description, callable) — the view a
future Approach-B agentic loop would consume; the Phase-1 pipeline calls the typed methods
directly.

## Reasoner (`reasoning/`)

```python
class Reasoner(Protocol):
    def predict(self, dossier: MatchDossier) -> ReasonResult: ...
    def self_evaluate(self, prediction: Prediction, result: MatchResult) -> str: ...

@dataclass(frozen=True)
class ReasonResult:
    probs: dict[Outcome, float]   # validated, normalized to 1.0
    confidence: float             # 0-1
    rationale: str
```

- **`OllamaReasoner`** — POSTs to `http://localhost:11434/api/chat` with `format: "json"`,
  `temperature: 0`, fixed `seed`, model from `AppConfig` (default `gemma4:12b-mlx`).
  `prompt.py` renders the dossier into a structured prompt requesting a strict JSON object.
  The adapter parses, validates and renormalizes probabilities, clamps confidence, and raises
  `ReasonerError` on unparseable/garbage output — it never trusts the model blindly.
  Host/model/timeout come from config. `self_evaluate` sends prediction + actual result and
  asks for a short written critique.
- **`DeterministicReasoner`** (fake) — derives probabilities from the dossier by blending
  market-implied odds with a fixed form/H2H adjustment. Default for tests and the offline
  harness, so neither needs Ollama running; also serves as a sanity baseline.

The agent depends only on the `Reasoner` protocol; the concrete reasoner is selected at the
config boundary.

## Persistence (`store.py`)

Append-only JSONL under a configurable data dir (default `./data/`):

```text
data/predictions.jsonl   data/results.jsonl   data/evaluations.jsonl
```

`PredictionStore` takes the three paths injected (tests use `tmp_path`). Methods:
`append_prediction`, `append_result`, `append_evaluation`, `load_predictions`, and
`pending()` (predictions lacking an evaluation). JSONL over SQLite for Phase 1:
human-readable, diff-able, trivial to test, and it directly feeds the Phase-2 dashboard.

## Settle + self-evaluate (`settle.py`)

```
settle(store, registry, reasoner):
    for pred in store.pending():
        result = registry.results.get_result(pred.match_ref)
        if result is None:        # not finished → skip, retry next run
            continue
        critique = reasoner.self_evaluate(pred, result)
        ev = score(pred, result, critique)   # builds the frozen Evaluation in one step
        store.append_result(result); store.append_evaluation(ev)
```

`score(pred, result, critique)` (in `evaluation.py`) computes per-prediction Brier score,
log-loss, correctness, and `beat_market` (did our probs assign higher likelihood to the
actual outcome than the bookmaker-implied probs?), and returns a fully-populated frozen
`Evaluation`. The critique is the reasoner's written reflection on misses, passed in so the
`Evaluation` stays immutable.
With the fake reasoner + fixture results, `settle` is unit-testable end to end.

## Eval harness (`harness.py`)

```python
@dataclass(frozen=True)
class Scenario:
    name: str                        # "ucl-2025-26", "wc-2026-final"
    registry: ToolRegistry           # fixture providers
    matches: list[MatchRef]
    results: dict[str, MatchResult]  # match_id → actual outcome

@dataclass(frozen=True)
class EvalReport:
    scenario: str
    n: int
    accuracy: float
    mean_brier: float                # lower is better
    mean_log_loss: float
    calibration: list[CalibrationBin]   # predicted-prob band vs observed frequency
    market_baseline: MarketBaseline      # same metrics for bookmaker-implied probs
    edge_vs_market: float                # our log_loss − market log_loss (negative = better)
    per_match: list[MatchScore]
```

`run_scenario(scenario, agent)` predicts every match, scores against known results, and
aggregates. The headline question — *does the agent beat the bookmaker?* — is answered by
`edge_vs_market`. Default reasoner is the deterministic fake (CI-safe, no Ollama); point it
at `OllamaReasoner` for a real run. Two built-in scenarios ship as fixtures: **UCL 2025/26**
(a slate of matches) and the **WC 2026 final** (single match).

## CLI (`cli.py` / `python -m soccer`)

```text
soccer predict --match <id> [--providers fixture|http] [--reasoner fake|ollama]
soccer settle  [--providers fixture|http] [--reasoner fake|ollama]
soccer eval    --scenario ucl-2025-26|wc-2026-final|all [--reasoner fake|ollama]
soccer report                      # text summary of logged predictions + running accuracy
```

`argparse`, one subcommand per operation. `report` prints a plain-text table from the JSONL
log — the seam the Phase-2 dashboard will read from the same data. All side effects (data
dir, Ollama host/model, provider mode) resolve through `AppConfig` at this boundary only.

## Error handling

- `ToolError` — provider failure; caught per tool in `build_dossier`, recorded in
  `dossier.missing`, never fatal.
- `ReasonerError` — unparseable/invalid model output; raised by `OllamaReasoner` after
  validation fails. Fatal for a single `predict` call (no silent fallback to garbage).
- Specific exceptions with actionable messages; no broad `except Exception` without re-raise
  or deliberate translation. Programmer errors fail fast.

## Configuration (`config.py`)

`AppConfig` dataclass read once at the CLI boundary. Fields: `data_dir`, `ollama_host`,
`ollama_model` (default `gemma4:12b-mlx`), `ollama_timeout`, `provider_mode`
(`fixture|http`), `reasoner` (`fake|ollama`). Documented in `.env.example`; no import-time
env reads.

## Testing strategy (`tests/` mirrors `src/`)

All deterministic via fixture providers + fake reasoner — no network, no Ollama, no
wall-clock dependence (timestamps injected):

- **models** — serialization round-trip; probability normalization/validation edge cases.
- **tools** — each fixture provider; `ToolError` → recorded in `dossier.missing`.
- **dossier** — full assembly + graceful degradation when a provider raises.
- **agent** — `predict()` returns a valid normalized `Prediction`.
- **store** — JSONL append/load round-trip; `pending()` filtering (`tmp_path`).
- **settle** — scoring math (Brier/log-loss/beat_market) + skip-when-unfinished.
- **evaluation/harness** — metric math on hand-checked numbers; `edge_vs_market`;
  calibration binning.
- **ollama** — adapter parses/validates a mocked HTTP response; bad JSON → `ReasonerError`.
- **cli** — each subcommand against a temp data dir.

Coverage targets the branches AGENTS.md cares about: error paths, missing-data handling, and
probability edge cases.

## Dependencies

- Runtime: standard library only where practical. HTTP via `httpx` (or stdlib `urllib`) for
  the Ollama adapter and future HTTP providers — chosen at planning time; keep runtime deps
  minimal per AGENTS.md.
- Dev: `ruff`, `mypy`, `pytest`, `pytest-cov`, `pre-commit`.

## Out of scope (Phase 2)

- Prediction-accuracy tuning (model/prompt iteration, ensembling, calibration fitting).
- Real `Http*Provider` endpoint implementations against chosen APIs.
- Polished web UI/dashboard (reads the same JSONL the `report` command summarizes).
- Optional Approach-B model-driven tool-selection loop.
