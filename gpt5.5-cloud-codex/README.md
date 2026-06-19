# Soccer Forecast

Soccer Forecast is a Python template for a multi-tool prediction agent. Given an upcoming
match, the agent gathers recent form, injuries, head-to-head history, venue and weather,
and bookmaker odds. It then produces a prediction, logs it, and can self-evaluate after
the match result is available.

The first implementation focuses on architecture, tool design, and evaluation. Prediction
accuracy and a polished dashboard are planned as later layers on top of the same interfaces.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
make check
```

Run a fixture-backed prediction:

```bash
soccer-forecast predict ucl-final-2026 --log predictions.jsonl
soccer-forecast evaluate predictions.jsonl
```

Forecast the current World Cup 2026 fixture catalog:

```bash
soccer-forecast list-matches world-cup-2026
soccer-forecast predict-competition world-cup-2026 --log predictions.jsonl
```

Fetch and model the FIFA 2026 World Cup group stage with API-Football snapshots:

```bash
export API_FOOTBALL_KEY="..."
soccer-forecast fetch-world-cup-data
soccer-forecast fetch-world-cup-match-updates --completed-round-limit 1
soccer-forecast predict-world-cup-group-stage
```

`fetch-world-cup-data` stores raw provider JSON under
`data/api-football/world-cup-2026/` by default. `predict-world-cup-group-stage` then
loads those local snapshots, builds player, coach, club, league, and national-team
rankings on a 0-100 scale, and predicts final scores for the first-round group stage.
Use `--output json` to emit machine-readable score predictions, or `--output markdown`
to print one human-friendly result table per group. Use
`--completed-round-limit 1 --remaining-only` after the first group-stage round to apply
first-round results, formations, starting XIs, and substitutions while outputting only
the remaining group matches. Re-running
`fetch-world-cup-data` resumes from existing snapshot files; add
`--request-delay-seconds 0.5` if the provider starts rate limiting requests.

Optional model inputs that API-Football does not consistently provide, such as league
average attendance and club major-title counts, can be added in
`data/api-football/world-cup-2026/external_factors.json`. See
`docs/world-cup-2026.md` for the supported keys.

Attach completed scores in batches:

```bash
soccer-forecast record-results results.json --log predictions.jsonl
soccer-forecast evaluate predictions.jsonl
```

`results.json` should be a JSON list:

```json
[
  {
    "match_id": "wc-2026-match-001",
    "home_score": 2,
    "away_score": 0,
    "completed_at": "2026-06-12T03:00:00+00:00"
  }
]
```

## Architecture

- `soccer.tools` defines narrow protocols for each information source.
- `soccer.fixture_tools` provides deterministic local tool implementations for tests and demos.
- `soccer.agent.PredictionAgent` orchestrates tool calls and writes prediction records.
- `soccer.reasoning.MatchupReasoner` turns gathered evidence into a prediction.
- `soccer.storage.JsonlPredictionLog` persists predictions and completed results.
- `soccer.evaluation.EvaluationHarness` scores settled predictions.
- `soccer.api_football` fetches API-Football snapshots for World Cup data.
- `soccer.world_cup_2026` normalizes snapshots, ranks entities, and predicts scores.

See `docs/architecture.md` for the current design and extension points.
