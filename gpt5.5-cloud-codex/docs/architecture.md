# Prediction Agent Architecture

## Goals

The agent is designed to answer one question for an upcoming soccer match:

> Based on current evidence, what is the most likely result, why, and how confident should
> we be?

The initial scope is architecture, tool contracts, and evaluation. The current implementation
uses local fixture-backed tools so tests stay deterministic and no network calls or credentials
are required.

## Tool Design

Each external capability is represented by a small protocol:

- `FormTool`: recent wins, draws, losses, goals for, and goals against.
- `InjuryNewsTool`: unavailable or doubtful players plus source labels.
- `HeadToHeadTool`: previous meetings between the two teams.
- `VenueTool`: stadium, city, country, and home team context.
- `WeatherTool`: match-day conditions for the venue.
- `OddsTool`: bookmaker market prices and implied probabilities.

Real providers can be added by implementing these protocols. The agent does not know whether
data came from fixtures, a paid API, a scraper, or an internal warehouse.

## Agent Flow

1. Accept a `MatchRequest`.
2. Pull evidence from every configured tool.
3. Build a `MatchEvidence` bundle.
4. Ask the reasoner for a `Prediction`.
5. Persist a `PredictionRecord` to the configured log.
6. After the match is settled, append a result and run self-evaluation.

The agent does not literally sleep until a result arrives. Instead, it logs a pending prediction
and exposes `record_result` as the boundary where a scheduler, webhook, or operator can provide
the final score.

For tournament workflows, `predict_many` accepts a sequence of match requests and records each
prediction. The CLI exposes this through `predict-competition`, backed by competition IDs in the
fixture catalog.

World Cup 2026 score modeling uses a separate data path:

1. `soccer.api_football.fetch_world_cup_2026_snapshot` fetches API-Football JSON and stores it
   locally without translation.
2. `soccer.world_cup_2026.load_world_cup_dataset` normalizes local snapshots into player,
   coach, club, league, national-team, and group-stage match profiles.
3. `rank_world_cup_entities` computes 0-100 rankings for every loaded entity type.
4. `predict_group_stage_scores` applies match-location adjustments and predicts final scores
   for first-round group stage matches.

The World Cup path is intentionally snapshot-first so API keys, provider rate limits, and
network availability do not affect repeatable prediction runs.

## Evaluation Harness

The harness reads logged records with final results and computes:

- settled prediction count
- exact outcome accuracy
- average confidence
- Brier score for the predicted outcome probability

This is intentionally simple and auditable. Future versions can add calibration curves,
competition-specific slices, closing-line value, and model-vs-bookmaker comparisons.

## Immediate Use Cases

The fixture catalog includes single-match scenarios and competition IDs:

- `ucl-final-2026`: UEFA Champions League 2025/26 final placeholder scenario.
- `world-cup-final-2026`: FIFA World Cup 2026 final placeholder scenario.
- `world-cup-2026`: a batch catalog for World Cup workflow testing.
- `demo-finals`: both final placeholder scenarios.

Those entries are architectural examples, not real forecasts. Real predictions require provider
implementations that can retrieve current fixtures, teams, injuries, odds, weather, and results.
For FIFA 2026 group-stage score forecasts, use `fetch-world-cup-data` to create the local
API-Football snapshot and `predict-world-cup-group-stage` to run the ranking model.
