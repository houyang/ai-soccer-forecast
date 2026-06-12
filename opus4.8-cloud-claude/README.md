# Soccer Prediction Agent

A multi-tool agent that builds a typed dossier for an upcoming match (form, injuries,
head-to-head, weather, venue, bookmaker odds), reasons with a local Ollama model behind
a swappable interface, logs a 1X2 prediction with rationale and confidence, settles
results on demand, and scores itself against the bookmaker via an offline eval harness.

## Setup

    python -m venv .venv && source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

## Commands

    soccer eval --scenario ucl-2025-26 --reasoner fake
    soccer eval --scenario wc-2026-final --reasoner ollama
    soccer predict --match wc-final
    soccer settle
    soccer report

Configuration is read from environment variables (see `.env.example`): data directory,
Ollama host/model/timeout, provider mode (`fixture|http`), and reasoner (`fake|ollama`).
The `fake` reasoner and `fixture` providers are fully offline and require no network or
Ollama; they are the default and what the test suite and CI use.

## FIFA 2026 World Cup group-stage predictor

The `soccer wc` commands pull the real FIFA 2026 World Cup dataset from API-Football
(API-Sports v3), rank every league/club/player/coach/national-team on a 0–100 scale, and
predict the result and final score of all 72 group-stage matches with an independent
Poisson model.

    # one-time: pull live data into the local cache (data/), needs an API key
    export SOCCER_API_FOOTBALL_KEY=your-key-here   # use a local .env; never commit it
    soccer wc fetch

    # everything below is fully offline (reads the cached dataset)
    soccer wc rank --top 15        # league / club / player / coach / team tables
    soccer wc predict              # 72 predicted scorelines

`predict` writes two files to `./perdiction/` (override with `SOCCER_PREDICTION_DIR`):
`worldcup-2026-predictions.json` (full machine-readable output) and
`worldcup-2026-predictions.md` (a human-friendly report grouped by group, then matchday,
ordered by kickoff time).

`fetch` is the only command that touches the network; it caches every API response under
`data/api/`, so it is replayable for free and `rank`/`predict` never need a key. Both the
key and the `data/` cache are git-ignored.

How the rankings feed the prediction (full detail in the design doc below):

- **leagues** ← WC-player count + average attendance + country pedigree
- **clubs** ← league rank + WC players + last-season win rate + titles
- **players** ← club & league rank + (position-adjusted) goals + match rating
- **coaches** ← squad quality they command + recent win rate
- **national teams** ← country pedigree + squad quality + coach + recent form + domestic
  base, with a host bonus; per-match host/home, travel (jet-lag) and weather adjustments
  are then applied inside the Poisson model.

## Quality gate

    make check   # ruff format check + ruff lint + mypy + pytest with coverage

## Architecture

See `docs/architecture.md`, the Phase 1 design at
`docs/superpowers/specs/2026-06-08-soccer-prediction-agent-phase1-design.md`, and the
World Cup predictor design at
`docs/superpowers/specs/2026-06-11-worldcup-2026-group-stage-predictor-design.md`.
