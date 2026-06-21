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

`predict` writes two files to `./prediction/` (override with `SOCCER_PREDICTION_DIR`):
`worldcup-2026-predictions.json` (full machine-readable output) and
`worldcup-2026-predictions.md` (a human-friendly report grouped by group, then matchday,
ordered by kickoff time).

Once matchdays have been played, refresh and re-forecast the rest of the group stage:

    # pull live results + lineups into the cached dataset (incremental, needs the API key)
    soccer wc refresh

    # re-forecast only the unplayed matches, folding in actual results, starting XIs,
    # and formations; writes a named .json/.md pair to the chosen directory
    soccer wc predict --remaining --out-dir predictions \
        --name worldcup-2026-predictions-after1st-group

`refresh` re-pulls only the `fixtures` and `fixtures/lineups` endpoints (the static
player/club/coach data is reused), so it costs only a handful of API calls. `predict
--remaining` then applies bounded per-team adjustments — momentum from each result versus
the pre-tournament line, a lineup-quality term from who actually started, and a small
formation lean — to every not-yet-played match. The `.json` carries the predictions, the
completed results, and the per-team adjustments; the `.md` lists actual results first, then
the updated predictions per group and matchday.

### `soccer wc card <fixture_id>`

Build a one-match pre-match preview: each team's coach, projected (or confirmed) starting XI
and formation, likely substitutes, and a lineup-aware prediction. Writes a PDF and JSON to the
prediction directory (`card-<fixture_id>.pdf` / `.json`).

```bash
soccer wc card 1320001                 # PDF + JSON from the cached dataset
soccer wc card 1320001 --format json   # JSON only (no reportlab needed)
soccer wc card 1320001 --refresh       # pull the official lineup/result first (needs API key)
```

Lineups are **projected** from each team's most recent earlier-matchday lineup (or, on
Matchday 1, from the coach-preferred formation and top-rated squad players) until the official
XI is published — run with `--refresh` close to kickoff to pick up the **confirmed** lineup.

PDF output needs the optional extra:

```bash
python -m pip install -e ".[pdf]"   # or: pip install 'soccer[pdf]'
```

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
