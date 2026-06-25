# WorldCup 2026 Post-Matchday-1 Re-forecast — Design

**Date:** 2026-06-18
**Status:** Approved (design phase)

## Problem

The group-stage predictor (`src/soccer/worldcup/`) produced a full-tournament forecast from a
pre-tournament dataset built 2026-06-11. Matchday 1 is now complete. We want to fold the
**actual MD1 results** and each team's **formation + starting XI / substitutes** into the model
so the remaining group-stage matches (unplayed MD2 + MD3) get sharper predictions, saved to a
new pair of output files.

The cached dataset has no results (all fixtures `NS`) and no lineup data. The live API-Football
endpoint *does*: as of 2026-06-18 it returns finished fixtures with scorelines and
`fixtures/lineups` blocks carrying `formation`, `startXI` (11), and `substitutes` (15).

## Approach (chosen: A — bounded rating/λ nudges)

Keep the existing Poisson-supremacy core. Layer two **bounded, damped** adjustments derived from
each team's played match(es), plus a small formation lean. One match is thin data, so every
adjustment is capped and a team with no played match degrades exactly to the baseline model.

Rejected alternatives: (B) re-rank teams from MD1 alone — one match is far too thin and overfits
blowouts; (C) rebuild team strength purely from the starting XI — discards pedigree/form and
unfairly downgrades teams that rotated in MD1.

## Components

### 1. Data plumbing (incremental live fetch)

- **`entities.py` — new `Lineup`** frozen dataclass: `fixture_id: int`, `team_id: int`,
  `formation: str` (e.g. `"4-3-3"`, `""` when missing), `start_ids: tuple[int, ...]`,
  `sub_ids: tuple[int, ...]`, with `to_dict`/`from_dict`. Add `lineups: tuple[Lineup, ...] = ()`
  to `WorldCup`, threaded through its `to_dict`/`from_dict`.
- **`apifootball.py` — `force_refresh: bool = False`** on `get()` / `_fetch_page()`: skips the
  cache *read* but still *writes* the response. Minimal way to re-pull the stale `fixtures`
  endpoint while keeping replayability for the freshly fetched data.
- **New module `live.py` — `refresh_live(wc, client) -> WorldCup`**: force-refetch
  `fixtures?league=1&season=2026` to fill `home_goals`/`away_goals` on played matches (matched by
  `fixture_id` against the existing dataset, preserving group/venue), then fetch
  `fixtures/lineups?fixture=…` for each finished match and attach `Lineup`s. Static entities
  (teams/players/clubs/coaches) are reused untouched — zero extra quota on the 1,248 players.
- **CLI `wc refresh`**: run `refresh_live`, write the merged dataset back to
  `data/worldcup-2026.json`.

### 2. Adjustment model (new module `adjust.py`)

Per team that has played, compute a bounded rating delta. All constants are named module-level
constants with a one-line rationale, matching `ranking.py`'s style.

- **Momentum (results signal):**
  `momentum = clamp(K_MOM * (actual_margin - expected_margin), ±CAP_MOM)`.
  `expected_margin` is the pre-tournament supremacy for that exact MD1 fixture (team-oriented);
  `actual_margin` is the real goal difference from the team's perspective. Averaged over the
  team's played matches. Defaults `K_MOM = 0.8` rating pts per goal of over/under-performance,
  `CAP_MOM = 4.0`.
- **Lineup (starters signal):**
  `lineup = clamp(K_LU * (xi_quality - squad_core_quality), ±CAP_LU)`, where `xi_quality` is the
  mean ranked player score of the actual 11 starters (most recent played match) and
  `squad_core_quality` is the top-16 mean the base team rating already assumed. Captures whether
  the stars actually started (fitness/availability). Defaults `K_LU = 0.15`, `CAP_LU = 3.0`.
- **Total** per-team rating delta = `clamp(momentum + lineup, ±CAP_TOTAL=5.0)`, folded into the
  effective rating.
- **Formation lean (λ split, not rating):** parse `defenders` = first formation digit, `forwards`
  = last digit. A small attacking nudge to the team's own λ for extra forwards, and a small
  defensive nudge suppressing the opponent's λ for extra defenders. Tiny coefficients; gracefully
  zero when the formation string is missing or unparseable.

`TeamAdjustment` dataclass exposes `rating_delta`, `momentum`, `lineup`, `attack_lean`,
`defense_lean` for transparency and testing. `compute_adjustments(wc, rankings) -> dict[int, TeamAdjustment]`.

### 3. Prediction flow & output

- **`predict.py`** — add `predict_remaining(wc, rankings)`: predicts only *unplayed* matches
  (`not match.played`) using adjusted effective ratings + formation lean; the rationale string
  gains the adjustment breakdown. `MatchPrediction` gains `home_adjustment` / `away_adjustment`
  float fields (default `0.0`, appended last so existing constructors/tests are unaffected). The
  original `predict_group_stage` stays intact as the baseline.
- **CLI** — `wc predict --remaining --out-dir prediction --name worldcup-2026-predictions-after1st-group`.
  Writes the JSON (remaining-game predictions + per-team adjustments) and a Markdown report. The
  report lists completed **actual** results (MD1 and any finished MD2) at the top for context,
  then the updated predictions per group / matchday. Default (no `--remaining`) keeps the original
  full-slate behavior and file names.
- **Final deliverables:** `prediction/worldcup-2026-predictions-after1st-group.json` and
  `prediction/worldcup-2026-predictions-after1st-group.md`.

### 4. Testing (all offline, per CLAUDE.md)

Fakes only — no network, wall-clock, or machine paths. Coverage:

- `entities`: `Lineup` and `WorldCup`-with-lineups JSON round-trip.
- `apifootball`: `force_refresh` skips cache read but still stores (fake transport + cache).
- `live`: `refresh_live` merges results + lineups via a fake client; unplayed matches stay
  unplayed; missing lineups are tolerated.
- `adjust`: momentum from over/under-performance, lineup delta from XI quality, all caps, and the
  no-played-match → zero-adjustment case; formation parse edge cases (empty / malformed).
- `predict`: `predict_remaining` returns only unplayed matches and shifts λ in the expected
  direction for a boosted vs. penalized team; deterministic output.
- `cli`: `wc predict --remaining` writes both files to the requested directory/name.

Then `make check` (ruff format, ruff lint, mypy, pytest + coverage).

## Definition of Done

- New/updated modules: `entities.py`, `apifootball.py`, `live.py`, `adjust.py`, `predict.py`,
  `cli.py`, plus tests for each.
- `data/worldcup-2026.json` refreshed with MD1 results + lineups.
- `prediction/worldcup-2026-predictions-after1st-group.{json,md}` generated.
- `make check` passes; docs/README updated for the new `wc refresh` and `wc predict --remaining`
  commands.
