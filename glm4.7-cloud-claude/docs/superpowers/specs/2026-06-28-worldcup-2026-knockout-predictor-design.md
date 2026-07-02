# FIFA 2026 World Cup — Knockout Predictor + Single-Match Preview Card

**Date:** 2026-06-28
**Project:** `glm4.7-cloud-claude` (the `soccer_agent` package)

## Goal

The FIFA 2026 group stage (the tournament's first round) is complete. Enhance the
`glm4.7` codebase so it can:

1. Produce **more accurate predictions for all remaining games** (the Round of 32 and the
   full knockout bracket through to the champion), using the actual group-stage results as
   the latest signal. Save the batch output to
   `predictions/worldcup-2026-predictions-after1st-group.md` and `.json`.
2. Add a **single-match pre-match prediction** feature that renders a one-page PDF
   containing each side's coach, starting XI with formation, likely subs, and the predicted
   result + scoreline probabilities.

## Scope and constraints

- **Data source:** the cached dataset at
  `opus4.8-cloud-claude/data/worldcup-2026.json`, **copied** into
  `glm4.7/data/worldcup-2026.json`. This file is treated as raw data only.
- **No reference to opus4.8's implementation.** None of opus4.8's `.py` modules are ported
  or consulted. All code in this feature is written originally for `glm4.7`. The
  prediction/ranking/lineup/bracket algorithms are standard domain techniques implemented
  from scratch here.
- **No network, no API key.** `glm4.7` has no `.env`/key. Everything runs offline off the
  copied JSON. The dataset's `lineups` array is empty, so formations and starting XIs are
  **projected** offline (see Lineup projection).
- **Dataset state (verified 2026-06-28):** 48 teams, 1248 players, 48 coaches, 88 matches.
  All 72 group-stage matches carry results (`home_goals`/`away_goals`); the 16 Round-of-32
  fixtures are unplayed but **already carry real team pairings** (`home_id`/`away_id`
  populated, kickoffs from 2026-06-28 onward). No lineups. R16/QF/SF/Final fixtures are not
  in the dataset — they are produced by the bracket simulation.
- The existing `soccer_agent` LangGraph agent scaffolding is left untouched. This feature
  is a new, self-contained subpackage.

## Dataset shape (the only thing reused from opus4.8)

Top-level JSON keys: `leagues`, `clubs`, `players`, `coaches`, `teams`, `matches`,
`lineups` (empty). Field shapes (verified):

- `teams[i]`: `id, name, group, confederation, is_host, player_ids[], coach_id,
  recent_w, recent_d, recent_l`.
- `players[i]`: `id, name, age, position ("Goalkeeper"|"Defender"|"Midfielder"|"Attacker"),
  club_id, goals, rating (0–10 scale), appearances, wc_team_id`.
- `coaches[i]`: `id, name, age, wins, draws, losses, titles, team_id`.
- `matches[i]`: `fixture_id, matchday (1–3 group, 0 = R32), group, home_id, away_id,
  kickoff (ISO), venue, home_goals, away_goals, round_name`.

## Architecture

New subpackage `soccer_agent/worldcup/`. Modules:

| Module | Responsibility |
|---|---|
| `entities.py` | Pydantic models (`Player`, `Coach`, `NationalTeam`, `WcMatch`, `WorldCup`) parsed from the cached JSON. glm4.7 already uses Pydantic, so models are Pydantic v2 `BaseModel`s (not frozen dataclasses). |
| `dataset.py` | `load_worldcup()` — locate `data/worldcup-2026.json` (search repo root / package dir) and parse into `WorldCup`. Pure function, no I/O side effects beyond the read. |
| `ranking.py` | Deterministic 0–100 ratings computed from the dataset. Dependency order: league → club → player → coach → national team, each a min-max-normalized blend of the fields above. Named constant weights with one-line rationales. This is the **static** (pre-knockout) rating. |
| `form.py` | **Accuracy enhancement #1.** Reads the 72 group-stage results, computes each team's actual record (P, W, D, L, GF, GA, GD, pts) and an *attack* and *defense* rating from actual goals scored/conceded (regressed toward the group mean with a small-sample shrinkage). Produces a `TeamForm` per team and a `recalibrated_strength(team_id)` that blends the static ranking with the group-stage goal differential. This is the "latest updates" signal. |
| `lineup.py` | **Accuracy enhancement #2.** Projects each team's formation, starting XI, and 7 subs offline. A curated `FORMATIONS` dict maps team name → real coach formation (e.g. France 4-2-3-1, Norway 4-4-2, Argentina 4-3-3). Starters chosen by position group (1 GK / D-M-F per the parsed formation) using player `rating`; subs = next-best by rating with bench coverage. Carries a `source = "projected"` provenance label. |
| `predict.py` | Poisson scoreline model. Effective rating = recalibrated strength adjusted for host-nation home field, inter-confederation travel penalty, and hot-venue/weather. Rating gap → goal supremacy → split a baseline match total (≈2.6 goals) into λ_home / λ_away with a floor. Independent-Poisson scoreline matrix (0–8 each side) with a Dixon-Coles low-score correction that lifts draw mass. Outputs modal exact score, W/D/L probabilities, and top-3 scorelines. **Lineup-aware (enhancement #3):** the rating used for each side is the projected starting XI's mean rating, not the full-squad mean, so absences/resting are reflected. |
| `standings.py` | Compute each group's final standings from the 72 actual results with FIFA tiebreakers (points → GD → GF → head-to-head). Used for the group-stage **recap** in the output only — R32 pairings are taken directly from the dataset, so this does not seed the bracket. |
| `bracket.py` | Takes the 16 dataset R32 fixtures (real pairings) and defines the R16→QF→SF→Final + 3rd-place slot graph. The dataset has no explicit R16 slot mapping, so R32 matches are paired into R16 by sorted `fixture_id` order (match 1 vs 2, 3 vs 4, …) and advanced through a balanced binary tree. **This pairing is an approximation**, documented in the output; R32 itself is exact. |
| `simulate.py` | Monte-Carlo: predict each R32 match, advance winners (extra-time + shootout modelling for drawn ties — ET inflates λ by ~1/3, shootout win prob shifted by rating edge capped at ±0.15), walk the bracket tree to a champion. Returns champion win probability, per-round advancement odds, and the modal bracket. Default 10 000 iterations. |
| `card.py` | `build_card(home_name, away_name)` → `MatchCard` with both `TeamCard`s (coach name + W-D-L record, formation, starters w/ name+position+rating, subs, source) and the `MatchPrediction` + top scorelines. Used by both the PDF renderer and the CLI. |
| `cardpdf.py` | `render_card_pdf(card, path)` → one-page A4 PDF via `reportlab` (lazy import; `reportlab` is an optional `[pdf]` extra). Two-column layout: home left, away right, each showing coach/formation/starting XI/subs; header with kickoff + venue; top section with prediction, expected goals, top scorelines, rationale. |
| `cli.py` + `__main__.py` | `python -m soccer_agent.worldcup predict` → writes the after1st-group md+json. `… card "Home" "Away"` → writes `<Home>-vs-<Away>.pdf` + `.json`. `… bracket` → prints champion + advancement odds (also embedded in `predict` output). |

### Data flow

```
dataset.load_worldcup()
  → ranking (static 0–100)
  → form.recalibrated_strength (blend with group-stage GD)   # "latest updates"
  → lineup.project (curated formation + XI by rating)
  → predict (Poisson + Dixon-Coles, lineup-aware λ)
  → standings (group recap) + bracket (R32 from dataset, R16+ approximated tree)
  → simulate (Monte-Carlo to champion)
  → outputs: predictions/*.md+json  |  predictions/<match>.pdf+json
```

## Outputs

### `predictions/worldcup-2026-predictions-after1st-group.md` + `.json`
- Header: generation note + method summary.
- Group-stage recap: final standings per group (P W D L GF GA GD Pts) computed from actual
  results.
- Round of 32: each of the 16 matches — kickoff, venue, prediction (score + W/D/L %),
  expected goals, top scorelines, one-line rationale citing recalibrated strength.
- Bracket simulation: champion win probability (top teams), per-round advancement odds for
  the field, and the modal (most frequent) bracket path.
- The `.json` mirrors the `.md` with structured fields (`standings`, `r32`,
  `bracket{champion, advancement, modal_path}`).

### `predictions/<Home>-vs-<Away>.pdf` + `.json`
One-page PDF: match header (teams, group/round, kickoff, venue); prediction line (score +
W/D/L %, expected goals, top scorelines, rationale); two columns each with coach (name +
W-D-L), formation, starting XI (position + name + rating), likely subs. The `.json` is the
`MatchCard.to_dict()`.

## Accuracy enhancements (the "more accurate" requirement)

1. **Group-stage recalibration** — actual WC2026 goals/results move each team's strength off
   the pre-tournament static rating, so a team that over/under-performed in the group is
   re-weighted for the knockout.
2. **Curated real formations** — a per-team formation table (not a uniform 4-3-3), so the
   projected shape matches the coach's real system.
3. **Lineup-aware λ** — expected goals driven by the projected starting XI's mean rating,
   not the full 26-man squad.
4. **Dixon-Coles low-score correction** — inflates 0-0/1-1 cells, trimming 1-0/0-1, lifting
   draw probability toward the observed rate (a known fix for independent-Poisson).

## Dependencies

- Add `reportlab` as an optional extra in `pyproject.toml`: `[project.optional-dependencies]
  pdf = ["reportlab>=4.0"]`. The `worldcup` package imports it lazily only when a PDF is
  requested, so `predict`/`bracket` work without it.
- The `worldcup` package itself depends only on the stdlib + `pydantic` (already a
  dependency). No new runtime deps besides the optional PDF one.

## Testing

New `tests/test_worldcup_*.py` using the copied dataset as a fixture:
- `ranking`: deterministic (same input → same output), teams rated 0–100, hosts ≥ non-hosts
  of equal squad.
- `form`: recalibration moves a team's strength when its group GD is non-zero; shrinkage
  toward mean for small samples.
- `lineup`: every team projects exactly 11 starters + 7 subs; starter position counts match
  the curated formation's slots; GK is always a goalkeeper.
- `predict`: W+D+L probabilities sum to 1.0 (±1e-9); λ ≥ floor; modal score is the argmax
  of the matrix.
- `standings`: group ordering obeys FIFA tiebreakers on a hand-checked group; 12 groups
  produced, each with 4 teams.
- `bracket`/`simulate`: the 16 dataset R32 matches are taken as-is; the bracket tree
  produces 8 R16 + 4 QF + 2 SF + 1 Final slots; simulation yields exactly one champion with
  probability mass summing to 1.0 across all 48 teams.
- `card`: `build_card` returns both teams with a coach, 11 starters, 7 subs, and a
  prediction; `to_dict` is JSON-serializable.

## Out of scope

- Live data fetching / API integration (no key available).
- Re-predicting the 72 already-played group matches (retroactive accuracy) — only the
  knockout is predicted.
- Modifying the existing LangGraph `soccer_agent` agent or its tools.
