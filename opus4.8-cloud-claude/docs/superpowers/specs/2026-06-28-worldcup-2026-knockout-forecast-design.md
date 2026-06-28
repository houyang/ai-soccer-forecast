# FIFA 2026 World Cup — Knockout-Stage Forecast

**Date:** 2026-06-28
**Status:** Approved (design)

## Goal

The group stage is complete. Extend the predictor from group-stage scorelines to the **full
knockout bracket** — Round of 32 through the Final, plus the third-place playoff — producing both a
single most-likely bracket (the headline forecast) and Monte-Carlo title/advancement odds. The
existing rating + Poisson + Dixon–Coles scoreline core is reused unchanged; the new work is the
knockout-specific resolution (no draws), the bracket structure, and the simulation.

## Evidence (live API probe, 2026-06-28)

A force-refreshed pull of `fixtures?league=1&season=2026` establishes the data we have to build on:

- **88 fixtures total.** All **72 group matches are `FT` (finished)** — the group stage is genuinely
  complete in the source.
- **16 `Round of 32` fixtures now exist** with concrete teams, dates, and (partial) venues, all
  status `NS`.
- **Nothing beyond R32.** No Round of 16, quarter-final, semi-final, third-place, or final fixtures
  exist in the API yet (they are created only as earlier rounds resolve).
- **The API `standings.rank` field is unreliable.** It labels teams such as "4th in Group I"
  (Norway) as advancing to the R32, which is impossible. Recomputing points from results puts Norway
  2nd in Group I. **Standings must be computed from match goals, not read from the API.**

These three facts drive the design: R32 is live data; R16→Final must be synthesized; standings are
computed locally.

## Decisions (from brainstorming)

1. **Group results + R32 come from the live API** via the existing `fetch`/`refresh` path.
2. **Output is layered**: a single modal bracket as the headline, plus Monte-Carlo advancement/title
   odds on top.
3. **Algorithm change is knockout-only**: add extra-time + penalty resolution. The rating/scoreline
   core (ranking, `BASE_MATCH_GOALS`, supremacy, `DRAW_RHO`, travel/weather) is **not** retuned, and
   no form-based re-rating is added.
4. **Bracket beyond R32 is a hybrid**: live R32 schedule as the leaves; the official FIFA-2026
   bracket tree (hardcoded reference) wires R16→Final; the two are cross-checked.

### Why the hybrid (alternatives rejected)

- **Infer adjacency from R32 dates/venues** — infeasible: the downstream fixtures do not exist in the
  API, so there is nothing to infer *from*, and R32 kickoff order does not encode bracket adjacency
  (most venue cities are `null`). It would be unvalidatable guesswork.
- **Re-derive the whole bracket from standings** (recompute the 8 best thirds + the full FIFA thirds
  allocation table + all pairings) — most code, and risks disagreeing with the actual published
  draw. Unnecessary because the API already gives the resolved R32 pairings.

## Design

### 1. Data model — `worldcup/entities.py`

Add one backward-compatible field to `WcMatch`:

```
round_name: str = ""   # "" for group stage; "Round of 32" etc. for knockout
```

`to_dict`/`from_dict` round-trip it with a `""` default, so existing datasets and the 27→72 group
records load unchanged. No other entity changes; synthesized R16→Final ties are **not** stored in
the dataset (they have no fixture id) — they live in the bracket structure at compute time.

### 2. Ingest fix — `worldcup/ingest.py`

- Remove the `if m.group` filter (line ~284) that currently **discards every knockout fixture**;
  keep all fixtures and populate `round_name` from the API `league.round`.
- `_matchday` keeps returning the group matchday for `"Group Stage - N"`; for knockout rounds it is
  not meaningful, so knockout `WcMatch` rows carry `matchday = 0` and rely on `round_name`.
- Group-stage `WcMatch` rows still set `group`; knockout rows set `group = ""`.

### 3. Standings — `worldcup/standings.py` (new)

Pure function over the played group matches. For each of the 12 groups, build a table with
points (3/1/0), goal difference, goals for, goals against. Order by the **FIFA tiebreaker
sequence**: points → goal difference → goals for → head-to-head (points, then GD, then GF among the
tied teams) → a deterministic final fallback (disciplinary is unavailable, so fall back to a stable
key — team id — standing in for the drawing of lots, documented as such).

Exposes:

- `group_tables(wc) -> dict[str, list[StandingRow]]` (ordered, rank 1–4 per group),
- `team_label(team_id) -> "1A" / "2B" / "3C"` lookups,
- the third-place ranking (used for the report and the cross-check).

The API `rank` field is never read.

### 4. Bracket — `worldcup/bracket.py` (new)

The official FIFA-2026 knockout tree as hardcoded reference data (transcribed from the published
bracket, **verified against a current source at implementation time**):

- The 8 R16 pairings expressed as links between R32 slots, then the 4 QF, 2 SF, Final, and
  third-place links — ~15 links total. Each R32 slot is keyed by its fixed identifying label (the
  group winner or runner-up it contains; the variable third-placed side is matched by the *other*,
  fixed slot).
- **Join**: label each live R32 `WcMatch` by its two teams' `(group, rank)` from §3, match each to a
  bracket R32 slot, and build a `Bracket` of rounds (R32 → R16 → QF → SF → Final + third place). Each
  tie is either concrete teams (R32) or "winner of tie X" / "loser of SF" placeholders.
- **Cross-check**: if a live R32 fixture's label pair cannot be matched to exactly one slot, raise a
  clear `BracketError` naming the offending fixture — never silently mispair.

### 5. Knockout match model — extend `worldcup/predict.py`

Reuse the regulation scoreline matrix (`_scoreline_matrix`, with `DRAW_RHO`). Add knockout
resolution:

- Regulation gives `p_home`, `p_draw`, `p_away` and a modal regulation score (for display).
- Extra time: a second Poisson matrix at a reduced goal rate (λ scaled by `ET_GOAL_FRACTION ≈ 1/3`,
  i.e. 30 added minutes vs 90) supplies `P(win in ET | regulation drawn)`.
- Penalties: residual ties resolve via a shootout probability centred on 0.5 with a small,
  capped rating-gap edge (`PEN_EDGE_PER_10`, clamped to e.g. [0.35, 0.65]).
- `P(home advances) = p_home + p_draw · [P(home wins ET) + P(ET drawn) · P(home wins pens)]`;
  `p_away_advance = 1 − p_home_advance`.

New frozen dataclass `KnockoutPrediction` (own `to_dict`): round name, the two team ids/names,
regulation modal score, W/D/L, `p_home_advance`, `p_away_advance`, and an `expected_extra_time`
flag (true when `p_draw` is the plurality outcome). New module constants:
`ET_GOAL_FRACTION`, `PEN_EDGE_PER_10`, `PEN_EDGE_CAP`.

### 6. Modal bracket + Monte Carlo — `worldcup/simulate.py` (new)

- **Modal bracket**: starting from the concrete R32 ties, predict each, advance the higher
  `p_advance` team, form the next round per the §4 tree, repeat to a champion; the two SF losers
  play the third-place tie. Returns the ordered list of `KnockoutPrediction`s with resolved
  matchups and the predicted podium (champion / runner-up / third).
- **Monte Carlo**: an injected `random.Random` (DI per CLAUDE.md — no global/default RNG, no
  wall-clock seeding). Simulate the full bracket `n_sims` times (default 20000); each realized tie is
  decided by `Bernoulli(p_home_advance)`. Pairwise advancement probabilities are **cached by
  `(home_id, away_id)`** (ratings are static, so ≤ 48·47 distinct ties), keeping the run fast. Tally
  each team's probability of reaching R16/QF/SF/Final and of winning the cup.

### 7. CLI + output — `worldcup/cli.py`

New `soccer wc knockout` subcommand:

- Loads the dataset; if no `Round of 32` fixtures are present, exit with a clear message to run
  `soccer wc fetch`/`refresh` first.
- Computes standings, builds the bracket, runs the modal bracket and Monte Carlo (`--sims`, default
  20000; `--seed`, default fixed for reproducibility), writes `worldcup-2026-knockout.json` and
  `.md` (flags `--out-dir`, `--name` mirror `predict`).
- The Markdown report shows: the full bracket round by round (each tie `Team A x–y Team B
  (adv A 62% / B 38%)`, with an AET/pens note where `expected_extra_time`), the predicted podium,
  and a champion/title-odds table (top teams by title probability with reach-SF / reach-Final odds).

### Scope of change

- New: `worldcup/standings.py`, `worldcup/bracket.py`, `worldcup/simulate.py`.
- Edited: `worldcup/entities.py` (`round_name`), `worldcup/ingest.py` (keep knockout fixtures),
  `worldcup/predict.py` (knockout resolution), `worldcup/cli.py` (`knockout` subcommand).
- No new third-party dependencies; pure functions; no import-time side effects; RNG and the dataset
  path injected at the CLI boundary.

## Testing (offline, injected RNG, no network/clock — per CLAUDE.md)

- **standings**: crafted group results exercise each tiebreaker level (points, GD, GF, head-to-head,
  deterministic fallback); ranks and labels are correct.
- **bracket**: a synthetic R32 set maps onto the tree to the expected R16 pairings; an unmatchable
  fixture raises `BracketError`.
- **predict (knockout)**: advancement probs sum to 1; the stronger team is favoured; a draw-heavy λ
  pair drives ET/penalty mass and sets `expected_extra_time`; equal teams give ≈0.5.
- **simulate**: a fixed dataset yields a deterministic modal champion; a seeded `random.Random`
  makes Monte-Carlo output reproducible; per-team probabilities are in [0,1] and the round-reach
  series is monotone non-increasing; the pre-tournament favourite has the highest title odds.
- **cli**: writes both files; missing-R32 path prints the fetch hint and returns non-zero.

## Out of scope / follow-ups

- Retuning the core scoreline constants or adding form-based re-rating from group results (the user
  scoped improvements to knockout modeling only).
- Storing synthesized R16→Final fixtures in the dataset, or live-refreshing later rounds as FIFA
  publishes them (a future `refresh`-side enhancement once those fixtures appear in the API).
- A PDF bracket card (the existing `card`/`cardpdf` path could be extended later).

## Definition of done

`ruff format` + `ruff check` + `mypy src tests` + full `pytest` pass; the dataset is refreshed to the
complete group stage + live R32; `soccer wc knockout` produces the bracket JSON + Markdown with a
champion, podium, and title-odds table; docs/README updated for the new subcommand; final summary
names the changed files and the validation commands run.
