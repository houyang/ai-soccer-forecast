# Single-Match Pre-Match Preview Card (`wc card`)

**Date:** 2026-06-19
**Status:** Approved (design)

## Goal

Add a feature that previews a single upcoming World Cup match before kickoff and produces a
**PDF** card containing, for both teams: the coach, the starting XI with formation, the likely
substitutes, and a lineup-aware (more accurate) prediction result. The preview uses the latest
information available at the time it is run.

## Scope

- New CLI subcommand `soccer wc card <fixture_id>`.
- A lineup-aware single-match prediction that improves on the baseline group-stage forecast by
  folding in tournament momentum and the specific (confirmed or projected) starting XI and
  formation of the upcoming match.
- PDF output via `reportlab`, exposed behind a new optional dependency extra `[pdf]`.
- JSON output of the same card model for inspection/testing.

Out of scope (this phase): odds integration, multi-match batch PDFs, any new networked entity
ingest beyond a single fixture's lineup/result refresh.

### Next phase: knockout-stage matches

The same `wc card <fixture_id>` flow must also work for knockout-stage matches (Round of 32 →
Final) in a later phase. To keep that cheap, this phase is built fixture-id-first and
stage-agnostic wherever possible:

- The command keys off a `fixture_id`, not a group, so any ingested fixture is addressable.
- The projection, card model, and Poisson prediction read no group-only assumptions; `group` is
  carried as a display label, not prediction input.
- Deferred to the knockout phase: ingesting knockout fixtures into the dataset, and modelling
  draw-breakers (extra time / penalties) so the card can show a qualified side rather than only a
  90-minute scoreline. The `MatchCard`/PDF layout should leave room for that without a rewrite.

## Decisions (from brainstorming)

1. **PDF engine:** `reportlab` as an optional extra `[pdf]`. Pure-pip, no system libraries,
   deterministic and offline. Imported lazily inside the renderer so the rest of the package
   imports without it.
2. **Lineup source precedence** when building a card:
   1. `confirmed` — an official lineup for *this* fixture already present in `wc.lineups`.
   2. `prior` — the team's most recent lineup from an earlier matchday in this tournament
      (formation + XI + subs). Best predictor of the next XI ("previous 2026 match").
   3. `projected` — no prior lineup (e.g. Matchday 1): use the coach-preferred formation (the
      most common formation across any available prior lineups for that team, else `4-3-3`),
      fill the formation's GK/DEF/MID/FWD slots with the top-rated squad players by position,
      and take the next highest-rated players as likely subs.
3. **Command shape:** dedicated `wc card <fixture_id>` (not a flag on `wc predict`).

## Architecture

Each concern is an isolated module reusing the existing Poisson model and adjustment math.

### 1. Lineup projection — `src/soccer/worldcup/lineup.py` (new)

```python
@dataclass(frozen=True)
class ProjectedLineup:
    team_id: int
    formation: str
    start_ids: tuple[int, ...]
    sub_ids: tuple[int, ...]
    source: str  # "confirmed" | "prior" | "projected"
    source_matchday: int | None  # matchday the "prior" lineup came from, else None

def project_lineup(wc: WorldCup, rankings: Rankings, team_id: int, fixture_id: int) -> ProjectedLineup: ...
```

- **confirmed:** a `Lineup` exists in `wc.lineups` with `fixture_id == fixture_id` and
  `team_id == team_id` → wrap it, `source="confirmed"`.
- **prior:** among `wc.lineups` for this team, pick the one whose match has the highest
  matchday strictly earlier than the target fixture's matchday → reuse its formation/XI/subs,
  `source="prior"`, `source_matchday` set.
- **projected:** derive `preferred_formation` (most common formation across the team's prior
  lineups, else `"4-3-3"`); parse it into position-group counts; from the squad
  (`wc.squad(team_id)`) sorted by `rankings.players`, fill GK(1)/DEF/MID/FWD slots by player
  `position`, backfilling from best-remaining if a group is short; the next ~7 by rating become
  `sub_ids`. `source="projected"`.

Helpers: `preferred_formation(wc, team_id) -> str`, and a position-bucketing utility that maps
API-Football position strings (`"Goalkeeper"/"Defender"/"Midfielder"/"Attacker"` or `G/D/M/F`)
to the four groups.

### 2. Lineup-aware prediction — extend `adjust.py` and `predict.py`

`adjust.py`:
```python
def adjustment_for_match(wc, rankings, team_id, lineup: ProjectedLineup | Lineup | None) -> TeamAdjustment: ...
```
Reuses the existing `_momentum` (from the team's played matches), `_lineup_delta` (driven by
the supplied XI), and `_formation_lean` (from the supplied formation). With `lineup=None` and no
played matches it returns the zero adjustment, so prediction degrades to the pre-tournament
baseline. `ProjectedLineup` and `Lineup` share the `start_ids`/`formation` attributes the helpers
read, so the existing helpers work unchanged.

`predict.py`:
```python
def predict_one(wc, rankings, fixture_id, home_lineup, away_lineup) -> MatchPrediction: ...
```
Builds `{home_id: adjustment_for_match(...), away_id: adjustment_for_match(...)}` and delegates
to the existing private `_predict(wc, rankings, match, adjustments)`. No change to the Poisson
core. Also expose a small `top_scorelines(lambda_home, lambda_away, n=3)` helper (built from the
existing scoreline matrix) for the PDF's "most likely scorelines" panel.

### 3. Card model — `src/soccer/worldcup/card.py` (new)

```python
@dataclass(frozen=True)
class TeamCard:
    team_id: int
    name: str
    coach_name: str | None
    coach_record: tuple[int, int, int] | None  # W-D-L
    formation: str
    starters: tuple[PlayerLine, ...]   # position, name, rating
    subs: tuple[PlayerLine, ...]
    source: str
    source_matchday: int | None

@dataclass(frozen=True)
class MatchCard:
    fixture_id: int
    group: str
    kickoff: datetime
    venue: str
    home: TeamCard
    away: TeamCard
    prediction: MatchPrediction
    top_scorelines: tuple[tuple[int, int, float], ...]  # (home, away, prob)
```

`build_card(wc, rankings, fixture_id) -> MatchCard` orchestrates: `project_lineup` for each side
→ `predict_one` → assemble. `PlayerLine`/`TeamCard`/`MatchCard` provide `to_dict` for the JSON
output. Pure and offline.

### 4. PDF rendering — `src/soccer/worldcup/cardpdf.py` (new)

```python
def render_card_pdf(card: MatchCard, path: Path) -> None: ...
```
Imports `reportlab` lazily; raises a clear error pointing to `pip install 'soccer[pdf]'` if it is
missing. Layout:
- Header: `Home vs Away`, group, kickoff (UTC), venue, and a confirmed/projected badge per side.
- Two team blocks: coach name + W-D-L record, formation, starting XI (position · name · rating),
  likely substitutes.
- Prediction panel: predicted scoreline, W/D/L %, expected goals (λ home/away), top-3 most likely
  scorelines, and the rationale + adjustment breakdown.

No I/O beyond writing the passed-in `path`.

### 5. Single-fixture refresh — extend `live.py`

```python
def refresh_fixture(wc: WorldCup, client, fixture_id: int) -> WorldCup: ...
```
Fetches the latest `fixtures?id=<fixture_id>` (to fill a result if finished) and
`fixtures/lineups?fixture=<fixture_id>` (confirmed lineup once published), merging into the
dataset's `matches`/`lineups`. Reuses the existing `_parse_lineups`/`_apply_results` helpers.
Costs at most two API calls. Used only by `wc card --refresh`.

### 6. CLI — `wc card` in `cli.py`

```
soccer wc card <fixture_id> [--refresh] [--out-dir DIR] [--name NAME] [--format {pdf,json,both}]
```
- Loads the cached dataset (offline default).
- `--refresh`: requires `SOCCER_API_FOOTBALL_KEY`; calls `refresh_fixture` and persists the
  dataset before building the card.
- Builds the card, writes `card-<fixture_id>.json` and/or `.pdf` (default `both`) into the output
  directory (default `config.prediction_dir`), and prints a short summary to stdout.

## Data flow

```
wc card 42 [--refresh]
  └─(optional) refresh_fixture(client, 42) → persist dataset
  └─ build_card(wc, rankings, 42)
       ├─ project_lineup(home) ─┐
       ├─ project_lineup(away) ─┤
       └─ predict_one(...) ─────┘→ MatchCard
  └─ render_card_pdf(card, card-42.pdf)  and/or  card-42.json
```

## Error handling

- Unknown `fixture_id` → clear CLI error listing how to find fixtures (`wc predict`).
- `--refresh` without an API key → same message pattern as `cmd_refresh`.
- Missing `reportlab` when a PDF is requested → actionable error naming the `[pdf]` extra; JSON
  output still works without it.
- A team with no squad data → projection still returns a (possibly short) XI; the card notes the
  thin data rather than crashing.

## Testing (offline, injected IO, tmp dirs)

- `lineup.py`: confirmed, prior-matchday, and Matchday-1 projection branches; formation parsing
  and slot-filling; position bucketing; subs selection; preferred-formation derivation.
- `adjust.adjustment_for_match` + `predict.predict_one`: a stronger XI / attacking formation
  shifts λ and W/D/L vs baseline; empty lineup + no played matches degrades to baseline;
  `top_scorelines` sums sensibly and is ordered.
- `card.build_card`: assembles both `TeamCard`s with correct source labels and a prediction.
- `cardpdf.render_card_pdf`: writes a file whose first bytes are `%PDF`; skipped if `reportlab`
  is unavailable.
- `live.refresh_fixture`: fake client merges a single fixture's result and lineup.
- CLI `cmd_card`: fake transport / tmp dirs; `--format` variants; unknown-fixture and
  missing-key error paths.

## Documentation

- README: `wc card` usage, the `[pdf]` extra, and a note that lineups are projected until the
  official XI is published (`--refresh` near kickoff to get the confirmed lineup).
- `docs/architecture.md`: short paragraph on the card pipeline.

## Definition of done

`make format lint typecheck test` all pass (PDF test skips cleanly if `reportlab` absent but is
installed via the `[pdf]` extra in CI/dev); docs updated; final summary names changed files and
validation commands.
