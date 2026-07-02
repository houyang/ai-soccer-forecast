# FIFA 2026 — Full Bracket Forecast + Algorithm Improvement

**Date:** 2026-06-29
**Project:** `glm4.7-cloud-claude` (`soccer_agent/worldcup/`)
**Builds on:** `2026-06-28-worldcup-2026-knockout-predictor-design.md` (shipped in PR #8)

## Goal

Now that the group stage is complete, (1) **improve the prediction algorithm** with an
attack/defense-split expected-goals model, (2) **predict every match of every round through
the Final** as a deterministic modal bracket, and (3) extend the **single-match PDF** to any
two teams (ad-hoc) and auto-generate a PDF for each predicted future-round match.

## Scope

- Continues on branch `feat/wc-knockout-predictor` (folds into PR #8). Same constraints:
  no opus4.8 implementation reuse; API key only in git-ignored `.env`; offline-deterministic
  except `live.py`; ruff `E,F,I,N,W` (ignore E501); Python 3.14 / Pydantic v2.
- The existing `predict` (after-group batch) and `simulate_bracket` (Monte-Carlo) are kept;
  this adds a deterministic modal-bracket forecast and improves the λ model.

## Design

### 1. Attack/defense-split λ (`predict.py`, `form.py`)

`TeamForm` already carries regressed `attack` (goals/game scored) and `defense`
(goals/conceded per game) from the group stage. Currently `recalibrated_strength` uses only
goal difference and `predict_one` splits a fixed 2.6 total by rating supremacy.

New λ model (in a new `predict_match(...)` core; `predict_one` delegates):
- Form-based expected goals: `form_lh = (home.attack + away.defense) / 2`,
  `form_la = (away.attack + home.defense) / 2` (a side scores at its attack rate, constrained
  by the opponent's defense; the average is the meeting point).
- Rating-based expected goals (current): `rating_lh = total/2 + supremacy/2`,
  `rating_la = total/2 - supremacy/2`, `total = BASE_MATCH_GOALS`.
- Blend: `lh = max(LAMBDA_FLOOR, W_FORM_LAMBDA * form_lh + (1 - W_FORM_LAMBDA) * rating_lh)`,
  likewise `la`. `W_FORM_LAMBDA = 0.4` (group-stage form shapes λ but the rating prior still
  dominates). Teams with no group form (none here — all 48 played 3) fall back to rating-only.

`predict_match` is fixture-agnostic (takes `home_id, away_id, venue, kickoff, ...`) so it
serves R32 (real fixture), future rounds (neutral, projected kickoff), and ad-hoc cards.
`predict_one(wc, rankings, strengths, forms, fixture_id, home_lu, away_lu)` looks up the
fixture and delegates to `predict_match`. `forms: dict[int, TeamForm]` is threaded through
all callers (`simulate.py`, `card.py`, `cli.py`, tests).

### 2. Proper bracket tree + deterministic modal forecast (`bracket.py`, `forecast.py` new)

`bracket.py`: keep `r32` (16 fixture_ids in schedule order) and `pairs` (8 R16 pairs). The
tree is the standard balanced binary bracket: R16[i] = winner(R32[2i]) vs winner(R32[2i+1]);
QF[i] = winner(R16[2i]) vs winner(R16[2i+1]); SF[i] = winner(QF[2i]) vs winner(QF[2i+1]);
Final = winner(SF[0]) vs winner(SF[1]); 3rd place = loser(SF[0]) vs loser(SF[1]). R16 pairing
follows R32 schedule order (documented as best-available; the dataset has no official slot map).

`forecast.py`:
- `BracketMatch` dataclass: `round_name, match_no, home_id, away_id, prediction,
  advancing_id, expected_extra_time, kickoff, venue`.
- `forecast_bracket(wc, rankings, strengths, forms, fetcher=None) -> BracketForecast`:
  predicts R32 (real fixtures), advances each match's modal winner (drawn knockout ties
  resolved by ET/penalty — advance `argmax(p_home, p_away)`, favoring higher effective rating
  on a tie; flag `expected_extra_time` when the modal score is drawn), then predicts R16/QF/SF
  /Final/3rd on neutral venues with projected (or live) lineups and projected kickoffs.
- `BracketForecast`: `rounds: dict[str, list[BracketMatch]]`, `champion_id`, `runner_up_id`,
  `to_dict()`.

### 3. Single-match PDF extensions (`card.py`, `cli.py`)

- `build_card(wc, rankings, strengths, forms, home_id, away_id, fetcher=None, fixture_id=None)`:
  no longer raises without a fixture — synthesizes a neutral venue/kickoff and calls
  `predict_match`. Works for ad-hoc two-team matchups and future-round matches.
- `bracket` CLI command: writes `predictions/worldcup-2026-knockout-bracket.{md,json}` (every
  match of every round + champion + modal path) and auto-generates a PDF for each predicted
  future-round match (R16/QF/SF/3rd/Final) into `predictions/bracket-cards/`. Keeps a brief
  Monte-Carlo champion top-5 (reuses `simulate_bracket`) alongside the modal bracket.
- `card "Home" "Away"` CLI: if no scheduled fixture matches the two teams, builds an ad-hoc
  neutral card (instead of erroring).

## Outputs (new)
- `predictions/worldcup-2026-knockout-bracket.{md,json}` — R32→Final+3rd, every match
  predicted (scoreline, W/D/L, λ, rationale, ET flag), champion + runner-up + modal path.
- `predictions/bracket-cards/<Round>-<matchNo>-<Home>-vs-<Away>.pdf` — one PDF per predicted
  future-round match (R16/QF/SF/3rd/Final).

## Tests
- `predict_match`: probs sum to 1; λ tracks attack/defense (a high-attack team has higher λ);
  fixture-agnostic (works with neutral venue).
- `predict_one` still passes existing tests (signature gains `forms`).
- `forecast_bracket`: produces 16 R32 + 8 R16 + 4 QF + 2 SF + 1 Final + 1 3rd; exactly one
  champion; every future-round match has two known teams.
- `card` ad-hoc: `build_card` with two team ids and no fixture returns a valid card with a
  prediction.
- `cli bracket`: writes the bracket md+json and ≥1 future-round PDF in `bracket-cards/`.

## Out of scope
- Changing the existing `predict` (after-group) output or `simulate_bracket` Monte-Carlo.
- Official FIFA R16 slot mapping (not in the dataset; pairing is documented as schedule-order).
