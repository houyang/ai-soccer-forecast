# Full Bracket Forecast + Algorithm Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the λ model with attack/defense-split expected goals, add a deterministic modal-bracket forecast through the Final (+ 3rd place), and extend the single-match PDF to any two teams + auto-generate a PDF per predicted future-round match.

**Architecture:** Refactor `predict.py` around a fixture-agnostic `predict_match` core that blends group-stage attack/defense with rating supremacy; add `forecast.py` walking a proper bracket tree; extend `card.py`/`cli.py` for ad-hoc cards and a `bracket` command that writes a full per-round md+json and future-round PDFs.

**Tech Stack:** Python 3.14, Pydantic v2, stdlib `math`, `reportlab` (optional), pytest.

## Global Constraints
- No reference to opus4.8's implementation; only the raw cached dataset is reused.
- API key only in git-ignored `.env`, read via `os.getenv`; never committed.
- Offline-deterministic except `live.py`; prediction functions pure given inputs + seed.
- Line length 100; ruff `select = ["E","F","I","N","W"]`, `ignore = ["E501"]`, target py312+. Python 3.14, Pydantic v2.
- Thread `forms: dict[int, TeamForm]` through all prediction callers. Keep `predict` (after-group) and `simulate_bracket` (Monte-Carlo) working.

---

### Task 1: Attack/defense-split λ + fixture-agnostic `predict_match`

**Files:**
- Modify: `soccer_agent/worldcup/predict.py`
- Modify: `soccer_agent/worldcup/simulate.py` (thread `forms`)
- Modify: `soccer_agent/worldcup/card.py` (thread `forms`, use `predict_match`)
- Modify: `tests/test_worldcup_predict.py`
- Interfaces:
  - Consumes: `WorldCup`, `Rankings`, `strengths: dict[int,float]`, `forms: dict[int,TeamForm]`, `ProjectedLineup`.
  - Produces: `predict_match(wc, rankings, strengths, forms, home_id, away_id, home_lu, away_lu, *, fixture_id=None, kickoff=None, venue="Neutral", group="", matchday=0, round_name="") -> MatchPrediction`; `predict_one(wc, rankings, strengths, forms, fixture_id, home_lu, away_lu) -> MatchPrediction` (delegates). New constant `W_FORM_LAMBDA = 0.4`. `simulate_bracket(..., forms, ...)` signature gains `forms` after `strengths`.

- [ ] **Step 1: Update the predict test for the new signature + attack/defense behavior**

```python
# tests/test_worldcup_predict.py  (replace contents)
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.lineup import project_lineup
from soccer_agent.worldcup.predict import predict_match, predict_one, top_scorelines, scoreline_matrix


def _setup():
    wc = load_worldcup()
    r = rank_all(wc)
    f = compute_forms(wc)
    return wc, r, recalibrated_strength(wc, r, f), f


def test_probs_sum_to_one_and_floor():
    wc, r, s, f = _setup()
    m = next(m for m in wc.matches if m.matchday == 0)
    hlu = project_lineup(wc, r, m.home_id, m.fixture_id)
    alu = project_lineup(wc, r, m.away_id, m.fixture_id)
    pred = predict_one(wc, r, s, f, m.fixture_id, hlu, alu)
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-9
    assert pred.lambda_home >= 0.18 and pred.lambda_away >= 0.18


def test_predict_match_is_fixture_agnostic():
    wc, r, s, f = _setup()
    m = next(m for m in wc.matches if m.matchday == 0)
    hlu = project_lineup(wc, r, m.home_id, 0)
    alu = project_lineup(wc, r, m.away_id, 0)
    pred = predict_match(wc, r, s, f, m.home_id, m.away_id, hlu, alu,
                         venue="Neutral", group="Knockout", round_name="Round of 32")
    assert pred.home_id == m.home_id
    assert pred.venue == "Neutral"
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-9


def test_attack_defense_shapes_lambda():
    wc, r, s, f = _setup()
    # The team with the best group-stage attack should produce a higher lambda when home
    # than the team with the worst attack, all else equal.
    by_attack = sorted(f.values(), key=lambda x: x.attack, reverse=True)
    strong = by_attack[0]
    weak = by_attack[-1]
    hlu = project_lineup(wc, r, strong.team_id, 0)
    alu = project_lineup(wc, r, weak.team_id, 0)
    pred = predict_match(wc, r, s, f, strong.team_id, weak.team_id, hlu, alu, venue="Neutral")
    assert pred.lambda_home > pred.lambda_away


def test_modal_score_is_matrix_argmax():
    lh, la = 1.5, 1.2
    mat = scoreline_matrix(lh, la)
    tops = top_scorelines(lh, la, 3)
    assert abs(tops[0][2] - max(max(row) for row in mat)) < 1e-9
```

- [ ] **Step 2: Run test — expect FAIL** (`predict_match` undefined, `forms` arg missing)

Run: `python -m pytest tests/test_worldcup_predict.py -q`
Expected: FAIL.

- [ ] **Step 3: Implement `predict_match` + refactor `predict_one` in `predict.py`**

First, make `MatchPrediction.to_dict` null-safe for kickoff (future rounds / ad-hoc cards pass `kickoff=None`): change the `"kickoff": self.kickoff.isoformat(),` line to `"kickoff": self.kickoff.isoformat() if self.kickoff else None,`.

Then replace the body of `predict.py` from the `def effective_rating(...)` line through the end of `predict_one` with:

```python
def effective_rating(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float],
    team_id: int, lineup: ProjectedLineup, is_home: bool, venue: str,
) -> tuple[float, float]:
    """Return (effective_rating, adjustment) for one side."""
    team = wc.teams[team_id]
    base = W_XI * strengths.get(team_id, _NEUTRAL) + (1 - W_XI) * _xi_mean_rating(wc, rankings, lineup)
    adj = 0.0
    if team.is_host and is_home:
        adj += HOST_HOME_FIELD
    adj -= TRAVEL_PENALTY.get(team.confederation, 1.0)
    if any(hint in venue for hint in _HOT_VENUE_HINTS) and team.confederation in _HEAT_SENSITIVE:
        adj -= WEATHER_PENALTY
    return base + adj, adj


W_FORM_LAMBDA = 0.4  # weight on group-stage attack/defense in the blended lambda


def _lambdas(
    eff_h: float, eff_a: float, forms, home_id: int, away_id: int,
) -> tuple[float, float]:
    """Blend rating-supremacy split with group-stage attack/defense expected goals."""
    supremacy = (eff_h - eff_a) / 10.0 * SUPREMACY_PER_10
    total = BASE_MATCH_GOALS
    rating_lh = total / 2.0 + supremacy / 2.0
    rating_la = total / 2.0 - supremacy / 2.0
    fh = forms.get(home_id)
    fa = forms.get(away_id)
    if fh is not None and fa is not None and (fh.played + fa.played) > 0:
        # A side scores at its attack rate, constrained by the opponent's defense.
        form_lh = (fh.attack + fa.defense) / 2.0
        form_la = (fa.attack + fh.defense) / 2.0
        lh = W_FORM_LAMBDA * form_lh + (1 - W_FORM_LAMBDA) * rating_lh
        la = W_FORM_LAMBDA * form_la + (1 - W_FORM_LAMBDA) * rating_la
    else:
        lh, la = rating_lh, rating_la
    return max(LAMBDA_FLOOR, lh), max(LAMBDA_FLOOR, la)


def predict_match(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float], forms,
    home_id: int, away_id: int, home_lu: ProjectedLineup, away_lu: ProjectedLineup,
    *, fixture_id: int | None = None, kickoff=None, venue: str = "Neutral",
    group: str = "", matchday: int = 0, round_name: str = "",
) -> MatchPrediction:
    """Fixture-agnostic prediction (works for real fixtures, future rounds, ad-hoc cards)."""
    from datetime import datetime as _dt
    eff_h, adj_h = effective_rating(wc, rankings, strengths, home_id, home_lu, True, venue)
    eff_a, adj_a = effective_rating(wc, rankings, strengths, away_id, away_lu, False, venue)
    lh, la = _lambdas(eff_h, eff_a, forms, home_id, away_id)

    mat = scoreline_matrix(lh, la)
    p_home = sum(mat[i][j] for i in range(MAX_GOALS + 1) for j in range(i))
    p_away = sum(mat[i][j] for i in range(MAX_GOALS + 1) for j in range(i + 1, MAX_GOALS + 1))
    p_draw = sum(mat[i][i] for i in range(MAX_GOALS + 1))
    best = max(((i, j) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)), key=lambda ij: mat[ij[0]][ij[1]])
    sh, sa = best
    rationale = (
        f"Eff {eff_h:.1f} vs {eff_a:.1f} -> supremacy {(eff_h-eff_a)/10.0*SUPREMACY_PER_10:+.2f}; "
        f"xG {lh:.2f}-{la:.2f}; adj {adj_h:+.1f}/{adj_a:+.1f}."
    )
    ko = kickoff if isinstance(kickoff, _dt) else None
    return MatchPrediction(
        fixture_id=fixture_id if fixture_id is not None else 0, matchday=matchday, group=group,
        kickoff=ko, venue=venue, home_id=home_id, away_id=away_id,
        home_name=wc.teams[home_id].name, away_name=wc.teams[away_id].name,
        lambda_home=lh, lambda_away=la, score_home=sh, score_away=sa,
        p_home=p_home, p_draw=p_draw, p_away=p_away, rationale=rationale,
        home_adjustment=adj_h, away_adjustment=adj_a,
    )


def predict_one(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float], forms,
    fixture_id: int, home_lu: ProjectedLineup, away_lu: ProjectedLineup,
) -> MatchPrediction:
    m = next((x for x in wc.matches if x.fixture_id == fixture_id), None)
    if m is None:
        raise ValueError(f"fixture {fixture_id} not found")
    return predict_match(
        wc, rankings, strengths, forms, m.home_id, m.away_id, home_lu, away_lu,
        fixture_id=m.fixture_id, kickoff=m.kickoff, venue=m.venue,
        group=m.group, matchday=m.matchday, round_name=m.round_name,
    )
```

Also add `W_FORM_LAMBDA` near the other constants (it is defined just above `_lambdas`; that's fine — leave it there).

- [ ] **Step 4: Thread `forms` through `simulate.py`**

In `simulate.py`, change the `simulate_bracket` signature to accept `forms` after `strengths`:
```python
def simulate_bracket(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float], forms,
    fetcher=None, n: int = 10000, seed: int = 2026,
) -> BracketSim:
```
And update the R32 predict call:
```python
        pred = predict_one(wc, rankings, strengths, forms, fid, hlu, alu)
```

- [ ] **Step 5: Run predict + bracket tests — expect PASS (bracket test will need forms too; update in Step 6)**

Run: `python -m pytest tests/test_worldcup_predict.py tests/test_worldcup_bracket.py -q`
Expected: predict tests PASS; bracket tests FAIL (simulate_bracket now requires `forms`).

- [ ] **Step 6: Update `tests/test_worldcup_bracket.py` to pass `forms`**

In `_setup()` add `forms = compute_forms(wc)` and return it; update the `simulate_bracket(...)` call to pass `forms` after `strengths`. Add the import `from soccer_agent.worldcup.form import compute_forms, recalibrated_strength`.

- [ ] **Step 7: Run the worldcup suite — expect PASS**

Run: `python -m pytest tests/test_worldcup_*.py -q`
Expected: all pass (the `card`/`cli` tests will FAIL until Task 3 updates them — if they fail here, that's expected; note it). If `test_worldcup_card.py`/`test_worldcup_cli.py` fail because `build_card`/`predict_one` signatures changed, leave them for Task 3.

- [ ] **Step 8: Commit**

```bash
git add soccer_agent/worldcup/predict.py soccer_agent/worldcup/simulate.py tests/test_worldcup_predict.py tests/test_worldcup_bracket.py
git commit -m "feat(wc): attack/defense-split lambda + fixture-agnostic predict_match"
```

---

### Task 2: Deterministic modal bracket forecast (`forecast.py`)

**Files:**
- Create: `soccer_agent/worldcup/forecast.py`
- Modify: `soccer_agent/worldcup/bracket.py` (document tree; add `r32_schedule` helper)
- Test: `tests/test_worldcup_forecast.py`
- Interfaces:
  - Consumes: `WorldCup`, `Rankings`, `strengths`, `forms`, `predict_match`, `project_lineup`, `build_bracket`.
  - Produces: `BracketMatch(round_name, match_no, home_id, away_id, prediction, advancing_id, expected_extra_time, kickoff, venue)`; `BracketForecast(rounds: dict[str, list[BracketMatch]], champion_id, runner_up_id, third_place_id)` with `to_dict()`; `forecast_bracket(wc, rankings, strengths, forms, fetcher=None) -> BracketForecast`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worldcup_forecast.py
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.forecast import forecast_bracket


def _setup():
    wc = load_worldcup()
    r = rank_all(wc)
    f = compute_forms(wc)
    return wc, r, recalibrated_strength(wc, r, f), f


def test_forecast_has_all_rounds_and_counts():
    wc, r, s, f = _setup()
    fc = forecast_bracket(wc, r, s, f)
    assert len(fc.rounds["R32"]) == 16
    assert len(fc.rounds["R16"]) == 8
    assert len(fc.rounds["QF"]) == 4
    assert len(fc.rounds["SF"]) == 2
    assert len(fc.rounds["Final"]) == 1
    assert len(fc.rounds["3rd"]) == 1
    assert fc.champion_id in wc.teams


def test_every_future_round_match_has_two_teams():
    wc, r, s, f = _setup()
    fc = forecast_bracket(wc, r, s, f)
    for rnd in ("R16", "QF", "SF", "Final", "3rd"):
        for bm in fc.rounds[rnd]:
            assert bm.home_id is not None and bm.away_id is not None
            assert bm.prediction is not None
            assert bm.advancing_id in (bm.home_id, bm.away_id)


def test_champion_won_the_final():
    wc, r, s, f = _setup()
    fc = forecast_bracket(wc, r, s, f)
    final = fc.rounds["Final"][0]
    assert fc.champion_id == final.advancing_id
    # runner-up is the loser of the final
    assert fc.runner_up_id == (final.away_id if final.advancing_id == final.home_id else final.home_id)


def test_to_dict_is_json_serializable():
    import json
    wc, r, s, f = _setup()
    fc = forecast_bracket(wc, r, s, f)
    json.dumps(fc.to_dict())
```

- [ ] **Step 2: Run test — expect FAIL** (module not found)

- [ ] **Step 3: Implement `forecast.py`**

```python
# soccer_agent/worldcup/forecast.py
"""Deterministic modal-bracket forecast: predict every match of every round to the Final.

R32 uses the real fixtures; later rounds pair winners in a balanced binary bracket
(R16[i] = winner(R32[2i]) vs winner(R32[2i+1]), etc.). Knockout ties drawn on the modal
scoreline go to extra time/penalties: the advancing side is argmax(p_home, p_away)
(tiebreak: higher recalibrated strength).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from soccer_agent.worldcup.bracket import build_bracket
from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.lineup import project_lineup
from soccer_agent.worldcup.predict import MatchPrediction, predict_match
from soccer_agent.worldcup.ranking import Rankings

# Projected round windows, as offsets from the latest R32 kickoff.
_ROUND_OFFSET_DAYS = {"R16": 3, "QF": 6, "SF": 10, "Final": 14, "3rd": 13}
_ROUND_VENUE = {
    "R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-Final",
    "SF": "Semi-Final", "Final": "Final", "3rd": "Third-Place Play-off",
}


@dataclass
class BracketMatch:
    round_name: str
    match_no: int
    home_id: Optional[int]
    away_id: Optional[int]
    prediction: Optional[MatchPrediction]
    advancing_id: Optional[int]
    expected_extra_time: bool
    kickoff: Optional[datetime]
    venue: str
    home_name: Optional[str] = None
    away_name: Optional[str] = None
    advancing_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_name, "match_no": self.match_no,
            "home": self.home_name, "away": self.away_name,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "advancing": self.advancing_name,
            "expected_extra_time": self.expected_extra_time,
            "kickoff": self.kickoff.isoformat() if self.kickoff else None,
            "venue": self.venue,
        }


@dataclass
class BracketForecast:
    rounds: dict[str, list[BracketMatch]] = field(default_factory=dict)
    champion_id: Optional[int] = None
    runner_up_id: Optional[int] = None
    third_place_id: Optional[int] = None
    wc: Optional[WorldCup] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": {rnd: [bm.to_dict() for bm in matches] for rnd, matches in self.rounds.items()},
            "champion": self._name(self.champion_id),
            "runner_up": self._name(self.runner_up_id),
            "third_place": self._name(self.third_place_id),
        }

    def _name(self, tid: Optional[int]) -> Optional[str]:
        return self.wc.teams[tid].name if (tid is not None and self.wc and tid in self.wc.teams) else None


def _advance(pred: MatchPrediction, strengths: dict[int, float]) -> tuple[int, bool]:
    """Return (advancing_id, expected_extra_time) for a knockout tie."""
    et = pred.score_home == pred.score_away  # modal score drawn -> ET/pen
    if pred.p_home >= pred.p_away:
        adv = pred.home_id
    else:
        adv = pred.away_id
    # tiebreak by strength if probs are effectively equal
    if abs(pred.p_home - pred.p_away) < 1e-9:
        adv = pred.home_id if strengths.get(pred.home_id, 50.0) >= strengths.get(pred.away_id, 50.0) else pred.away_id
    return adv, et


def forecast_bracket(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float], forms, fetcher=None,
) -> BracketForecast:
    bracket = build_bracket(wc)
    r32_fixtures = [next(x for x in wc.matches if x.fixture_id == fid) for fid in bracket.r32]
    base_ko = max(m.kickoff for m in r32_fixtures)
    out = BracketForecast(wc=wc, rounds={"R32": [], "R16": [], "QF": [], "SF": [], "Final": [], "3rd": []})

    def name(tid: int) -> str:
        return wc.teams[tid].name

    def predict_pair(home_id: int, away_id: int, round_name: str, match_no: int, kickoff, venue: str) -> BracketMatch:
        hlu = project_lineup(wc, rankings, home_id, 0, fetcher)
        alu = project_lineup(wc, rankings, away_id, 0, fetcher)
        pred = predict_match(wc, rankings, strengths, forms, home_id, away_id, hlu, alu,
                             kickoff=kickoff, venue=venue, group=round_name, round_name=round_name)
        adv, et = _advance(pred, strengths)
        return BracketMatch(round_name, match_no, home_id, away_id, pred, adv, et, kickoff, venue,
                            home_name=name(home_id), away_name=name(away_id), advancing_name=name(adv))

    # R32: real fixtures.
    for i, m in enumerate(r32_fixtures, start=1):
        hlu = project_lineup(wc, rankings, m.home_id, m.fixture_id, fetcher)
        alu = project_lineup(wc, rankings, m.away_id, m.fixture_id, fetcher)
        pred = predict_match(wc, rankings, strengths, forms, m.home_id, m.away_id, hlu, alu,
                             fixture_id=m.fixture_id, kickoff=m.kickoff, venue=m.venue,
                             group=m.group or "R32", matchday=m.matchday, round_name="Round of 32")
        adv, et = _advance(pred, strengths)
        out.rounds["R32"].append(BracketMatch("R32", i, m.home_id, m.away_id, pred, adv, et, m.kickoff, m.venue,
                                              home_name=name(m.home_id), away_name=name(m.away_id), advancing_name=name(adv)))

    # Walk the tree: winners feed the next round.
    winners = [bm.advancing_id for bm in out.rounds["R32"]]  # 16
    losers_sf: list[int] = []

    def walk(slot_teams: list[int], round_name: str) -> list[int]:
        offset = _ROUND_OFFSET_DAYS[round_name]
        ko = base_ko + timedelta(days=offset)
        venue = _ROUND_VENUE[round_name]
        nxt: list[int] = []
        for i in range(0, len(slot_teams), 2):
            match_no = i // 2 + 1
            bm = predict_pair(slot_teams[i], slot_teams[i + 1], round_name, match_no, ko, venue)
            out.rounds[round_name].append(bm)
            nxt.append(bm.advancing_id)
            if round_name == "SF":
                loser = slot_teams[i] if bm.advancing_id == slot_teams[i + 1] else slot_teams[i + 1]
                losers_sf.append(loser)
        return nxt

    r16_winners = walk(winners, "R16")          # 8 -> 4
    qf_winners = walk(r16_winners, "QF")        # 4 -> 2
    sf_winners = walk(qf_winners, "SF")         # 2 -> 1 (SF winners), losers_sf filled
    final_bm = predict_pair(sf_winners[0], sf_winners[1], "Final", 1,
                            base_ko + timedelta(days=_ROUND_OFFSET_DAYS["Final"]), _ROUND_VENUE["Final"])
    out.rounds["Final"].append(final_bm)
    out.champion_id = final_bm.advancing_id
    out.runner_up_id = final_bm.away_id if final_bm.advancing_id == final_bm.home_id else final_bm.home_id
    third_bm = predict_pair(losers_sf[0], losers_sf[1], "3rd", 1,
                            base_ko + timedelta(days=_ROUND_OFFSET_DAYS["3rd"]), _ROUND_VENUE["3rd"])
    out.rounds["3rd"].append(third_bm)
    out.third_place_id = third_bm.advancing_id
    return out
```

- [ ] **Step 4: Run test — expect PASS**

Run: `python -m pytest tests/test_worldcup_forecast.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add soccer_agent/worldcup/forecast.py tests/test_worldcup_forecast.py
git commit -m "feat(wc): deterministic modal-bracket forecast through the Final + 3rd place"
```

---

### Task 3: Ad-hoc `card` + `bracket` CLI command

**Files:**
- Modify: `soccer_agent/worldcup/card.py` (thread `forms`, ad-hoc neutral card via `predict_match`)
- Modify: `soccer_agent/worldcup/cli.py` (`_engine` returns `forms`; `bracket` command writes md+json + future-round PDFs; `card` ad-hoc)
- Modify: `tests/test_worldcup_card.py`, `tests/test_worldcup_cli.py`
- Interfaces:
  - `build_card(wc, rankings, strengths, forms, home_id, away_id, fetcher=None, fixture_id=None, kickoff=None, venue=None) -> MatchCard` (no longer raises without a fixture; synthesizes neutral).

- [ ] **Step 1: Update `card.py`**

Change `build_card` signature to add `forms` after `strengths` and add optional `kickoff`/`venue`. Replace the prediction block (lines that raise `ValueError`) with a `predict_match` call that works with or without a fixture:

```python
from soccer_agent.worldcup.predict import MatchPrediction, predict_match, predict_one, top_scorelines
```
...(keep imports otherwise)...

Replace the body of `build_card` with:
```python
def build_card(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float], forms,
    home_id: int, away_id: int, fetcher=None, fixture_id: int | None = None,
    kickoff=None, venue: str | None = None,
) -> MatchCard:
    m = None
    if fixture_id is not None:
        m = next((x for x in wc.matches if x.fixture_id == fixture_id), None)
    if m is None:
        m = next((x for x in wc.matches if x.matchday == 0 and {x.home_id, x.away_id} == {home_id, away_id}), None)
    fid = m.fixture_id if m else None
    group = m.group if m and m.group else "Knockout"
    ko = kickoff if kickoff is not None else (m.kickoff if m else None)
    ven = venue if venue is not None else (m.venue if m else "Neutral venue")

    hlu = project_lineup(wc, rankings, home_id, fid or 0, fetcher)
    alu = project_lineup(wc, rankings, away_id, fid or 0, fetcher)

    pred = predict_match(wc, rankings, strengths, forms, home_id, away_id, hlu, alu,
                         fixture_id=fid, kickoff=ko, venue=ven, group=group, round_name="Knockout")
    tops = tuple(top_scorelines(pred.lambda_home, pred.lambda_away, 3))
    return MatchCard(
        fixture_id=fid, group=group, kickoff=ko, venue=ven,
        home=_team_card(wc, rankings, home_id, hlu),
        away=_team_card(wc, rankings, away_id, alu),
        prediction=pred, top_scorelines=tops,
    )
```

- [ ] **Step 2: Update `cli.py`**

`_engine()` returns `forms` too:
```python
def _engine():
    wc = load_worldcup()
    rankings = rank_all(wc)
    forms = compute_forms(wc)
    strengths = recalibrated_strength(wc, rankings, forms)
    fetcher = LineupFetcher() if os.getenv("API_FOOTBALL_KEY") else None
    return wc, rankings, strengths, forms, fetcher
```
Update every `cmd` branch to unpack `wc, rankings, strengths, forms, fetcher = _engine()` and pass `forms`:
- `_write_predictions(wc, rankings, strengths, forms, fetcher)` → `simulate_bracket(wc, rankings, strengths, forms, fetcher=fetcher, n=10000)`.
- `_write_card(wc, rankings, strengths, forms, fetcher, home_name, away_name)` → if no R32 fixture matches, call `build_card(...)` without a fixture (ad-hoc neutral) instead of erroring:
```python
def _write_card(wc, rankings, strengths, forms, fetcher, home_name, away_name) -> int:
    home = _team_by_name(wc, home_name)
    away = _team_by_name(wc, away_name)
    m = next((x for x in wc.matches if x.matchday == 0 and {x.home_id, x.away_id} == {home.id, away.id}), None)
    fid = m.fixture_id if m else None
    card = build_card(wc, rankings, strengths, forms, home.id, away.id, fetcher=fetcher, fixture_id=fid)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{home.name}-vs-{away.name}"
    (PRED_DIR / f"{stem}.json").write_text(json.dumps(card.to_dict(), indent=2))
    try:
        render_card_pdf(card, PRED_DIR / f"{stem}.pdf")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
    return 0
```
- Replace the `bracket` branch with a new `_write_bracket` that writes the modal bracket md+json + future-round PDFs (and keeps a brief MC champion top-5). Add imports:
```python
from soccer_agent.worldcup.forecast import forecast_bracket
from soccer_agent.worldcup.simulate import simulate_bracket
```
```python
def _write_bracket(wc, rankings, strengths, forms, fetcher) -> int:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    fc = forecast_bracket(wc, rankings, strengths, forms, fetcher=fetcher)
    payload = fc.to_dict()
    # Brief Monte-Carlo champion context.
    sim = simulate_bracket(wc, rankings, strengths, forms, fetcher=fetcher, n=10000)
    champ_top5 = sorted(sim.champion.items(), key=lambda kv: kv[1], reverse=True)[:5]
    payload["monte_carlo_champion_top5"] = [{"team": wc.teams[t].name, "probability": round(p, 4)} for t, p in champ_top5]
    payload["method"] = ("Deterministic modal bracket: each match's modal winner advances (drawn "
                         "ties go to ET/penalties). R32 = real fixtures; R16→Final pairing follows "
                         "R32 schedule order (documented). MC champion odds are a separate 10k sim.")
    (PRED_DIR / "worldcup-2026-knockout-bracket.json").write_text(json.dumps(payload, indent=2))

    lines = ["# FIFA 2026 World Cup — Knockout Bracket Forecast", ""]
    lines.append("Deterministic modal bracket: every match predicted; the modal winner advances each round.")
    lines.append("Drawn knockout ties go to extra time/penalties (marked `ET`). R32 = real fixtures; "
                 "R16→Final pairing follows R32 schedule order (best-available; dataset has no official slot map).")
    lines.append("")
    for rnd in ("R32", "R16", "QF", "SF", "3rd", "Final"):
        matches = fc.rounds.get(rnd, [])
        if not matches:
            continue
        title = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-Finals",
                 "SF": "Semi-Finals", "Final": "Final", "3rd": "Third-Place Play-off"}[rnd]
        lines.append(f"\n## {title}\n")
        for bm in matches:
            p = bm.prediction
            ko = bm.kickoff.strftime("%Y-%m-%d %H:%M UTC") if bm.kickoff else "TBD"
            et = " (ET/pen)" if bm.expected_extra_time else ""
            lines.append(
                f"- `{ko}`  **{bm.home_name} {p.score_home}-{p.score_away} {bm.away_name}**{et}  "
                f"(W {p.p_home:.0%} / D {p.p_draw:.0%} / L {p.p_away:.0%})  "
                f"-> advances: **{bm.advancing_name}**"
            )
    lines.append("\n## Champion")
    lines.append(f"**{fc.to_dict()['champion']}** (runner-up: {fc.to_dict()['runner_up']}; "
                 f"third: {fc.to_dict()['third_place']})")
    lines.append("\n### Monte-Carlo champion odds (10k sims, top 5)")
    for t, p in champ_top5:
        lines.append(f"- {wc.teams[t].name}: {p:.1%}")
    (PRED_DIR / "worldcup-2026-knockout-bracket.md").write_text("\n".join(lines))

    # Auto-generate a PDF for each predicted future-round match.
    cards_dir = PRED_DIR / "bracket-cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for rnd in ("R16", "QF", "SF", "3rd", "Final"):
        for bm in fc.rounds.get(rnd, []):
            card = build_card(wc, rankings, strengths, forms, bm.home_id, bm.away_id,
                              fetcher=fetcher, kickoff=bm.kickoff, venue=bm.venue)
            stem = f"{rnd}-{bm.match_no}-{bm.home_name}-vs-{bm.away_name}"
            (cards_dir / f"{stem}.json").write_text(json.dumps(card.to_dict(), indent=2))
            try:
                render_card_pdf(card, cards_dir / f"{stem}.pdf")
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
    return 0
```
Update `main()`:
```python
    wc, rankings, strengths, forms, fetcher = _engine()
    if cmd == "predict":
        return _write_predictions(wc, rankings, strengths, forms, fetcher)
    if cmd == "card":
        if len(argv) < 3:
            print("usage: card \"Home\" \"Away\"", file=sys.stderr)
            return 2
        return _write_card(wc, rankings, strengths, forms, fetcher, argv[1], argv[2])
    if cmd == "bracket":
        return _write_bracket(wc, rankings, strengths, forms, fetcher)
```

- [ ] **Step 3: Update tests**

`tests/test_worldcup_card.py`: update `_setup()` to compute `forms` and pass to `build_card`. The existing `test_build_card_structure` and `test_render_card_pdf_skips_without_reportlab` keep working (pass `forms`). Add:
```python
def test_build_card_ad_hoc_no_fixture():
    wc, r, s, f = _setup()
    # Two teams that did NOT play each other in R32 -> ad-hoc neutral card.
    card = build_card(wc, r, s, f, home_id=1, away_id=2, fixture_id=None)
    assert card.prediction is not None
    assert card.venue == "Neutral venue"
    assert len(card.home.starters) == 11
```
`tests/test_worldcup_cli.py`: update `_engine` usage is internal; the tests call `main([...])`. Add a bracket test:
```python
def test_bracket_writes_outputs(tmp_path, monkeypatch):
    import pytest
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    code = main(["bracket"])
    assert code == 0
    md = tmp_path / "predictions" / "worldcup-2026-knockout-bracket.md"
    js = tmp_path / "predictions" / "worldcup-2026-knockout-bracket.json"
    assert md.exists() and js.exists()
    import json
    d = json.loads(js.read_text())
    assert "rounds" in d and "champion" in d
    assert len(d["rounds"]["R32"]) == 16 and len(d["rounds"]["Final"]) == 1
    cards = list((tmp_path / "predictions" / "bracket-cards").glob("*.pdf"))
    assert cards, "expected future-round PDFs"
```
And update `test_card_writes_pdf_and_json` — it already derives the first R32 fixture's teams and calls `main(["card", home, away])`, which still works (that fixture exists). Keep it.

- [ ] **Step 4: Run the worldcup suite — expect PASS**

Run: `python -m pytest tests/test_worldcup_*.py -q`
Expected: all pass (including the new ad-hoc card + bracket tests).

- [ ] **Step 5: Commit**

```bash
git add soccer_agent/worldcup/card.py soccer_agent/worldcup/cli.py tests/test_worldcup_card.py tests/test_worldcup_cli.py
git commit -m "feat(wc): ad-hoc match card + bracket command with per-round PDFs"
```

---

### Task 4: Verify, regenerate outputs, finalize

**Files:** run commands; commit regenerated `predictions/`.

- [ ] **Step 1: Lint + full worldcup suite**

```bash
ruff check soccer_agent/worldcup tests/test_worldcup_*.py
python -m pytest tests/test_worldcup_*.py -q
```
Expected: ruff clean; all tests pass.

- [ ] **Step 2: Regenerate the bracket forecast with live lineups**

```bash
set -a; source .env; set +a
python -m soccer_agent.worldcup bracket
```
Expected: writes `predictions/worldcup-2026-knockout-bracket.{md,json}` and `predictions/bracket-cards/*.pdf`. Confirm champion + that bracket-cards has PDFs for R16/QF/SF/3rd/Final.

- [ ] **Step 3: Regenerate the after-group predictions (model changed) + a sample ad-hoc card**

```bash
python -m soccer_agent.worldcup predict
python -m soccer_agent.worldcup card "Argentina" "France"   # ad-hoc neutral (no R32 fixture)
```
Expected: `predict` regenerates the after-group md+json (champion odds may shift slightly due to attack/defense λ); `card` writes an ad-hoc `Argentina-vs-France.{pdf,json}` with neutral venue.

- [ ] **Step 4: Sanity-check**

```bash
python -c "import json; d=json.load(open('predictions/worldcup-2026-knockout-bracket.json')); print('champion:', d['champion'], '| rounds:', {k: len(v) for k,v in d['rounds'].items()})"
ls predictions/bracket-cards/*.pdf | wc -l
```
Expected: champion name; round counts 16/8/4/2/1/1; ≥8 PDFs in bracket-cards.

- [ ] **Step 5: Verify .env / data/live not staged; commit**

```bash
git status --porcelain | grep -E '\.env|data/live' && echo ABORT || echo safe
git add predictions soccer_agent/worldcup docs/superpowers/specs/2026-06-29-worldcup-2026-full-bracket-forecast-design.md docs/superpowers/plans/2026-06-29-worldcup-2026-full-bracket-forecast.md
git commit -m "data(wc): generate full knockout bracket forecast + per-round match cards"
```

- [ ] **Step 6: Push**

```bash
git push
```
