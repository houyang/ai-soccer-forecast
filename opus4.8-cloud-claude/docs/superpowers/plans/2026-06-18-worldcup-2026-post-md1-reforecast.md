# WorldCup 2026 Post-Matchday-1 Re-forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold actual matchday-1 results and each team's formation/starting-XI into the group-stage predictor so remaining-game forecasts are sharper, written to a new pair of output files.

**Architecture:** Keep the Poisson-supremacy core. Add a live incremental fetch (results + lineups) into the dataset, a new `adjust` module that turns played matches into bounded per-team rating deltas + a formation λ-lean, and a `predict_remaining` path that forecasts only unplayed matches. A team with no played match yields a zero adjustment, so the model degrades to the existing baseline.

**Tech Stack:** Python 3.11+, stdlib only (urllib, dataclasses), pytest, ruff, mypy. Package `soccer` under `src/` layout.

---

## File Structure

- `src/soccer/worldcup/entities.py` (modify) — add `Lineup` dataclass; add `lineups` to `WorldCup`.
- `src/soccer/worldcup/apifootball.py` (modify) — add `force_refresh` to `get`/`_fetch_page`.
- `src/soccer/worldcup/live.py` (create) — `refresh_live(wc, client)`: merge live results + lineups.
- `src/soccer/worldcup/adjust.py` (create) — `compute_adjustments(wc, rankings)`, `TeamAdjustment`.
- `src/soccer/worldcup/predict.py` (modify) — `predict_remaining`; adjustment-aware `_predict`; new `MatchPrediction` fields.
- `src/soccer/worldcup/cli.py` (modify) — `wc refresh` command; `wc predict --remaining/--out-dir/--name`.
- Tests mirror each module under `tests/worldcup/`.
- `README.md` (modify) — document the two new commands.

---

## Task 1: `Lineup` entity and `WorldCup.lineups`

**Files:**
- Modify: `src/soccer/worldcup/entities.py`
- Test: `tests/worldcup/test_entities.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/worldcup/test_entities.py`:

```python
def test_lineup_round_trips() -> None:
    from soccer.worldcup.entities import Lineup

    lu = Lineup(
        fixture_id=9001,
        team_id=1,
        formation="4-3-3",
        start_ids=(1, 2, 3),
        sub_ids=(4, 5),
    )
    assert Lineup.from_dict(lu.to_dict()) == lu


def test_world_cup_round_trips_lineups() -> None:
    from soccer.worldcup.entities import Lineup, WorldCup

    wc = WorldCup(lineups=(Lineup(9001, 1, "4-3-3", (1, 2), (3,)),))
    restored = WorldCup.from_dict(wc.to_dict())
    assert restored.lineups == wc.lineups
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/worldcup/test_entities.py -k lineup -v`
Expected: FAIL with `ImportError: cannot import name 'Lineup'`.

- [ ] **Step 3: Add the `Lineup` dataclass**

In `src/soccer/worldcup/entities.py`, add after the `WcMatch` class (before `WorldCup`):

```python
@dataclass(frozen=True)
class Lineup:
    fixture_id: int
    team_id: int
    formation: str
    start_ids: tuple[int, ...]
    sub_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "team_id": self.team_id,
            "formation": self.formation,
            "start_ids": list(self.start_ids),
            "sub_ids": list(self.sub_ids),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Lineup:
        return cls(
            fixture_id=int(raw["fixture_id"]),
            team_id=int(raw["team_id"]),
            formation=str(raw["formation"]),
            start_ids=tuple(int(x) for x in raw["start_ids"]),
            sub_ids=tuple(int(x) for x in raw["sub_ids"]),
        )
```

- [ ] **Step 4: Thread `lineups` through `WorldCup`**

In the `WorldCup` dataclass, add the field after `matches`:

```python
    matches: tuple[WcMatch, ...] = ()
    lineups: tuple[Lineup, ...] = ()
```

In `WorldCup.to_dict`, add to the returned dict:

```python
            "matches": [m.to_dict() for m in self.matches],
            "lineups": [lu.to_dict() for lu in self.lineups],
```

In `WorldCup.from_dict`, add the keyword arg:

```python
            matches=tuple(WcMatch.from_dict(x) for x in raw.get("matches", [])),
            lineups=tuple(Lineup.from_dict(x) for x in raw.get("lineups", [])),
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/worldcup/test_entities.py -v`
Expected: PASS (new tests plus existing entity tests).

- [ ] **Step 6: Commit**

```bash
git add src/soccer/worldcup/entities.py tests/worldcup/test_entities.py
git commit -m "feat(wc): add Lineup entity and WorldCup.lineups"
```

---

## Task 2: `force_refresh` on the API client

**Files:**
- Modify: `src/soccer/worldcup/apifootball.py`
- Test: `tests/worldcup/test_apifootball.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/worldcup/test_apifootball.py`:

```python
def test_force_refresh_bypasses_cache_read_but_stores(tmp_path: Path) -> None:
    base = "https://api.test"
    transport = RecordingTransport(
        {f"{base}/fixtures?league=1": (200, _page([{"id": 7}], 1, 1))}
    )
    cache = JsonCache(tmp_path)
    client = ApiFootballClient("k", base_url=base, transport=transport, cache=cache)
    client.get("fixtures", {"league": 1})  # warms the cache
    client.get("fixtures", {"league": 1}, force_refresh=True)  # ignores cache, refetches
    assert len(transport.calls) == 2
    assert cache.has("fixtures?league=1&page=1")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/worldcup/test_apifootball.py -k force_refresh -v`
Expected: FAIL with `TypeError: get() got an unexpected keyword argument 'force_refresh'`.

- [ ] **Step 3: Add `force_refresh` to `_fetch_page` and `get`**

In `src/soccer/worldcup/apifootball.py`, change `_fetch_page` signature and cache-read guard:

```python
    def _fetch_page(
        self, path: str, params: Mapping[str, Any], page: int, *, force_refresh: bool = False
    ) -> dict[str, Any]:
        key = self._cache_key(path, params, page)
        if self._cache is not None and not force_refresh:
            cached = self._cache.load(key)
            if cached is not None:
                return dict(cached)
```

Change `get` to accept and forward the flag:

```python
    def get(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
        *,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        """Return the concatenated ``response`` array across all pages."""
        params = dict(params or {})
        out: list[dict[str, Any]] = []
        page = 1
        while page <= _MAX_PAGES:
            payload = self._fetch_page(path, params, page, force_refresh=force_refresh)
            out.extend(payload.get("response", []))
            paging = payload.get("paging") or {}
            current = int(paging.get("current", page))
            total = int(paging.get("total", current))
            if current >= total:
                break
            page = current + 1
        return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/worldcup/test_apifootball.py -v`
Expected: PASS (all client tests, including the existing cache test).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/worldcup/apifootball.py tests/worldcup/test_apifootball.py
git commit -m "feat(wc): add force_refresh to ApiFootballClient.get"
```

---

## Task 3: `live.py` — merge live results and lineups

**Files:**
- Create: `src/soccer/worldcup/live.py`
- Test: `tests/worldcup/test_live.py`

- [ ] **Step 1: Write the failing test**

Create `tests/worldcup/test_live.py`:

```python
from __future__ import annotations

from typing import Any

from soccer.worldcup.entities import WorldCup
from soccer.worldcup.live import refresh_live


class FakeClient:
    """Stands in for ApiFootballClient: serves canned responses by (path, fixture)."""

    def __init__(self, fixtures: list[dict[str, Any]], lineups: dict[int, list[dict[str, Any]]]):
        self._fixtures = fixtures
        self._lineups = lineups
        self.forced: list[str] = []

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        params = params or {}
        if path == "fixtures":
            if force_refresh:
                self.forced.append(path)
            return self._fixtures
        if path == "fixtures/lineups":
            return self._lineups.get(int(params["fixture"]), [])
        raise AssertionError(f"unexpected path {path}")


def _fixture(fid: int, home: int, away: int, status: str, hg: int, ag: int) -> dict[str, Any]:
    return {
        "fixture": {"id": fid, "status": {"short": status}},
        "teams": {"home": {"id": home}, "away": {"id": away}},
        "goals": {"home": hg, "away": ag},
    }


def _lineup_block(team: int, formation: str, starters: list[int], subs: list[int]) -> dict[str, Any]:
    return {
        "team": {"id": team},
        "formation": formation,
        "startXI": [{"player": {"id": p}} for p in starters],
        "substitutes": [{"player": {"id": p}} for p in subs],
    }


def test_refresh_live_applies_results_and_lineups(sample_world_cup: WorldCup) -> None:
    # sample fixture 9001 is England(1) vs Mexico(2), matchday 1, currently unplayed.
    client = FakeClient(
        fixtures=[_fixture(9001, 1, 2, "FT", 2, 0)],
        lineups={
            9001: [
                _lineup_block(1, "4-3-3", [1, 2], []),
                _lineup_block(2, "5-4-1", [3, 4], []),
            ]
        },
    )
    updated = refresh_live(sample_world_cup, client)
    played = next(m for m in updated.matches if m.fixture_id == 9001)
    assert played.played and played.home_goals == 2 and played.away_goals == 0
    assert client.forced == ["fixtures"]  # fixtures pulled fresh, not from cache
    assert {lu.team_id for lu in updated.lineups} == {1, 2}
    eng = next(lu for lu in updated.lineups if lu.team_id == 1)
    assert eng.formation == "4-3-3" and eng.start_ids == (1, 2)


def test_refresh_live_skips_unfinished_and_tolerates_missing_lineups(
    sample_world_cup: WorldCup,
) -> None:
    client = FakeClient(fixtures=[_fixture(9001, 1, 2, "NS", 0, 0)], lineups={})
    updated = refresh_live(sample_world_cup, client)
    assert not any(m.played for m in updated.matches)
    assert updated.lineups == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/worldcup/test_live.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'soccer.worldcup.live'`.

- [ ] **Step 3: Implement `live.py`**

Create `src/soccer/worldcup/live.py`:

```python
"""Incremental live refresh: merge actual results and lineups into an existing dataset.

This path never re-fetches the static entities (teams/players/clubs/coaches) — it only
pulls fresh ``fixtures`` (to fill scorelines) and ``fixtures/lineups`` for finished matches,
so refreshing mid-tournament costs only a handful of API calls. The injected client matches
:class:`~soccer.worldcup.apifootball.ApiFootballClient`'s ``get`` signature, so tests pass a
fake with no network.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Protocol

from soccer.worldcup.entities import Lineup, WcMatch, WorldCup
from soccer.worldcup.ingest import WC_LEAGUE_ID, WC_SEASON, _safe_int

logger = logging.getLogger(__name__)

Json = dict[str, Any]
_FINISHED = {"FT", "AET", "PEN"}


class _Client(Protocol):
    def get(
        self, path: str, params: dict[str, Any] | None = ..., *, force_refresh: bool = ...
    ) -> list[Json]: ...


def _results_by_fixture(fixtures: list[Json]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for item in fixtures:
        fixture = item.get("fixture", {})
        if fixture.get("status", {}).get("short") not in _FINISHED:
            continue
        goals = item.get("goals", {})
        out[_safe_int(fixture.get("id"))] = (
            _safe_int(goals.get("home")),
            _safe_int(goals.get("away")),
        )
    return out


def _apply_results(
    matches: tuple[WcMatch, ...], results: dict[int, tuple[int, int]]
) -> tuple[WcMatch, ...]:
    updated: list[WcMatch] = []
    for match in matches:
        if match.fixture_id in results and not match.played:
            home_goals, away_goals = results[match.fixture_id]
            updated.append(replace(match, home_goals=home_goals, away_goals=away_goals))
        else:
            updated.append(match)
    return tuple(updated)


def _parse_lineups(fixture_id: int, blocks: list[Json]) -> list[Lineup]:
    out: list[Lineup] = []
    for block in blocks:
        team_id = _safe_int(block.get("team", {}).get("id"))
        start_ids = tuple(
            _safe_int(entry.get("player", {}).get("id")) for entry in block.get("startXI", [])
        )
        sub_ids = tuple(
            _safe_int(entry.get("player", {}).get("id")) for entry in block.get("substitutes", [])
        )
        out.append(
            Lineup(
                fixture_id=fixture_id,
                team_id=team_id,
                formation=str(block.get("formation") or ""),
                start_ids=start_ids,
                sub_ids=sub_ids,
            )
        )
    return out


def refresh_live(wc: WorldCup, client: _Client) -> WorldCup:
    """Return a copy of ``wc`` with results filled in and lineups attached."""
    fixtures = client.get("fixtures", {"league": WC_LEAGUE_ID, "season": WC_SEASON}, force_refresh=True)
    matches = _apply_results(wc.matches, _results_by_fixture(fixtures))
    lineups: list[Lineup] = []
    for match in matches:
        if not match.played:
            continue
        blocks = client.get("fixtures/lineups", {"fixture": match.fixture_id})
        if not blocks:
            logger.warning("lineups unavailable for fixture %s", match.fixture_id)
            continue
        lineups.extend(_parse_lineups(match.fixture_id, blocks))
    return replace(wc, matches=matches, lineups=tuple(lineups))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/worldcup/test_live.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/worldcup/live.py tests/worldcup/test_live.py
git commit -m "feat(wc): add live refresh of results and lineups"
```

---

## Task 4: `adjust.py` — per-team rating deltas and formation lean

**Files:**
- Create: `src/soccer/worldcup/adjust.py`
- Test: `tests/worldcup/test_adjust.py`

- [ ] **Step 1: Write the failing test**

Create `tests/worldcup/test_adjust.py`:

```python
from __future__ import annotations

from dataclasses import replace

from soccer.worldcup.adjust import (
    CAP_TOTAL,
    TeamAdjustment,
    compute_adjustments,
    parse_formation,
)
from soccer.worldcup.entities import Lineup, WorldCup
from soccer.worldcup.ranking import rank_all


def test_parse_formation() -> None:
    assert parse_formation("4-3-3") == (4, 3)
    assert parse_formation("5-4-1") == (5, 1)
    assert parse_formation("") is None
    assert parse_formation("nonsense") is None


def test_no_played_match_means_no_adjustment(sample_world_cup: WorldCup) -> None:
    adj = compute_adjustments(sample_world_cup, rank_all(sample_world_cup))
    assert adj == {}


def _play(wc: WorldCup, home_goals: int, away_goals: int) -> WorldCup:
    match = replace(wc.matches[0], home_goals=home_goals, away_goals=away_goals)
    return replace(wc, matches=(match,))


def test_overperformance_gives_positive_momentum(sample_world_cup: WorldCup) -> None:
    # England (home, the favourite) thrashes Mexico beyond expectation -> positive momentum.
    wc = _play(sample_world_cup, 5, 0)
    adj = compute_adjustments(wc, rank_all(wc))
    assert adj[1].momentum > 0
    assert adj[2].momentum < 0  # Mexico under-performed


def test_rating_delta_is_capped(sample_world_cup: WorldCup) -> None:
    wc = _play(sample_world_cup, 8, 0)
    adj = compute_adjustments(wc, rank_all(wc))
    assert abs(adj[1].rating_delta) <= CAP_TOTAL + 1e-9


def test_formation_lean_from_lineup(sample_world_cup: WorldCup) -> None:
    wc = _play(sample_world_cup, 1, 0)
    wc = replace(wc, lineups=(Lineup(9001, 1, "3-4-3", (1, 2), ()),))
    adj = compute_adjustments(wc, rank_all(wc))
    # 3 at the back -> negative defensive lean (less solid); 3 up top -> neutral attack.
    assert adj[1].defense_lean < 0
    assert adj[1].attack_lean == 0.0


def test_team_adjustment_defaults_to_zero() -> None:
    assert TeamAdjustment() == TeamAdjustment(0.0, 0.0, 0.0, 0.0, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/worldcup/test_adjust.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'soccer.worldcup.adjust'`.

- [ ] **Step 3: Implement `adjust.py`**

Create `src/soccer/worldcup/adjust.py`:

```python
"""Post-matchday adjustments derived only from matches a team has actually played.

Each played match yields a bounded rating delta (momentum from the scoreline vs the
pre-tournament line, plus a lineup-quality term from who actually started) and a small
formation-based lambda lean. A team with no played match gets a zero adjustment, so the
prediction layer degrades exactly to the pre-tournament baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

from soccer.worldcup.entities import Lineup, WcMatch, WorldCup
from soccer.worldcup.predict import SUPREMACY_PER_10, _effective_rating
from soccer.worldcup.ranking import Rankings

# Momentum: rating points per goal of over-/under-performance vs the pre-tournament line.
K_MOM = 0.8
CAP_MOM = 4.0
# Lineup: rating points per point of (starting-XI quality - squad-core quality).
K_LU = 0.15
CAP_LU = 3.0
# One match must not swamp pedigree, so the combined rating delta is capped.
CAP_TOTAL = 5.0
# Formation lean (goals): per forward above 3, per defender above 4.
FORM_ATTACK = 0.06
FORM_DEFENSE = 0.05
_SQUAD_CORE = 16  # mirrors ranking._SQUAD_CORE: strongest N players define squad quality
_NEUTRAL = 50.0


@dataclass(frozen=True)
class TeamAdjustment:
    rating_delta: float = 0.0
    momentum: float = 0.0
    lineup: float = 0.0
    attack_lean: float = 0.0
    defense_lean: float = 0.0


def _clamp(value: float, cap: float) -> float:
    return max(-cap, min(cap, value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def base_supremacy(wc: WorldCup, rankings: Rankings, match: WcMatch) -> float:
    """Pre-tournament home-oriented goal supremacy with no post-match adjustments."""
    base_h = rankings.teams.get(match.home_id, _NEUTRAL)
    base_a = rankings.teams.get(match.away_id, _NEUTRAL)
    eff_h = _effective_rating(wc, match.home_id, base_h, is_home=True, venue=match.venue)
    eff_a = _effective_rating(wc, match.away_id, base_a, is_home=False, venue=match.venue)
    return SUPREMACY_PER_10 * (eff_h - eff_a) / 10.0


def parse_formation(formation: str) -> tuple[int, int] | None:
    """Return (defenders, forwards) from e.g. '4-3-3'; None if unparseable."""
    parts = formation.split("-")
    if len(parts) < 2:
        return None
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    return nums[0], nums[-1]


def _momentum(wc: WorldCup, rankings: Rankings, team_id: int, played: list[WcMatch]) -> float:
    deltas: list[float] = []
    for match in played:
        is_home = match.home_id == team_id
        supremacy = base_supremacy(wc, rankings, match)
        expected = supremacy if is_home else -supremacy
        assert match.home_goals is not None and match.away_goals is not None
        goals_for = match.home_goals if is_home else match.away_goals
        goals_against = match.away_goals if is_home else match.home_goals
        actual = float(goals_for - goals_against)
        deltas.append(_clamp(K_MOM * (actual - expected), CAP_MOM))
    return _mean(deltas)


def _lineup_delta(
    wc: WorldCup, rankings: Rankings, team_id: int, lineup: Lineup | None
) -> float:
    if lineup is None or not lineup.start_ids:
        return 0.0
    team = wc.teams[team_id]
    squad = sorted(
        (rankings.players.get(pid, _NEUTRAL) for pid in team.player_ids), reverse=True
    )
    squad_core = _mean(squad[:_SQUAD_CORE]) if squad else _NEUTRAL
    xi = _mean([rankings.players.get(pid, _NEUTRAL) for pid in lineup.start_ids])
    return _clamp(K_LU * (xi - squad_core), CAP_LU)


def _formation_lean(lineup: Lineup | None) -> tuple[float, float]:
    if lineup is None:
        return 0.0, 0.0
    parsed = parse_formation(lineup.formation)
    if parsed is None:
        return 0.0, 0.0
    defenders, forwards = parsed
    return FORM_ATTACK * (forwards - 3), FORM_DEFENSE * (defenders - 4)


def compute_adjustments(wc: WorldCup, rankings: Rankings) -> dict[int, TeamAdjustment]:
    played_by_team: dict[int, list[WcMatch]] = {}
    for match in wc.matches:
        if match.played:
            played_by_team.setdefault(match.home_id, []).append(match)
            played_by_team.setdefault(match.away_id, []).append(match)
    # Most recent lineup per team (lineups are appended in fixture order by refresh_live).
    lineup_by_team: dict[int, Lineup] = {lu.team_id: lu for lu in wc.lineups}

    out: dict[int, TeamAdjustment] = {}
    for team_id, played in played_by_team.items():
        lineup = lineup_by_team.get(team_id)
        momentum = _momentum(wc, rankings, team_id, played)
        lineup_delta = _lineup_delta(wc, rankings, team_id, lineup)
        attack_lean, defense_lean = _formation_lean(lineup)
        out[team_id] = TeamAdjustment(
            rating_delta=_clamp(momentum + lineup_delta, CAP_TOTAL),
            momentum=momentum,
            lineup=lineup_delta,
            attack_lean=attack_lean,
            defense_lean=defense_lean,
        )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/worldcup/test_adjust.py -v`
Expected: PASS (all six tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/worldcup/adjust.py tests/worldcup/test_adjust.py
git commit -m "feat(wc): add post-match team adjustments (momentum, lineup, formation)"
```

---

## Task 5: `predict_remaining` and adjustment-aware `_predict`

**Files:**
- Modify: `src/soccer/worldcup/predict.py`
- Test: `tests/worldcup/test_predict.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/worldcup/test_predict.py`:

```python
def test_predict_remaining_only_unplayed_and_shifts_lambda(sample_world_cup: WorldCup) -> None:
    from dataclasses import replace

    from soccer.worldcup.adjust import TeamAdjustment
    from soccer.worldcup.predict import predict_remaining

    rankings = rank_all(sample_world_cup)
    # Baseline (no adjustments) for fixture 9001.
    base = predict_remaining(sample_world_cup, rankings, {})
    assert len(base) == 1  # the single match is unplayed
    base_lambda_home = base[0].lambda_home

    # Boost England (home, id 1) -> its lambda_home should rise vs baseline.
    boosted = predict_remaining(
        sample_world_cup, rankings, {1: TeamAdjustment(rating_delta=5.0)}
    )
    assert boosted[0].lambda_home > base_lambda_home
    assert boosted[0].home_adjustment == 5.0

    # A played match drops out of the remaining set.
    played = replace(sample_world_cup.matches[0], home_goals=1, away_goals=0)
    wc_played = replace(sample_world_cup, matches=(played,))
    assert predict_remaining(wc_played, rank_all(wc_played), {}) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/worldcup/test_predict.py -k remaining -v`
Expected: FAIL with `ImportError: cannot import name 'predict_remaining'`.

- [ ] **Step 3: Add adjustment fields and a TYPE_CHECKING import**

In `src/soccer/worldcup/predict.py`, under the existing imports add:

```python
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from soccer.worldcup.adjust import TeamAdjustment
```

(The `Any` import already exists — keep a single `from typing import ...` line; merge if needed.)

Add two fields at the END of `MatchPrediction` (after `rationale`):

```python
    rationale: str
    home_adjustment: float = 0.0
    away_adjustment: float = 0.0
```

Add them to `to_dict` (after `"rationale"`):

```python
            "rationale": self.rationale,
            "home_adjustment": self.home_adjustment,
            "away_adjustment": self.away_adjustment,
```

- [ ] **Step 4: Make `_predict` adjustment-aware**

Replace the body of `_predict` in `src/soccer/worldcup/predict.py` with:

```python
def _predict(
    wc: WorldCup,
    rankings: Rankings,
    match: WcMatch,
    adjustments: dict[int, TeamAdjustment] | None = None,
) -> MatchPrediction:
    home = wc.teams[match.home_id]
    away = wc.teams[match.away_id]
    adj = adjustments or {}
    adj_h = adj.get(match.home_id)
    adj_a = adj.get(match.away_id)
    rd_h = adj_h.rating_delta if adj_h else 0.0
    rd_a = adj_a.rating_delta if adj_a else 0.0
    base_h = rankings.teams.get(match.home_id, 50.0) + rd_h
    base_a = rankings.teams.get(match.away_id, 50.0) + rd_a
    eff_h = _effective_rating(wc, match.home_id, base_h, is_home=True, venue=match.venue)
    eff_a = _effective_rating(wc, match.away_id, base_a, is_home=False, venue=match.venue)

    supremacy = SUPREMACY_PER_10 * (eff_h - eff_a) / 10.0
    atk_h = adj_h.attack_lean if adj_h else 0.0
    def_h = adj_h.defense_lean if adj_h else 0.0
    atk_a = adj_a.attack_lean if adj_a else 0.0
    def_a = adj_a.defense_lean if adj_a else 0.0
    lam_home = max(BASE_MATCH_GOALS / 2.0 + supremacy / 2.0 + atk_h - def_a, LAMBDA_FLOOR)
    lam_away = max(BASE_MATCH_GOALS / 2.0 - supremacy / 2.0 + atk_a - def_h, LAMBDA_FLOOR)

    matrix = _scoreline_matrix(lam_home, lam_away)
    p_home, p_draw, p_away = _outcome_probs(matrix)
    score_home, score_away = _modal_score(matrix)
    rationale = (
        f"Effective rating {eff_h:.1f} vs {eff_a:.1f} -> supremacy {supremacy:+.2f}; "
        f"xG {lam_home:.2f}-{lam_away:.2f}"
    )
    if rd_h or rd_a:
        rationale += f"; adj {rd_h:+.2f}/{rd_a:+.2f}"
    rationale += "."
    return MatchPrediction(
        fixture_id=match.fixture_id,
        matchday=match.matchday,
        group=match.group,
        kickoff=match.kickoff,
        home_id=match.home_id,
        away_id=match.away_id,
        home_name=home.name,
        away_name=away.name,
        lambda_home=round(lam_home, 3),
        lambda_away=round(lam_away, 3),
        score_home=score_home,
        score_away=score_away,
        p_home=round(p_home, 4),
        p_draw=round(p_draw, 4),
        p_away=round(p_away, 4),
        rationale=rationale,
        home_adjustment=round(rd_h, 3),
        away_adjustment=round(rd_a, 3),
    )
```

- [ ] **Step 5: Add `predict_remaining`**

Append to `src/soccer/worldcup/predict.py` (after `predict_group_stage`):

```python
def predict_remaining(
    wc: WorldCup,
    rankings: Rankings,
    adjustments: dict[int, TeamAdjustment],
) -> list[MatchPrediction]:
    """Predict only the not-yet-played matches, applying per-team adjustments."""
    ordered = sorted(
        (m for m in wc.matches if not m.played),
        key=lambda m: (m.matchday, m.group, m.fixture_id),
    )
    return [_predict(wc, rankings, m, adjustments) for m in ordered]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_predict.py -v`
Expected: PASS (new test plus all existing predict tests — `predict_group_stage` behavior is unchanged because it calls `_predict` without adjustments).

- [ ] **Step 7: Commit**

```bash
git add src/soccer/worldcup/predict.py tests/worldcup/test_predict.py
git commit -m "feat(wc): add predict_remaining with adjustment-aware scoreline model"
```

---

## Task 6: CLI — `wc refresh` and `wc predict --remaining`

**Files:**
- Modify: `src/soccer/worldcup/cli.py`
- Test: `tests/worldcup/test_cli.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/worldcup/test_cli.py`:

```python
def test_predict_remaining_writes_named_files(
    tmp_path: Path, sample_world_cup: WorldCup
) -> None:
    from dataclasses import replace

    from soccer.worldcup.cli import cmd_predict

    # Mark the only match as played so "remaining" has nothing to forecast but a result to show.
    played = replace(sample_world_cup.matches[0], home_goals=2, away_goals=0)
    wc = replace(sample_world_cup, matches=(played,))
    _write_dataset(tmp_path, wc)

    args = argparse.Namespace(
        remaining=True, out_dir=str(tmp_path / "perdictions"), name="after1st"
    )
    rc = cmd_predict(args, _config(tmp_path))
    assert rc == 0
    out_dir = tmp_path / "perdictions"
    payload = json.loads((out_dir / "after1st.json").read_text())
    assert set(payload) == {"predictions", "results", "adjustments"}
    report = (out_dir / "after1st.md").read_text()
    assert "Completed results" in report
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/worldcup/test_cli.py -k remaining -v`
Expected: FAIL (`cmd_predict` does not yet accept `remaining`/`out_dir`/`name`).

- [ ] **Step 3: Update imports and add output/report helpers**

In `src/soccer/worldcup/cli.py`, extend the imports:

```python
from dataclasses import asdict

from soccer.worldcup.adjust import TeamAdjustment, compute_adjustments
from soccer.worldcup.apifootball import ApiFootballClient, ApiFootballError, urllib_transport
from soccer.worldcup.live import refresh_live
from soccer.worldcup.predict import (
    MatchPrediction,
    predict_group_stage,
    predict_remaining,
)
```

Add helpers after `_report_path`:

```python
def _resolve_outputs(args: argparse.Namespace, config: AppConfig) -> tuple[Path, Path]:
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else _prediction_dir(config)
    name = getattr(args, "name", None) or "worldcup-2026-predictions"
    return out_dir / f"{name}.json", out_dir / f"{name}.md"


def _render_remaining_report(wc: WorldCup, predictions: list[MatchPrediction]) -> str:
    """Markdown: completed actual results first, then updated predictions per group/matchday."""
    lines = [
        "# FIFA 2026 World Cup — Updated Predictions (after Matchday 1)",
        "",
        "Actual results so far, then predicted result and final score for every remaining",
        "group-stage match. Percentages are home win / draw / away win.",
        "",
        "## Completed results",
        "",
    ]
    played = sorted((m for m in wc.matches if m.played), key=lambda m: (m.matchday, m.kickoff))
    for match in played:
        home = wc.teams[match.home_id].name
        away = wc.teams[match.away_id].name
        lines.append(
            f"- `MD{match.matchday}` **{home} {match.home_goals}-{match.away_goals} {away}**"
        )
    lines.append("")
    by_group: dict[str, dict[int, list[MatchPrediction]]] = {}
    for pred in predictions:
        by_group.setdefault(pred.group, {}).setdefault(pred.matchday, []).append(pred)
    for group in sorted(by_group):
        lines += [f"## {group}", ""]
        for matchday in sorted(by_group[group]):
            lines += [f"### Matchday {matchday}", ""]
            for pred in sorted(by_group[group][matchday], key=lambda p: p.kickoff):
                kickoff = pred.kickoff.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
                lines.append(
                    f"- `{kickoff}`  **{pred.home_name} {pred.score_home}-{pred.score_away} "
                    f"{pred.away_name}**  "
                    f"(W {pred.p_home:.0%} / D {pred.p_draw:.0%} / L {pred.p_away:.0%})"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"
```

- [ ] **Step 4: Rewrite `cmd_predict` and add `cmd_refresh`**

Replace `cmd_predict` in `src/soccer/worldcup/cli.py` with:

```python
def cmd_predict(args: argparse.Namespace, config: AppConfig) -> int:
    wc = load_dataset(_dataset_path(config))
    rankings = rank_all(wc)
    remaining = getattr(args, "remaining", False)
    json_path, report_path = _resolve_outputs(args, config)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if remaining:
        adjustments = compute_adjustments(wc, rankings)
        predictions = predict_remaining(wc, rankings, adjustments)
        results = [
            {
                "fixture_id": m.fixture_id,
                "matchday": m.matchday,
                "group": m.group,
                "home_name": wc.teams[m.home_id].name,
                "away_name": wc.teams[m.away_id].name,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
            }
            for m in sorted(
                (m for m in wc.matches if m.played), key=lambda m: (m.matchday, m.kickoff)
            )
        ]
        payload: object = {
            "predictions": [p.to_dict() for p in predictions],
            "results": results,
            "adjustments": {str(tid): asdict(a) for tid, a in adjustments.items()},
        }
        report = _render_remaining_report(wc, predictions)
    else:
        predictions = predict_group_stage(wc, rankings)
        payload = [p.to_dict() for p in predictions]
        report = _render_report(predictions)
    _print_predictions(predictions)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print(f"\nwrote {len(predictions)} predictions -> {json_path}")
    print(f"wrote readable report -> {report_path}")
    return 0


def cmd_refresh(args: argparse.Namespace, config: AppConfig) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not config.api_football_key:
        print("SOCCER_API_FOOTBALL_KEY is not set; cannot refresh live data", flush=True)
        return 1
    wc = load_dataset(_dataset_path(config))
    client = ApiFootballClient(
        config.api_football_key,
        base_url=config.api_football_base_url,
        transport=urllib_transport(timeout=30.0),
        cache=JsonCache(config.data_dir / "api"),
        throttle_seconds=args.throttle,
    )
    try:
        updated = refresh_live(wc, client)
    except ApiFootballError as exc:
        print(f"refresh failed: {exc}")
        return 1
    path = _dataset_path(config)
    path.write_text(json.dumps(updated.to_dict()), encoding="utf-8")
    played = sum(1 for m in updated.matches if m.played)
    print(f"refreshed {played} played matches, {len(updated.lineups)} lineups -> {path}")
    return 0
```

- [ ] **Step 5: Register the new CLI arguments and subcommand**

In `add_wc_subparser`, extend the `predict` parser and add the `refresh` parser:

```python
    p_predict = wc_sub.add_parser("predict", help="predict group-stage scorelines")
    p_predict.add_argument(
        "--remaining",
        action="store_true",
        help="forecast only unplayed matches, using actual results + lineups",
    )
    p_predict.add_argument("--out-dir", default=None, help="output directory for the files")
    p_predict.add_argument("--name", default=None, help="basename for the .json/.md files")
    p_predict.set_defaults(func=cmd_predict)

    p_refresh = wc_sub.add_parser("refresh", help="merge live results + lineups into the dataset")
    p_refresh.add_argument("--throttle", type=float, default=0.02, help="seconds between calls")
    p_refresh.set_defaults(func=cmd_refresh)
```

(Remove the old `p_predict` block that this replaces.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_cli.py -v`
Expected: PASS (new test plus the existing `test_predict_writes_file`, whose default-path list output is preserved).

- [ ] **Step 7: Commit**

```bash
git add src/soccer/worldcup/cli.py tests/worldcup/test_cli.py
git commit -m "feat(wc): add 'wc refresh' and 'wc predict --remaining' commands"
```

---

## Task 7: Generate live outputs and finalize

**Files:**
- Modify: `README.md`
- Generated: `data/worldcup-2026.json` (refreshed), `perdictions/worldcup-2026-predictions-after1st-group.{json,md}`

- [ ] **Step 1: Document the new commands in README**

In `README.md`, in the World Cup section, add after the `predict` description:

```markdown
- `soccer wc refresh` — pull live matchday results and lineups into the cached dataset
  (incremental; reuses the static player/club data, needs `SOCCER_API_FOOTBALL_KEY`).
- `soccer wc predict --remaining --out-dir perdictions --name worldcup-2026-predictions-after1st-group`
  — re-forecast only the unplayed matches, folding in actual results, starting XIs, and
  formations. Writes a `.json` (predictions + results + per-team adjustments) and a `.md` report.
```

- [ ] **Step 2: Commit the docs**

```bash
git add README.md
git commit -m "docs: document wc refresh and predict --remaining"
```

- [ ] **Step 3: Run the full check suite**

Run: `make check`
Expected: ruff format clean, ruff lint clean, mypy clean, pytest all pass with coverage.
Fix any issues, then re-run until green.

- [ ] **Step 4: Refresh live data (network; needs the API key in .env)**

Run:

```bash
set -a; source .env; set +a
python -m soccer wc refresh
```

Expected: prints `refreshed N played matches, M lineups -> data/worldcup-2026.json` (N ≈ 24+).

- [ ] **Step 5: Generate the after-MD1 predictions**

Run:

```bash
python -m soccer wc predict --remaining --out-dir perdictions --name worldcup-2026-predictions-after1st-group
```

Expected: writes `perdictions/worldcup-2026-predictions-after1st-group.json` and `.md`.

- [ ] **Step 6: Sanity-check the output**

Run: `head -40 perdictions/worldcup-2026-predictions-after1st-group.md`
Expected: a "Completed results" section listing real MD1 scores, then per-group predicted
remaining matches. Confirm no played match appears in the predictions and the adjustments look
bounded (|rating_delta| ≤ 5).

- [ ] **Step 7: Commit the generated artifacts**

```bash
git add perdictions/ data/worldcup-2026.json
git commit -m "data(wc): refreshed dataset + after-MD1 remaining-game predictions"
```

---

## Self-Review Notes

- **Spec coverage:** Lineup entity (T1), force_refresh (T2), live merge (T3), adjustments incl.
  momentum/lineup/formation + caps (T4), predict_remaining + new fields (T5), CLI refresh +
  --remaining with actuals-in-report (T6), README + live generation (T7). All spec sections map
  to a task.
- **Circular import avoided:** `predict.py` imports `TeamAdjustment` only under `TYPE_CHECKING`;
  `adjust.py` imports concrete functions from `predict.py`. CLI wires both at runtime.
- **Backward compatibility:** new `MatchPrediction` fields have defaults; default `wc predict`
  still writes a JSON list, keeping `test_predict_writes_file` valid.
- **Type/name consistency:** `compute_adjustments`, `predict_remaining`, `refresh_live`,
  `force_refresh`, `_resolve_outputs`, `_render_remaining_report` used identically across tasks.
```
