# Single-Match Pre-Match Preview Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `soccer wc card <fixture_id>` command that previews a single upcoming match — coach, starting XI with formation, likely subs for both teams, and a lineup-aware prediction — and writes it as a PDF (and JSON).

**Architecture:** Three new isolated modules (`lineup.py` projection, `card.py` card model, `cardpdf.py` PDF renderer) plus small extensions to `adjust.py`, `predict.py`, `live.py`, and `cli.py`. The Poisson scoreline core in `predict.py` is reused unchanged; the "more accurate" result comes from feeding the upcoming match's confirmed-or-projected XI and formation, plus tournament momentum, through the existing adjustment layer.

**Tech Stack:** Python 3.11+, stdlib + `certifi`; `reportlab` for PDF behind a new optional extra `[pdf]`. Tooling: ruff (format+lint), mypy strict, pytest.

## Global Constraints

- Target Python 3.11+, `src/` layout; importable package is `soccer`.
- Minimal dependencies: no new hard dependency. `reportlab>=4` is added ONLY as the optional `[pdf]` extra (and to `[dev]` so tests can exercise it); it must be imported lazily inside the renderer so the rest of the package imports without it.
- Dependency injection for side effects: no network/time/randomness in library code or tests; tests use tmp dirs + fakes. The only networked path is `--refresh` via the injected `ApiFootballClient`.
- No import-time side effects; read config/env at the CLI boundary only.
- Ruff is canonical for format+lint; mypy strict must pass (`mypy src tests`); line length 100.
- Every behavior change ships with tests, including error paths.
- `group` is a display label only — never an input to projection or prediction (keeps the knockout-stage next phase cheap).

---

### Task 1: Lineup projection module (`lineup.py`)

**Files:**
- Create: `src/soccer/worldcup/lineup.py`
- Test: `tests/worldcup/test_lineup.py`

**Interfaces:**
- Consumes: `WorldCup`, `Lineup`, `Player` from `soccer.worldcup.entities`; `Rankings` from `soccer.worldcup.ranking` (uses `rankings.players: dict[int, float]`); `WorldCup.squad(team_id) -> list[Player]`. `Player.position` is one of `"Goalkeeper" | "Defender" | "Midfielder" | "Attacker"`.
- Produces:
  - `class ProjectedLineup` (frozen dataclass) with fields `team_id: int`, `formation: str`, `start_ids: tuple[int, ...]`, `sub_ids: tuple[int, ...]`, `source: str` (`"confirmed" | "prior" | "projected"`), `source_matchday: int | None`.
  - `formation_slots(formation: str) -> tuple[int, int, int]` → `(defenders, midfielders, forwards)`.
  - `preferred_formation(wc: WorldCup, team_id: int) -> str`.
  - `project_lineup(wc: WorldCup, rankings: Rankings, team_id: int, fixture_id: int) -> ProjectedLineup`.
  - Module constants `DEFAULT_FORMATION = "4-3-3"`, `SUB_COUNT = 7`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/worldcup/test_lineup.py
from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from soccer.worldcup.entities import Lineup, WcMatch, WorldCup
from soccer.worldcup.lineup import (
    DEFAULT_FORMATION,
    formation_slots,
    preferred_formation,
    project_lineup,
)
from soccer.worldcup.ranking import rank_all


def test_formation_slots_parses_defenders_mids_forwards() -> None:
    assert formation_slots("4-3-3") == (4, 3, 3)
    assert formation_slots("4-2-3-1") == (4, 5, 1)
    assert formation_slots("nonsense") == (4, 3, 3)


def test_preferred_formation_picks_most_common(sample_world_cup: WorldCup) -> None:
    wc = replace(
        sample_world_cup,
        lineups=(
            Lineup(9001, 1, "3-5-2", (1, 2), ()),
            Lineup(9002, 1, "3-5-2", (1, 2), ()),
            Lineup(9003, 1, "4-4-2", (1, 2), ()),
        ),
    )
    assert preferred_formation(wc, 1) == "3-5-2"
    assert preferred_formation(wc, 999) == DEFAULT_FORMATION


def test_project_lineup_uses_confirmed_lineup(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    wc = replace(sample_world_cup, lineups=(Lineup(9001, 1, "3-5-2", (1, 2), (3,)),))
    lu = project_lineup(wc, rankings, 1, 9001)
    assert lu.source == "confirmed"
    assert lu.formation == "3-5-2"
    assert lu.start_ids == (1, 2)
    assert lu.sub_ids == (3,)
    assert lu.source_matchday is None


def test_project_lineup_falls_back_to_prior_matchday(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    md2 = WcMatch(
        fixture_id=9002,
        matchday=2,
        group="Group A",
        home_id=1,
        away_id=2,
        kickoff=datetime(2026, 6, 18, 19, 0, tzinfo=UTC),
        venue="venue",
        home_goals=None,
        away_goals=None,
    )
    wc = replace(
        sample_world_cup,
        matches=sample_world_cup.matches + (md2,),
        lineups=(Lineup(9001, 1, "4-4-2", (1, 2), ()),),
    )
    lu = project_lineup(wc, rankings, 1, 9002)
    assert lu.source == "prior"
    assert lu.source_matchday == 1
    assert lu.formation == "4-4-2"
    assert lu.start_ids == (1, 2)


def test_project_lineup_projects_from_squad_on_matchday_one(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    lu = project_lineup(sample_world_cup, rankings, 1, 9001)
    assert lu.source == "projected"
    assert lu.formation == DEFAULT_FORMATION
    # team 1's squad is players (1, 2); both start, no subs left.
    assert set(lu.start_ids) == {1, 2}
    assert lu.sub_ids == ()


def test_project_lineup_unknown_fixture_raises(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    with pytest.raises(ValueError, match="not found"):
        project_lineup(sample_world_cup, rankings, 1, 123456)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/worldcup/test_lineup.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soccer.worldcup.lineup'`.

- [ ] **Step 3: Write the implementation**

```python
# src/soccer/worldcup/lineup.py
"""Project the most likely starting XI, subs, and formation for an upcoming match.

Precedence: an official lineup already attached to this fixture ("confirmed"); else the
team's most recent lineup from an earlier matchday in this tournament ("prior"); else a
squad-based projection using the coach-preferred formation ("projected"). The result feeds
both the printed card and the lineup-aware prediction, so it carries a ``source`` provenance
label rather than silently guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from soccer.worldcup.entities import Lineup, Player, WorldCup
from soccer.worldcup.ranking import Rankings

DEFAULT_FORMATION = "4-3-3"
SUB_COUNT = 7
_NEUTRAL = 50.0
_POSITION_GROUP = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Attacker": "FWD",
    "G": "GK",
    "D": "DEF",
    "M": "MID",
    "F": "FWD",
}


@dataclass(frozen=True)
class ProjectedLineup:
    team_id: int
    formation: str
    start_ids: tuple[int, ...]
    sub_ids: tuple[int, ...]
    source: str  # "confirmed" | "prior" | "projected"
    source_matchday: int | None = None


def formation_slots(formation: str) -> tuple[int, int, int]:
    """Return (defenders, midfielders, forwards) from e.g. '4-3-3' or '4-2-3-1'.

    The goalkeeper is implicit (always 1). Unparseable input falls back to a 4-3-3 shape.
    """
    try:
        nums = [int(part) for part in formation.split("-")]
    except ValueError:
        nums = []
    if len(nums) < 2:
        return (4, 3, 3)
    defenders = nums[0]
    forwards = nums[-1]
    midfielders = sum(nums[1:-1])
    return (defenders, midfielders, forwards)


def _position_group(position: str) -> str:
    return _POSITION_GROUP.get(position, "MID")


def preferred_formation(wc: WorldCup, team_id: int) -> str:
    counts: dict[str, int] = {}
    for lineup in wc.lineups:
        if lineup.team_id == team_id and lineup.formation:
            counts[lineup.formation] = counts.get(lineup.formation, 0) + 1
    if not counts:
        return DEFAULT_FORMATION
    return max(counts, key=lambda formation: counts[formation])


def _project_xi(
    wc: WorldCup, rankings: Rankings, team_id: int, formation: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    squad: list[Player] = sorted(
        wc.squad(team_id),
        key=lambda p: rankings.players.get(p.id, _NEUTRAL),
        reverse=True,
    )
    defenders, midfielders, forwards = formation_slots(formation)
    need = {"GK": 1, "DEF": defenders, "MID": midfielders, "FWD": forwards}
    by_group: dict[str, list[Player]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for player in squad:
        by_group[_position_group(player.position)].append(player)

    chosen: list[int] = []
    chosen_ids: set[int] = set()
    for group, count in need.items():
        for player in by_group[group][:count]:
            chosen.append(player.id)
            chosen_ids.add(player.id)
    # Backfill to 11 from the best remaining players when a position group is short.
    if len(chosen) < 11:
        for player in squad:
            if player.id not in chosen_ids:
                chosen.append(player.id)
                chosen_ids.add(player.id)
                if len(chosen) >= 11:
                    break

    start_ids = tuple(chosen[:11])
    start_set = set(start_ids)
    sub_ids = tuple(p.id for p in squad if p.id not in start_set)[:SUB_COUNT]
    return start_ids, sub_ids


def project_lineup(
    wc: WorldCup, rankings: Rankings, team_id: int, fixture_id: int
) -> ProjectedLineup:
    target = next((m for m in wc.matches if m.fixture_id == fixture_id), None)
    if target is None:
        raise ValueError(f"fixture {fixture_id} not found in dataset")

    for lineup in wc.lineups:
        if lineup.fixture_id == fixture_id and lineup.team_id == team_id:
            return ProjectedLineup(
                team_id=team_id,
                formation=lineup.formation or DEFAULT_FORMATION,
                start_ids=lineup.start_ids,
                sub_ids=lineup.sub_ids,
                source="confirmed",
                source_matchday=None,
            )

    matchday_by_fixture = {m.fixture_id: m.matchday for m in wc.matches}
    prior: Lineup | None = None
    prior_matchday = -1
    for lineup in wc.lineups:
        if lineup.team_id != team_id:
            continue
        matchday = matchday_by_fixture.get(lineup.fixture_id, -1)
        if matchday < target.matchday and matchday > prior_matchday:
            prior, prior_matchday = lineup, matchday
    if prior is not None:
        return ProjectedLineup(
            team_id=team_id,
            formation=prior.formation or DEFAULT_FORMATION,
            start_ids=prior.start_ids,
            sub_ids=prior.sub_ids,
            source="prior",
            source_matchday=prior_matchday,
        )

    formation = preferred_formation(wc, team_id)
    start_ids, sub_ids = _project_xi(wc, rankings, team_id, formation)
    return ProjectedLineup(
        team_id=team_id,
        formation=formation,
        start_ids=start_ids,
        sub_ids=sub_ids,
        source="projected",
        source_matchday=None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_lineup.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Lint, format, typecheck**

Run: `ruff format src/soccer/worldcup/lineup.py tests/worldcup/test_lineup.py && ruff check src/soccer/worldcup/lineup.py tests/worldcup/test_lineup.py && mypy src tests`
Expected: all pass, no errors.

- [ ] **Step 6: Commit**

```bash
git add src/soccer/worldcup/lineup.py tests/worldcup/test_lineup.py
git commit -m "feat(wc): add starting-XI projection for a single fixture"
```

---

### Task 2: Lineup-aware adjustment and single-match prediction

**Files:**
- Modify: `src/soccer/worldcup/adjust.py`
- Modify: `src/soccer/worldcup/predict.py`
- Test: `tests/worldcup/test_card_predict.py`

**Interfaces:**
- Consumes: `ProjectedLineup` from Task 1; existing private helpers `_momentum`, `_lineup_delta`, `_formation_lean`, `_clamp`, `CAP_TOTAL`, `TeamAdjustment` in `adjust.py`; existing `_predict`, `_scoreline_matrix`, `predict_match`, `MatchPrediction` in `predict.py`.
- Produces:
  - `adjust.adjustment_for_match(wc: WorldCup, rankings: Rankings, team_id: int, lineup: _LineupLike | None) -> TeamAdjustment`, where `_LineupLike` is a `typing.Protocol` with `formation: str` and `start_ids: tuple[int, ...]` (satisfied by both `Lineup` and `ProjectedLineup`).
  - `predict.predict_one(wc: WorldCup, rankings: Rankings, fixture_id: int, home_lineup: Lineup | ProjectedLineup | None, away_lineup: Lineup | ProjectedLineup | None) -> MatchPrediction`.
  - `predict.top_scorelines(lambda_home: float, lambda_away: float, n: int = 3) -> list[tuple[int, int, float]]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/worldcup/test_card_predict.py
from __future__ import annotations

from soccer.worldcup.adjust import adjustment_for_match
from soccer.worldcup.entities import WorldCup
from soccer.worldcup.lineup import ProjectedLineup
from soccer.worldcup.predict import predict_match, predict_one, top_scorelines
from soccer.worldcup.ranking import rank_all


def test_adjustment_for_match_reflects_formation_lean(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    attacking = ProjectedLineup(1, "3-3-4", (1, 2), (), "projected", None)
    adj = adjustment_for_match(sample_world_cup, rankings, 1, attacking)
    # 3-3-4 -> 4 forwards (attack lean up), 3 defenders (defense lean down).
    assert adj.attack_lean > 0.0
    assert adj.defense_lean < 0.0


def test_predict_one_degrades_to_baseline_without_lineups(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    baseline = predict_match(sample_world_cup, rankings, 9001)
    one = predict_one(sample_world_cup, rankings, 9001, None, None)
    # No played matches + no lineup => zero adjustment => identical to the baseline forecast.
    assert one.lambda_home == baseline.lambda_home
    assert one.lambda_away == baseline.lambda_away
    assert one.p_home == baseline.p_home


def test_predict_one_attacking_home_raises_home_xg(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    baseline = predict_match(sample_world_cup, rankings, 9001)
    attacking = ProjectedLineup(1, "3-3-4", (1, 2), (), "projected", None)
    one = predict_one(sample_world_cup, rankings, 9001, attacking, None)
    assert one.lambda_home > baseline.lambda_home


def test_top_scorelines_is_sorted_and_bounded() -> None:
    tops = top_scorelines(1.4, 1.1, n=3)
    assert len(tops) == 3
    probs = [p for _, _, p in tops]
    assert probs == sorted(probs, reverse=True)
    assert 0.0 < sum(probs) < 1.0
    home, away, _ = tops[0]
    assert isinstance(home, int) and isinstance(away, int)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/worldcup/test_card_predict.py -v`
Expected: FAIL — `ImportError: cannot import name 'predict_one'` / `'adjustment_for_match'`.

- [ ] **Step 3a: Extend `adjust.py`**

Add the `Protocol` import at the top of `src/soccer/worldcup/adjust.py` (the `from __future__ import annotations` line stays first):

```python
from typing import Protocol
```

Change the two existing helper signatures so they accept anything lineup-shaped. Replace the signature line of `_lineup_delta`:

```python
def _lineup_delta(
    wc: WorldCup, rankings: Rankings, team_id: int, lineup: _LineupLike | None
) -> float:
```

Replace the signature line of `_formation_lean`:

```python
def _formation_lean(lineup: _LineupLike | None) -> tuple[float, float]:
```

Add the protocol definition just above `_clamp` (after the module constants):

```python
class _LineupLike(Protocol):
    formation: str
    start_ids: tuple[int, ...]
```

Append the new public function at the end of `adjust.py`:

```python
def adjustment_for_match(
    wc: WorldCup, rankings: Rankings, team_id: int, lineup: _LineupLike | None
) -> TeamAdjustment:
    """Adjustment for an upcoming match: momentum from played games plus this match's XI.

    Unlike :func:`compute_adjustments` (which uses each team's last finished lineup), this is
    driven by the confirmed-or-projected lineup for the specific fixture being previewed.
    """
    played = [
        m for m in wc.matches if m.played and team_id in (m.home_id, m.away_id)
    ]
    momentum = _momentum(wc, rankings, team_id, played)
    lineup_delta = _lineup_delta(wc, rankings, team_id, lineup)
    attack_lean, defense_lean = _formation_lean(lineup)
    return TeamAdjustment(
        rating_delta=_clamp(momentum + lineup_delta, CAP_TOTAL),
        momentum=momentum,
        lineup=lineup_delta,
        attack_lean=attack_lean,
        defense_lean=defense_lean,
    )
```

- [ ] **Step 3b: Extend `predict.py`**

Add an import of `ProjectedLineup` and `Lineup` near the existing entity import in `src/soccer/worldcup/predict.py`. The current line is:

```python
from soccer.worldcup.entities import WcMatch, WorldCup
```

Replace it with:

```python
from soccer.worldcup.entities import Lineup, WcMatch, WorldCup
from soccer.worldcup.lineup import ProjectedLineup
```

Append both new functions at the end of `predict.py`:

```python
def top_scorelines(
    lambda_home: float, lambda_away: float, n: int = 3
) -> list[tuple[int, int, float]]:
    """Return the ``n`` most likely exact scorelines as (home_goals, away_goals, prob)."""
    matrix = _scoreline_matrix(lambda_home, lambda_away)
    cells = [(i, j, p) for i, row in enumerate(matrix) for j, p in enumerate(row)]
    cells.sort(key=lambda cell: cell[2], reverse=True)
    return cells[:n]


def predict_one(
    wc: WorldCup,
    rankings: Rankings,
    fixture_id: int,
    home_lineup: Lineup | ProjectedLineup | None,
    away_lineup: Lineup | ProjectedLineup | None,
) -> MatchPrediction:
    """Lineup-aware forecast for a single fixture (confirmed or projected lineups)."""
    # Imported here to avoid an import cycle: adjust imports predict at module load.
    from soccer.worldcup.adjust import adjustment_for_match

    match = next((m for m in wc.matches if m.fixture_id == fixture_id), None)
    if match is None:
        raise ValueError(f"fixture {fixture_id} not found in dataset")
    adjustments = {
        match.home_id: adjustment_for_match(wc, rankings, match.home_id, home_lineup),
        match.away_id: adjustment_for_match(wc, rankings, match.away_id, away_lineup),
    }
    return _predict(wc, rankings, match, adjustments)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_card_predict.py tests/worldcup/test_adjust.py tests/worldcup/test_predict.py -v`
Expected: PASS (new tests pass; existing adjust/predict tests still pass).

- [ ] **Step 5: Lint, format, typecheck**

Run: `ruff format src/soccer/worldcup/adjust.py src/soccer/worldcup/predict.py tests/worldcup/test_card_predict.py && ruff check src/soccer/worldcup/adjust.py src/soccer/worldcup/predict.py tests/worldcup/test_card_predict.py && mypy src tests`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/soccer/worldcup/adjust.py src/soccer/worldcup/predict.py tests/worldcup/test_card_predict.py
git commit -m "feat(wc): add lineup-aware single-match prediction"
```

---

### Task 3: Match card model (`card.py`)

**Files:**
- Create: `src/soccer/worldcup/card.py`
- Test: `tests/worldcup/test_card.py`

**Interfaces:**
- Consumes: `WorldCup`, `Player`, `Coach` from entities; `Rankings`; `project_lineup`/`ProjectedLineup` (Task 1); `predict_one`/`top_scorelines`/`MatchPrediction` (Task 2).
- Produces:
  - `class PlayerLine` (frozen): `player_id: int`, `name: str`, `position: str`, `rating: float`; `to_dict()`.
  - `class TeamCard` (frozen): `team_id: int`, `name: str`, `coach_name: str | None`, `coach_record: tuple[int, int, int] | None`, `formation: str`, `starters: tuple[PlayerLine, ...]`, `subs: tuple[PlayerLine, ...]`, `source: str`, `source_matchday: int | None`; `to_dict()`.
  - `class MatchCard` (frozen): `fixture_id: int`, `group: str`, `kickoff: datetime`, `venue: str`, `home: TeamCard`, `away: TeamCard`, `prediction: MatchPrediction`, `top_scorelines: tuple[tuple[int, int, float], ...]`; `to_dict()`.
  - `build_card(wc: WorldCup, rankings: Rankings, fixture_id: int) -> MatchCard`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/worldcup/test_card.py
from __future__ import annotations

import pytest

from soccer.worldcup.card import build_card
from soccer.worldcup.entities import WorldCup
from soccer.worldcup.ranking import rank_all


def test_build_card_assembles_both_teams(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    card = build_card(sample_world_cup, rankings, 9001)
    assert card.fixture_id == 9001
    assert card.group == "Group A"
    assert card.home.name == "England"
    assert card.away.name == "Mexico"
    assert card.home.source == "projected"  # no lineups in the sample dataset
    assert card.home.coach_name == "Strong Coach"
    assert card.home.coach_record == (8, 1, 1)
    assert len(card.home.starters) >= 1
    assert len(card.top_scorelines) == 3
    assert card.prediction.fixture_id == 9001


def test_build_card_to_dict_has_expected_keys(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    card = build_card(sample_world_cup, rankings, 9001)
    data = card.to_dict()
    assert {"fixture_id", "group", "kickoff", "venue", "home", "away", "prediction"} <= set(data)
    assert {"name", "formation", "starters", "subs", "source"} <= set(data["home"])
    assert {"player_id", "name", "position", "rating"} <= set(data["home"]["starters"][0])


def test_build_card_unknown_fixture_raises(sample_world_cup: WorldCup) -> None:
    rankings = rank_all(sample_world_cup)
    with pytest.raises(ValueError, match="not found"):
        build_card(sample_world_cup, rankings, 4242)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/worldcup/test_card.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soccer.worldcup.card'`.

- [ ] **Step 3: Write the implementation**

```python
# src/soccer/worldcup/card.py
"""Assemble a single-match preview card: lineups, coaches, and a lineup-aware prediction.

Pure and offline. ``build_card`` projects each side's lineup, runs the lineup-aware forecast,
and packages everything (including a JSON-friendly ``to_dict``) for the PDF renderer and the
``wc card`` command.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from soccer.worldcup.entities import WorldCup
from soccer.worldcup.lineup import ProjectedLineup, project_lineup
from soccer.worldcup.predict import MatchPrediction, predict_one, top_scorelines
from soccer.worldcup.ranking import Rankings

_NEUTRAL = 50.0


@dataclass(frozen=True)
class PlayerLine:
    player_id: int
    name: str
    position: str
    rating: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "position": self.position,
            "rating": self.rating,
        }


@dataclass(frozen=True)
class TeamCard:
    team_id: int
    name: str
    coach_name: str | None
    coach_record: tuple[int, int, int] | None
    formation: str
    starters: tuple[PlayerLine, ...]
    subs: tuple[PlayerLine, ...]
    source: str
    source_matchday: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "name": self.name,
            "coach_name": self.coach_name,
            "coach_record": list(self.coach_record) if self.coach_record else None,
            "formation": self.formation,
            "starters": [p.to_dict() for p in self.starters],
            "subs": [p.to_dict() for p in self.subs],
            "source": self.source,
            "source_matchday": self.source_matchday,
        }


@dataclass(frozen=True)
class MatchCard:
    fixture_id: int
    group: str
    kickoff: datetime
    venue: str
    home: TeamCard
    away: TeamCard
    prediction: MatchPrediction
    top_scorelines: tuple[tuple[int, int, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "group": self.group,
            "kickoff": self.kickoff.isoformat(),
            "venue": self.venue,
            "home": self.home.to_dict(),
            "away": self.away.to_dict(),
            "prediction": self.prediction.to_dict(),
            "top_scorelines": [list(s) for s in self.top_scorelines],
        }


def _player_line(wc: WorldCup, rankings: Rankings, player_id: int) -> PlayerLine:
    player = wc.players.get(player_id)
    rating = round(rankings.players.get(player_id, _NEUTRAL), 1)
    if player is None:
        return PlayerLine(player_id=player_id, name=f"#{player_id}", position="?", rating=rating)
    return PlayerLine(
        player_id=player_id,
        name=player.name,
        position=player.position,
        rating=rating,
    )


def _team_card(
    wc: WorldCup, rankings: Rankings, team_id: int, lineup: ProjectedLineup
) -> TeamCard:
    team = wc.teams[team_id]
    coach = wc.coaches.get(team.coach_id) if team.coach_id is not None else None
    coach_name = coach.name if coach else None
    coach_record = (coach.wins, coach.draws, coach.losses) if coach else None
    starters = tuple(_player_line(wc, rankings, pid) for pid in lineup.start_ids)
    subs = tuple(_player_line(wc, rankings, pid) for pid in lineup.sub_ids)
    return TeamCard(
        team_id=team_id,
        name=team.name,
        coach_name=coach_name,
        coach_record=coach_record,
        formation=lineup.formation,
        starters=starters,
        subs=subs,
        source=lineup.source,
        source_matchday=lineup.source_matchday,
    )


def build_card(wc: WorldCup, rankings: Rankings, fixture_id: int) -> MatchCard:
    match = next((m for m in wc.matches if m.fixture_id == fixture_id), None)
    if match is None:
        raise ValueError(f"fixture {fixture_id} not found in dataset")
    home_lineup = project_lineup(wc, rankings, match.home_id, fixture_id)
    away_lineup = project_lineup(wc, rankings, match.away_id, fixture_id)
    prediction = predict_one(wc, rankings, fixture_id, home_lineup, away_lineup)
    tops = tuple(top_scorelines(prediction.lambda_home, prediction.lambda_away, 3))
    return MatchCard(
        fixture_id=fixture_id,
        group=match.group,
        kickoff=match.kickoff,
        venue=match.venue,
        home=_team_card(wc, rankings, match.home_id, home_lineup),
        away=_team_card(wc, rankings, match.away_id, away_lineup),
        prediction=prediction,
        top_scorelines=tops,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_card.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint, format, typecheck**

Run: `ruff format src/soccer/worldcup/card.py tests/worldcup/test_card.py && ruff check src/soccer/worldcup/card.py tests/worldcup/test_card.py && mypy src tests`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/soccer/worldcup/card.py tests/worldcup/test_card.py
git commit -m "feat(wc): add single-match card model and builder"
```

---

### Task 4: PDF renderer (`cardpdf.py`) + `[pdf]` extra

**Files:**
- Modify: `pyproject.toml` (add `[pdf]` optional extra; add `reportlab` to `[dev]`)
- Create: `src/soccer/worldcup/cardpdf.py`
- Test: `tests/worldcup/test_cardpdf.py`

**Interfaces:**
- Consumes: `MatchCard`, `TeamCard`, `PlayerLine` from Task 3.
- Produces: `render_card_pdf(card: MatchCard, path: Path) -> None`. Writes a PDF to `path`. Raises `RuntimeError` with an install hint if `reportlab` is not importable.

- [ ] **Step 1: Add the optional dependency in `pyproject.toml`**

Replace the existing optional-dependencies block:

```toml
[project.optional-dependencies]
dev = ["ruff>=0.5", "mypy>=1.10", "pytest>=8", "pytest-cov>=5", "pre-commit>=3"]
```

with:

```toml
[project.optional-dependencies]
pdf = ["reportlab>=4"]
dev = ["ruff>=0.5", "mypy>=1.10", "pytest>=8", "pytest-cov>=5", "pre-commit>=3", "reportlab>=4"]
```

- [ ] **Step 2: Install the new dev dependency**

Run: `python -m pip install -e ".[dev]"`
Expected: `reportlab` installs successfully.

- [ ] **Step 3: Write the failing test**

```python
# tests/worldcup/test_cardpdf.py
from __future__ import annotations

from pathlib import Path

import pytest

from soccer.worldcup.card import build_card
from soccer.worldcup.cardpdf import render_card_pdf
from soccer.worldcup.entities import WorldCup
from soccer.worldcup.ranking import rank_all


def test_render_card_pdf_writes_a_pdf_file(tmp_path: Path, sample_world_cup: WorldCup) -> None:
    pytest.importorskip("reportlab")
    card = build_card(sample_world_cup, rank_all(sample_world_cup), 9001)
    out = tmp_path / "card-9001.pdf"
    render_card_pdf(card, out)
    data = out.read_bytes()
    assert data[:4] == b"%PDF"
    assert len(data) > 500
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/worldcup/test_cardpdf.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'soccer.worldcup.cardpdf'`.

- [ ] **Step 5: Write the implementation**

```python
# src/soccer/worldcup/cardpdf.py
"""Render a :class:`~soccer.worldcup.card.MatchCard` to a one-page PDF.

``reportlab`` is imported lazily inside :func:`render_card_pdf` so the rest of the package
imports without it; it is only required when a PDF is actually requested (the ``[pdf]`` extra).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soccer.worldcup.card import MatchCard, TeamCard

_INSTALL_HINT = "PDF output requires reportlab; install with: pip install 'soccer[pdf]'"


def render_card_pdf(card: MatchCard, path: Path) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(_INSTALL_HINT) from exc

    width, height = A4
    pdf = canvas.Canvas(str(path), pagesize=A4)
    left = 18 * mm
    right = width - 18 * mm
    cursor = height - 20 * mm

    def line(text: str, *, size: int = 10, gap: float = 5.2, x: float = left) -> None:
        nonlocal cursor
        pdf.setFont("Helvetica", size)
        pdf.drawString(x, cursor, text)
        cursor -= gap * mm

    kickoff = card.kickoff.strftime("%Y-%m-%d %H:%M %Z").strip()
    line(f"{card.home.name}  vs  {card.away.name}", size=16, gap=8)
    line(f"{card.group}  ·  {kickoff}  ·  {card.venue}", size=9, gap=8)

    pred = card.prediction
    line(
        f"Prediction: {pred.home_name} {pred.score_home}-{pred.score_away} {pred.away_name}"
        f"   (W {pred.p_home:.0%} / D {pred.p_draw:.0%} / L {pred.p_away:.0%})",
        size=12,
        gap=6,
    )
    line(f"Expected goals: {pred.lambda_home:.2f} - {pred.lambda_away:.2f}", size=9)
    tops = ", ".join(f"{h}-{a} ({p:.0%})" for h, a, p in card.top_scorelines)
    line(f"Most likely scorelines: {tops}", size=9, gap=7)
    line(pred.rationale, size=8, gap=8)

    def team_block(team: TeamCard, x: float) -> None:
        nonlocal cursor
        top = cursor
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(x, cursor, f"{team.name}  ({team.formation})")
        cursor -= 5.5 * mm
        badge = team.source if team.source_matchday is None else f"{team.source} MD{team.source_matchday}"
        coach = team.coach_name or "?"
        record = "-".join(str(n) for n in team.coach_record) if team.coach_record else "?"
        line(f"Coach: {coach}  ({record} W-D-L)   [{badge}]", size=8, gap=5, x=x)
        line("Starting XI:", size=9, gap=5, x=x)
        for p in team.starters:
            line(f"  {p.position[:3]:<3} {p.name}  ({p.rating:.0f})", size=8, gap=4.2, x=x)
        line("Likely subs:", size=9, gap=5, x=x)
        for p in team.subs:
            line(f"  {p.position[:3]:<3} {p.name}  ({p.rating:.0f})", size=8, gap=4.2, x=x)
        cursor = top  # reset so the away column starts level with the home column

    block_top = cursor
    team_block(card.home, left)
    cursor = block_top
    team_block(card.away, left + (right - left) / 2.0)

    pdf.showPage()
    pdf.save()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/worldcup/test_cardpdf.py -v`
Expected: PASS.

- [ ] **Step 7: Lint, format, typecheck**

Run: `ruff format src/soccer/worldcup/cardpdf.py tests/worldcup/test_cardpdf.py && ruff check src/soccer/worldcup/cardpdf.py tests/worldcup/test_cardpdf.py && mypy src tests`
Expected: all pass. (Note: `mm` is used in the nested helpers; if ruff flags an unused import it is a real bug — verify it is referenced.)

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml src/soccer/worldcup/cardpdf.py tests/worldcup/test_cardpdf.py
git commit -m "feat(wc): render single-match card to PDF via reportlab [pdf] extra"
```

---

### Task 5: Single-fixture live refresh (`live.py`)

**Files:**
- Modify: `src/soccer/worldcup/live.py`
- Test: `tests/worldcup/test_live.py` (append)

**Interfaces:**
- Consumes: existing `_Client` Protocol, `_results_by_fixture`, `_apply_results`, `_parse_lineups` in `live.py`; `WorldCup`, `replace`.
- Produces: `refresh_fixture(wc: WorldCup, client: _Client, fixture_id: int) -> WorldCup` — merges one fixture's latest result and lineup; replaces any existing lineups for that fixture.

- [ ] **Step 1: Write the failing tests** (append to `tests/worldcup/test_live.py`)

```python
def test_refresh_fixture_attaches_confirmed_lineup_pre_match(
    sample_world_cup: WorldCup,
) -> None:
    from soccer.worldcup.live import refresh_fixture

    # Pre-match: status not finished, but the official lineup is already published.
    client = FakeClient(
        fixtures=[_fixture(9001, 1, 2, "NS", 0, 0)],
        lineups={9001: [_lineup_block(1, "4-2-3-1", [1, 2], [3])]},
    )
    updated = refresh_fixture(sample_world_cup, client, 9001)
    assert not any(m.played for m in updated.matches)  # no result yet
    home = next(lu for lu in updated.lineups if lu.team_id == 1)
    assert home.fixture_id == 9001
    assert home.formation == "4-2-3-1"
    assert home.start_ids == (1, 2)


def test_refresh_fixture_fills_result_when_finished(sample_world_cup: WorldCup) -> None:
    from soccer.worldcup.live import refresh_fixture

    client = FakeClient(fixtures=[_fixture(9001, 1, 2, "FT", 3, 1)], lineups={})
    updated = refresh_fixture(sample_world_cup, client, 9001)
    played = next(m for m in updated.matches if m.fixture_id == 9001)
    assert played.played and played.home_goals == 3 and played.away_goals == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/worldcup/test_live.py -v`
Expected: FAIL — `ImportError: cannot import name 'refresh_fixture'`.

- [ ] **Step 3: Write the implementation** (append to `src/soccer/worldcup/live.py`)

```python
def refresh_fixture(wc: WorldCup, client: _Client, fixture_id: int) -> WorldCup:
    """Merge a single fixture's latest result and lineup into ``wc``.

    Costs at most two API calls. Used by ``wc card --refresh`` to pick up an official lineup
    (or a finished scoreline) for the one match being previewed.
    """
    fixtures = client.get("fixtures", {"id": fixture_id}, force_refresh=True)
    matches = _apply_results(wc.matches, _results_by_fixture(fixtures))
    blocks = client.get("fixtures/lineups", {"fixture": fixture_id})
    new_lineups = _parse_lineups(fixture_id, blocks) if blocks else []
    kept = [lu for lu in wc.lineups if lu.fixture_id != fixture_id]
    return replace(wc, matches=matches, lineups=tuple(kept) + tuple(new_lineups))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_live.py -v`
Expected: PASS (existing 2 tests + 2 new).

- [ ] **Step 5: Lint, format, typecheck**

Run: `ruff format src/soccer/worldcup/live.py tests/worldcup/test_live.py && ruff check src/soccer/worldcup/live.py tests/worldcup/test_live.py && mypy src tests`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/soccer/worldcup/live.py tests/worldcup/test_live.py
git commit -m "feat(wc): add single-fixture live refresh for card previews"
```

---

### Task 6: CLI `wc card` command

**Files:**
- Modify: `src/soccer/worldcup/cli.py`
- Test: `tests/worldcup/test_cli.py` (append)

**Interfaces:**
- Consumes: `build_card` (Task 3), `render_card_pdf` (Task 4), `refresh_fixture` (Task 5), existing `load_dataset`, `_dataset_path`, `rank_all`, `ApiFootballClient`, `ApiFootballError`, `urllib_transport`, `JsonCache`.
- Produces: `cmd_card(args, config) -> int` and a `wc card` subparser. CLI: `soccer wc card <fixture_id> [--refresh] [--out-dir DIR] [--name NAME] [--format {pdf,json,both}]`. `args` namespace fields: `fixture_id: int`, `refresh: bool`, `out_dir: str | None`, `name: str | None`, `format: str`, `throttle: float`.

- [ ] **Step 1: Write the failing tests** (append to `tests/worldcup/test_cli.py`)

```python
def test_card_writes_json(tmp_path: Path, sample_world_cup: WorldCup) -> None:
    from soccer.worldcup.cli import cmd_card

    _write_dataset(tmp_path, sample_world_cup)
    args = argparse.Namespace(
        fixture_id=9001,
        refresh=False,
        out_dir=None,
        name=None,
        format="json",
        throttle=0.0,
    )
    rc = cmd_card(args, _config(tmp_path))
    assert rc == 0
    data = json.loads((tmp_path / "perdiction" / "card-9001.json").read_text())
    assert data["fixture_id"] == 9001
    assert data["home"]["name"] == "England"
    assert "prediction" in data


def test_card_writes_pdf(tmp_path: Path, sample_world_cup: WorldCup) -> None:
    import pytest

    pytest.importorskip("reportlab")
    from soccer.worldcup.cli import cmd_card

    _write_dataset(tmp_path, sample_world_cup)
    args = argparse.Namespace(
        fixture_id=9001, refresh=False, out_dir=None, name=None, format="both", throttle=0.0
    )
    rc = cmd_card(args, _config(tmp_path))
    assert rc == 0
    assert (tmp_path / "perdiction" / "card-9001.pdf").read_bytes()[:4] == b"%PDF"


def test_card_unknown_fixture_returns_error(
    tmp_path: Path, sample_world_cup: WorldCup, capsys: pytest.CaptureFixture[str]
) -> None:
    from soccer.worldcup.cli import cmd_card

    _write_dataset(tmp_path, sample_world_cup)
    args = argparse.Namespace(
        fixture_id=4242, refresh=False, out_dir=None, name=None, format="json", throttle=0.0
    )
    rc = cmd_card(args, _config(tmp_path))
    assert rc == 1
    assert "not found" in capsys.readouterr().out


def test_card_refresh_without_key_fails(
    tmp_path: Path, sample_world_cup: WorldCup, capsys: pytest.CaptureFixture[str]
) -> None:
    from soccer.worldcup.cli import cmd_card

    _write_dataset(tmp_path, sample_world_cup)
    args = argparse.Namespace(
        fixture_id=9001, refresh=True, out_dir=None, name=None, format="json", throttle=0.0
    )
    rc = cmd_card(args, _config(tmp_path, key=None))
    assert rc == 1
    assert "not set" in capsys.readouterr().out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/worldcup/test_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'cmd_card'`.

- [ ] **Step 3a: Add imports to `cli.py`**

Add these imports alongside the existing `from soccer.worldcup...` imports in `src/soccer/worldcup/cli.py`:

```python
from soccer.worldcup.card import build_card
from soccer.worldcup.cardpdf import render_card_pdf
from soccer.worldcup.live import refresh_fixture, refresh_live
```

(Replace the existing `from soccer.worldcup.live import refresh_live` line with the combined import above.)

- [ ] **Step 3b: Add `cmd_card` to `cli.py`**

Add this function after `cmd_refresh`:

```python
def cmd_card(args: argparse.Namespace, config: AppConfig) -> int:
    wc = load_dataset(_dataset_path(config))
    if getattr(args, "refresh", False):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        if not config.api_football_key:
            print("SOCCER_API_FOOTBALL_KEY is not set; cannot refresh live data", flush=True)
            return 1
        client = ApiFootballClient(
            config.api_football_key,
            base_url=config.api_football_base_url,
            transport=urllib_transport(timeout=30.0),
            cache=JsonCache(config.data_dir / "api"),
            throttle_seconds=args.throttle,
        )
        try:
            wc = refresh_fixture(wc, client, args.fixture_id)
        except ApiFootballError as exc:
            print(f"refresh failed: {exc}")
            return 1
        _dataset_path(config).write_text(json.dumps(wc.to_dict()), encoding="utf-8")

    rankings = rank_all(wc)
    try:
        card = build_card(wc, rankings, args.fixture_id)
    except ValueError as exc:
        print(f"{exc}; run `soccer wc predict` to list fixture ids")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else _prediction_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"card-{args.fixture_id}"
    written: list[Path] = []
    if args.format in ("json", "both"):
        json_path = out_dir / f"{name}.json"
        json_path.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
        written.append(json_path)
    if args.format in ("pdf", "both"):
        pdf_path = out_dir / f"{name}.pdf"
        try:
            render_card_pdf(card, pdf_path)
        except RuntimeError as exc:
            print(str(exc))
            return 1
        written.append(pdf_path)

    pred = card.prediction
    print(
        f"{card.home.name} {pred.score_home}-{pred.score_away} {card.away.name} "
        f"(W {pred.p_home:.0%} / D {pred.p_draw:.0%} / L {pred.p_away:.0%}); "
        f"lineups {card.home.source}/{card.away.source}"
    )
    for path in written:
        print(f"wrote {path}")
    return 0
```

- [ ] **Step 3c: Register the subparser**

In `add_wc_subparser`, after the `p_refresh` block, add:

```python
    p_card = wc_sub.add_parser("card", help="single-match pre-match preview (PDF/JSON)")
    p_card.add_argument("fixture_id", type=int, help="fixture id to preview")
    p_card.add_argument(
        "--refresh",
        action="store_true",
        help="pull this fixture's latest lineup/result first (needs an API key)",
    )
    p_card.add_argument("--out-dir", default=None, help="output directory for the files")
    p_card.add_argument("--name", default=None, help="basename for the output files")
    p_card.add_argument(
        "--format", choices=["pdf", "json", "both"], default="both", help="output format"
    )
    p_card.add_argument("--throttle", type=float, default=0.02, help="seconds between calls")
    p_card.set_defaults(func=cmd_card)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_cli.py -v`
Expected: PASS (existing tests + 4 new).

- [ ] **Step 5: Lint, format, typecheck**

Run: `ruff format src/soccer/worldcup/cli.py tests/worldcup/test_cli.py && ruff check src/soccer/worldcup/cli.py tests/worldcup/test_cli.py && mypy src tests`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/soccer/worldcup/cli.py tests/worldcup/test_cli.py
git commit -m "feat(wc): add 'wc card' single-match preview command"
```

---

### Task 7: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Update `README.md`**

Find the section documenting the `wc` subcommands (near `wc predict` / `wc refresh`) and add a `wc card` entry. Add this after the `wc refresh` documentation:

```markdown
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
```

- [ ] **Step 2: Update `docs/architecture.md`**

Add a short paragraph in the World Cup pipeline section:

```markdown
### Single-match preview card

`soccer wc card <fixture_id>` previews one upcoming match. `lineup.project_lineup` resolves the
most likely XI/formation (confirmed → prior matchday → squad projection); `predict.predict_one`
reuses the Poisson core but applies tournament momentum plus that match's lineup-quality and
formation lean via `adjust.adjustment_for_match`; `card.build_card` packages coaches, lineups,
and the forecast; and `cardpdf.render_card_pdf` renders it with reportlab (the optional `[pdf]`
extra, imported lazily). `--refresh` merges one fixture's latest lineup/result via
`live.refresh_fixture`.
```

- [ ] **Step 3: Verify the full check suite**

Run: `make check`
Expected: format check, lint, mypy, and the full pytest suite all pass.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/architecture.md
git commit -m "docs(wc): document the 'wc card' single-match preview"
```

---

## Self-Review

**Spec coverage:**
- PDF with coach, starting XI + formation, possible subs, prediction → Tasks 3 (model) + 4 (PDF). ✓
- `wc card <fixture_id>` command → Task 6. ✓
- Lineup precedence confirmed → prior → projected (coach-preferred formation, prior 2026 lineups) → Task 1. ✓
- More accurate (lineup + momentum aware) prediction → Task 2. ✓
- reportlab as optional `[pdf]` extra, lazily imported → Task 4. ✓
- JSON output → Tasks 3 (`to_dict`) + 6. ✓
- `--refresh` single-fixture live pull → Task 5 + 6. ✓
- Error handling (unknown fixture, missing key, missing reportlab) → Tasks 1/3 (ValueError), 6 (key + ValueError→exit 1), 4/6 (RuntimeError→exit 1). ✓
- Docs (README extra + usage, architecture note) → Task 7. ✓
- Knockout next-phase compatibility (fixture-id keyed, group as label only) → honored across Tasks 1–6 (no group input to projection/prediction). ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to Task N" — every code step shows full code. ✓

**Type consistency:**
- `ProjectedLineup(team_id, formation, start_ids, sub_ids, source, source_matchday)` constructed positionally in tests matches the field order in Task 1. ✓
- `adjustment_for_match` / `predict_one` / `top_scorelines` signatures identical across Tasks 2, 3, 6. ✓
- `MatchCard.to_dict()` keys (`fixture_id, group, kickoff, venue, home, away, prediction, top_scorelines`) match the assertions in Task 3 and the CLI JSON consumer in Task 6. ✓
- `render_card_pdf(card, path)` signature consistent across Tasks 4 and 6. ✓
- `refresh_fixture(wc, client, fixture_id)` signature consistent across Tasks 5 and 6. ✓
- `cmd_card` reads `args.fixture_id/refresh/out_dir/name/format/throttle`; the subparser defines all six. ✓

No issues found.
