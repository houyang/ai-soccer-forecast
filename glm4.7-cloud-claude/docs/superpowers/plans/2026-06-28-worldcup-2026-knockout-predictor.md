# World Cup 2026 Knockout Predictor + Match-Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline (cached-data + optional live-lineup) World Cup 2026 knockout predictor to `glm4.7-cloud-claude` that predicts all 16 Round-of-32 matches and simulates the bracket to a champion, plus a single-match pre-match PDF preview card (coach, formation, starting XI, subs, prediction).

**Architecture:** New self-contained `soccer_agent/worldcup/` subpackage. Loads the cached dataset copied from `opus4.8-cloud-claude/data/worldcup-2026.json` into Pydantic models, computes static 0–100 rankings, recalibrates team strength with actual group-stage results, projects lineups (live-fetched when the API key is present, else projected from a curated formation table + squad ratings), runs an independent-Poisson + Dixon-Coles scoreline model, and Monte-Carlo simulates the knockout bracket. A `reportlab` PDF renders single-match cards. The existing LangGraph `soccer_agent` agent is untouched.

**Tech Stack:** Python 3.14, Pydantic v2, stdlib `math`/`urllib`/`json`, `reportlab` (optional `[pdf]` extra), pytest. API-Sports v3 (`https://v3.football.api-sports.io`) for optional live lineups.

## Global Constraints

- **No reference to opus4.8's implementation.** Only its raw cached JSON dataset is reused (copied into `glm4.7/data/worldcup-2026.json`). All code is written originally for `glm4.7`.
- **The API-Football key (provided separately in the task conversation) must never be committed.** It lives only in a git-ignored `.env`, read via `os.getenv("API_FOOTBALL_KEY")`. The `.gitignore` that ignores `.env` is committed **before** `.env` is created. The literal key must never appear in any committed file (including this plan).
- API-Sports v3 base URL: `https://v3.football.api-sports.io`; auth header `x-apisports-key`. World Cup = league `1`, season `2026`. The API rejects an explicit `page=1`, so do not send `page` for single-page endpoints.
- Player/team/coach/fixture IDs in the dataset ARE the API-Sports IDs — they match lineup API responses directly.
- Line length 100, ruff `select = ["E","F","I","N","W"]`, `ignore = ["E501"]`, target py312+ (matches `pyproject.toml`).
- Every prediction function is **pure and deterministic** given its inputs; network lives only in `live.py`. Tests must not hit the network.

## File Structure

Create under `soccer_agent/worldcup/`:
- `__init__.py` — public exports.
- `entities.py` — Pydantic models for the dataset.
- `dataset.py` — locate + load `data/worldcup-2026.json`.
- `reference.py` — `country_strength(name) -> float` (0–100) static pedigree table.
- `ranking.py` — `rank_all(wc) -> Rankings` (league→club→player→coach→team, 0–100).
- `form.py` — group-stage recalibration: `TeamForm`, `compute_forms`, `recalibrated_strength`.
- `live.py` — optional API-Sports lineup fetcher with on-disk cache.
- `lineup.py` — `project_lineup` (curated `FORMATIONS` + squad projection + live/prior integration).
- `predict.py` — Poisson + Dixon-Coles model, `predict_one`, `top_scorelines`.
- `standings.py` — `group_standings(wc)` with FIFA tiebreakers.
- `bracket.py` — R32 fixtures (from dataset) + R16→Final tree.
- `simulate.py` — Monte-Carlo bracket simulation.
- `card.py` — `build_card` single-match preview.
- `cardpdf.py` — `render_card_pdf` via reportlab (lazy import).
- `cli.py` + `__main__.py` — `predict` / `card` / `bracket` commands.

Other:
- Modify: `pyproject.toml` (add `[pdf]` extra).
- Create: `.gitignore`, `data/worldcup-2026.json` (copy), `.env` (git-ignored, not committed).
- Tests: `tests/test_worldcup_*.py`.
- Outputs: `predictions/worldcup-2026-predictions-after1st-group.{md,json}`, `predictions/<Home>-vs-<Away>.{pdf,json}`.

---

### Task 1: Project scaffolding, .gitignore, dataset copy, pyproject extra

**Files:**
- Create: `.gitignore`
- Create: `data/worldcup-2026.json` (copy from `../opus4.8-cloud-claude/data/worldcup-2026.json`)
- Create: `soccer_agent/worldcup/__init__.py` (empty package marker for now)
- Modify: `pyproject.toml` (add `pdf` extra)
- Test: `tests/test_worldcup_dataset.py`

**Interfaces:**
- Produces: the package `soccer_agent.worldcup` importable; `data/worldcup-2026.json` present; `reportlab` installable via `pip install -e '.[pdf]'`.

- [ ] **Step 1: Create `.gitignore` (MUST be committed before any `.env`)**

```gitignore
# Secrets — never commit
.env
.env.*

# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/

# Live-lineup fetch cache
data/live/
```

- [ ] **Step 2: Commit .gitignore immediately**

```bash
git add .gitignore
git commit -m "chore: add .gitignore ignoring .env and caches"
```

- [ ] **Step 3: Create the empty worldcup package**

```bash
mkdir -p soccer_agent/worldcup
printf '"""World Cup 2026 knockout predictor subpackage."""\n' > soccer_agent/worldcup/__init__.py
```

- [ ] **Step 4: Copy the cached dataset**

```bash
mkdir -p data
cp ../opus4.8-cloud-claude/data/worldcup-2026.json data/worldcup-2026.json
```

- [ ] **Step 5: Add the `[pdf]` extra to `pyproject.toml`**

In `[project.optional-dependencies]`, after the `dev = [...]` block, add:

```toml
pdf = [
    "reportlab>=4.0",
]
```

- [ ] **Step 6: Write the failing test**

```python
# tests/test_worldcup_dataset.py
import json
from pathlib import Path


def test_dataset_present_and_well_formed():
    path = Path(__file__).resolve().parents[1] / "data" / "worldcup-2026.json"
    assert path.exists(), "data/worldcup-2026.json must be copied in"
    data = json.loads(path.read_text())
    for key in ("teams", "players", "coaches", "matches"):
        assert key in data and len(data[key]) > 0
    assert len(data["teams"]) == 48
    assert len(data["matches"]) == 88
```

- [ ] **Step 7: Run test — expect PASS (no code yet, just data)**

Run: `pytest tests/test_worldcup_dataset.py -v`
Expected: PASS.

- [ ] **Step 8: Verify .env would be ignored, then create it (NOT committed)**

First confirm `.env` is ignored, then write the key into it. The key is provided in the task conversation — paste it where indicated. It must never be typed into any committed file.

```bash
git check-ignore .env && echo OK
# Write the API-Football key (provided in conversation) into .env. Replace <KEY> with the actual value.
printf 'API_FOOTBALL_KEY=<KEY>\n' > .env
git status --porcelain | grep -q '\.env' && echo "ERROR: .env tracked!" || echo "env-safe"
```
Expected: `OK` then `env-safe`. If `ERROR`, stop — fix `.gitignore` before continuing. The `.env` file is intentionally not `git add`-ed in any step.

- [ ] **Step 9: Commit scaffolding (NOT .env)**

```bash
git add soccer_agent/worldcup/__init__.py data/worldcup-2026.json pyproject.toml tests/test_worldcup_dataset.py
git status --porcelain | grep '\.env' && echo "ABORT: .env staged" || true
git commit -m "feat(wc): scaffold worldcup subpackage, copy cached dataset, add pdf extra"
```

---

### Task 2: Entities + dataset loader

**Files:**
- Create: `soccer_agent/worldcup/entities.py`
- Create: `soccer_agent/worldcup/dataset.py`
- Test: `tests/test_worldcup_entities.py`

**Interfaces:**
- Produces: `WorldCup` model with `teams: dict[int, NationalTeam]`, `players: dict[int, Player]`, `coaches: dict[int, Coach]`, `matches: list[WcMatch]`, `lineups: list[Lineup]`; methods `squad(team_id) -> list[Player]`, `groups() -> dict[str, list[NationalTeam]]`. `load_worldcup(path=None) -> WorldCup`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worldcup_entities.py
from soccer_agent.worldcup.dataset import load_worldcup


def test_load_worldcup_keys_and_squad():
    wc = load_worldcup()
    assert len(wc.teams) == 48
    assert len(wc.players) == 1248
    assert len(wc.coaches) == 48
    any_team = next(iter(wc.teams.values()))
    squad = wc.squad(any_team.id)
    assert 20 <= len(squad) <= 30
    assert all(p.wc_team_id == any_team.id for p in squad)


def test_groups_are_four_teams_each():
    wc = load_worldcup()
    groups = wc.groups()
    assert len(groups) == 12
    for name, teams in groups.items():
        assert name.startswith("Group ")
        assert len(teams) == 4


def test_matches_round_trip():
    wc = load_worldcup()
    r32 = [m for m in wc.matches if m.matchday == 0]
    assert len(r32) == 16
    assert all(m.home_goals is None for m in r32)
    group_played = [m for m in wc.matches if m.matchday in (1, 2, 3) and m.played]
    assert len(group_played) == 72
```

- [ ] **Step 2: Run test — expect FAIL (`ModuleNotFoundError`)**

Run: `pytest tests/test_worldcup_entities.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `entities.py`**

```python
# soccer_agent/worldcup/entities.py
"""Pydantic models for the cached FIFA 2026 World Cup dataset."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class League(BaseModel):
    id: int
    name: str
    country: str
    n_teams: int
    matches_played: int
    avg_attendance: float


class Club(BaseModel):
    id: int
    name: str
    country: str
    league_id: Optional[int] = None
    wins: int
    draws: int
    losses: int
    titles: int

    @property
    def played(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.played if self.played else 0.0


class Player(BaseModel):
    id: int
    name: str
    age: Optional[int] = None
    position: str  # Goalkeeper | Defender | Midfielder | Attacker
    club_id: Optional[int] = None
    goals: int
    rating: float
    appearances: int
    wc_team_id: int


class Coach(BaseModel):
    id: int
    name: str
    age: Optional[int] = None
    wins: int
    draws: int
    losses: int
    titles: int
    team_id: int

    @property
    def played(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.played if self.played else 0.0


class NationalTeam(BaseModel):
    id: int
    name: str
    group: str
    confederation: str
    is_host: bool
    player_ids: tuple[int, ...] = Field(default_factory=tuple)
    coach_id: Optional[int] = None
    recent_w: int
    recent_d: int
    recent_l: int


class WcMatch(BaseModel):
    fixture_id: int
    matchday: int
    group: str
    home_id: int
    away_id: int
    kickoff: datetime
    venue: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    round_name: str = ""

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None


class Lineup(BaseModel):
    fixture_id: int
    team_id: int
    formation: str
    start_ids: tuple[int, ...] = Field(default_factory=tuple)
    sub_ids: tuple[int, ...] = Field(default_factory=tuple)


class WorldCup(BaseModel):
    leagues: dict[int, League] = Field(default_factory=dict)
    clubs: dict[int, Club] = Field(default_factory=dict)
    players: dict[int, Player] = Field(default_factory=dict)
    coaches: dict[int, Coach] = Field(default_factory=dict)
    teams: dict[int, NationalTeam] = Field(default_factory=dict)
    matches: list[WcMatch] = Field(default_factory=list)
    lineups: list[Lineup] = Field(default_factory=list)

    def squad(self, team_id: int) -> list[Player]:
        team = self.teams[team_id]
        return [self.players[pid] for pid in team.player_ids if pid in self.players]

    def groups(self) -> dict[str, list[NationalTeam]]:
        out: dict[str, list[NationalTeam]] = {}
        for team in self.teams.values():
            out.setdefault(team.group, []).append(team)
        return {g: sorted(ts, key=lambda t: t.name) for g, ts in sorted(out.items())}

    @classmethod
    def from_dict(cls, raw: dict) -> "WorldCup":
        return cls(
            leagues={x["id"]: League(**x) for x in raw.get("leagues", [])},
            clubs={x["id"]: Club(**x) for x in raw.get("clubs", [])},
            players={x["id"]: Player(**x) for x in raw.get("players", [])},
            coaches={x["id"]: Coach(**x) for x in raw.get("coaches", [])},
            teams={x["id"]: NationalTeam(**x) for x in raw.get("teams", [])},
            matches=[WcMatch(**x) for x in raw.get("matches", [])],
            lineups=[Lineup(**x) for x in raw.get("lineups", [])],
        )
```

- [ ] **Step 4: Implement `dataset.py`**

```python
# soccer_agent/worldcup/dataset.py
"""Locate and load the cached World Cup dataset."""
from __future__ import annotations

import json
from pathlib import Path

from soccer_agent.worldcup.entities import WorldCup

# This module's file: soccer_agent/worldcup/dataset.py -> repo root is 3 parents up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = _REPO_ROOT / "data" / "worldcup-2026.json"


def load_worldcup(path: str | Path | None = None) -> WorldCup:
    """Load the cached dataset, defaulting to data/worldcup-2026.json under the repo root."""
    p = Path(path) if path else DATA_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"World Cup dataset not found at {p}. Copy it from opus4.8-cloud-claude/data/."
        )
    return WorldCup.from_dict(json.loads(p.read_text()))
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/test_worldcup_entities.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add soccer_agent/worldcup/entities.py soccer_agent/worldcup/dataset.py tests/test_worldcup_entities.py
git commit -m "feat(wc): add Pydantic entities and dataset loader"
```

---

### Task 3: Reference country-strength table

**Files:**
- Create: `soccer_agent/worldcup/reference.py`
- Test: `tests/test_worldcup_reference.py`

**Interfaces:**
- Produces: `country_strength(name: str) -> float` (0–100). Known nations return a curated value; unknown returns 50.0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worldcup_reference.py
from soccer_agent.worldcup.reference import country_strength


def test_known_countries_ordered():
    assert country_strength("Brazil") > country_strength("New Zealand")
    assert country_strength("France") > 80.0
    assert country_strength("Germany") > 75.0


def test_unknown_is_neutral():
    assert country_strength("Atlantis") == 50.0


def test_all_wc_teams_have_a_value():
    from soccer_agent.worldcup.dataset import load_worldcup
    wc = load_worldcup()
    for t in wc.teams.values():
        assert 0.0 <= country_strength(t.name) <= 100.0
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_worldcup_reference.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `reference.py`**

```python
# soccer_agent/worldcup/reference.py
"""Static 0-100 pedigree scores per national team (pre-tournament strength prior).

Values reflect historical World Cup pedigree and current federation standing. They are a
prior only; group-stage results override them in `form.recalibrated_strength`.
"""
from __future__ import annotations

_NEUTRAL = 50.0

# Curated pedigree (0-100). Tuned so traditional powers lead; minnows trail.
_STRENGTH: dict[str, float] = {
    "Argentina": 94.0, "France": 93.0, "Brazil": 90.0, "England": 89.0,
    "Spain": 88.0, "Germany": 87.0, "Portugal": 86.0, "Netherlands": 85.0,
    "Belgium": 82.0, "Croatia": 81.0, "Italy": 80.0,
    "Colombia": 78.0, "Uruguay": 78.0, "Morocco": 77.0, "Mexico": 76.0,
    "USA": 75.0, "Switzerland": 74.0, "Japan": 74.0, "Senegal": 73.0,
    "Ecuador": 71.0, "Australia": 70.0, "South Korea": 70.0, "Sweden": 70.0,
    "Norway": 72.0, "Austria": 71.0, "Czech Republic": 70.0, "Türkiye": 71.0,
    "Ivory Coast": 70.0, "Ghana": 69.0, "Egypt": 69.0, "Tunisia": 67.0,
    "Iran": 68.0, "Saudi Arabia": 65.0, "Iraq": 63.0, "Jordan": 60.0,
    "Qatar": 61.0, "Uzbekistan": 62.0, "Canada": 71.0, "Panama": 60.0,
    "Paraguay": 68.0, "Scotland": 69.0, "Wales": 70.0,
    "Algeria": 68.0, "Cape Verde Islands": 58.0, "Congo DR": 60.0,
    "Curaçao": 57.0, "Haiti": 58.0, "New Zealand": 55.0, "Bosnia & Herzegovina": 64.0,
    "South Africa": 66.0,
}


def country_strength(name: str) -> float:
    """Return a 0-100 pedigree score for a national team name; 50.0 if unknown."""
    return _STRENGTH.get(name, _NEUTRAL)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_worldcup_reference.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add soccer_agent/worldcup/reference.py tests/test_worldcup_reference.py
git commit -m "feat(wc): add country-strength reference table"
```

---

### Task 4: Static rankings (league→club→player→coach→team)

**Files:**
- Create: `soccer_agent/worldcup/ranking.py`
- Test: `tests/test_worldcup_ranking.py`

**Interfaces:**
- Consumes: `WorldCup`, `country_strength`.
- Produces: `Rankings` dataclass `{leagues, clubs, players, coaches, teams: dict[int,float]}`; `rank_all(wc) -> Rankings`; `top_n(scores, n)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worldcup_ranking.py
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.ranking import rank_all, top_n


def test_rankings_deterministic_and_bounded():
    wc = load_worldcup()
    r1 = rank_all(wc)
    r2 = rank_all(wc)
    assert r1.teams == r2.teams
    for score in r1.teams.values():
        assert 0.0 <= score <= 100.0
    for score in r1.players.values():
        assert 0.0 <= score <= 100.0


def test_top_teams_make_sense():
    wc = load_worldcup()
    r = rank_all(wc)
    top = top_n(r.teams, 5)
    assert len(top) == 5
    names = [wc.teams[tid].name for tid, _ in top]
    # At least three traditional powers in the top 5.
    powers = {"Argentina", "France", "Brazil", "England", "Spain", "Germany", "Portugal"}
    assert len(set(names) & powers) >= 3


def test_hosts_get_bonus():
    wc = load_worldcup()
    r = rank_all(wc)
    # USA/Mexico/Canada are hosts; they should rate respectably.
    for tid, team in wc.teams.items():
        if team.is_host:
            assert r.teams[tid] >= 55.0
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_worldcup_ranking.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `ranking.py`**

```python
# soccer_agent/worldcup/ranking.py
"""Deterministic 0-100 rankings in dependency order: league -> club -> player -> coach -> team.

Each tier is a min-max-normalized blend of the fields below; unknowns fall back to neutral.
This is the STATIC (pre-knockout) rating; group-stage results recalibrate it in `form.py`.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from soccer_agent.worldcup.entities import Player, WorldCup
from soccer_agent.worldcup.reference import country_strength

W_LEAGUE_PLAYERS, W_LEAGUE_ATT, W_LEAGUE_COUNTRY = 0.45, 0.25, 0.30
W_CLUB_LEAGUE, W_CLUB_PLAYERS, W_CLUB_WINRATE, W_CLUB_TITLES = 0.40, 0.25, 0.25, 0.10
W_PLAYER_CLUB, W_PLAYER_LEAGUE, W_PLAYER_GOALS, W_PLAYER_RATING = 0.35, 0.15, 0.20, 0.30
W_COACH_SQUAD, W_COACH_WINRATE = 0.55, 0.45
W_TEAM_COUNTRY, W_TEAM_SQUAD, W_TEAM_COACH, W_TEAM_FORM, W_TEAM_DOMESTIC = 0.30, 0.30, 0.15, 0.15, 0.10
HOST_BONUS = 3.0
_NEUTRAL = 50.0
_RATING_FLOOR, _RATING_CEIL = 6.0, 8.0
_GOAL_EXPECT = {"Attacker": 18.0, "Midfielder": 9.0, "Defender": 4.0, "Goalkeeper": 1.0}
_SQUAD_CORE = 16


@dataclass(frozen=True)
class Rankings:
    leagues: dict[int, float] = field(default_factory=dict)
    clubs: dict[int, float] = field(default_factory=dict)
    players: dict[int, float] = field(default_factory=dict)
    coaches: dict[int, float] = field(default_factory=dict)
    teams: dict[int, float] = field(default_factory=dict)


def _minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return dict.fromkeys(values, 0.5)
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _clamp(v: float) -> float:
    return max(0.0, min(100.0, v))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _wc_players_per_league(wc: WorldCup) -> dict[int, int]:
    counts = dict.fromkeys(wc.leagues, 0)
    for p in wc.players.values():
        club = wc.clubs.get(p.club_id) if p.club_id is not None else None
        if club is not None and club.league_id in counts:
            counts[club.league_id] += 1
    return counts


def _wc_players_per_club(wc: WorldCup) -> dict[int, int]:
    counts = dict.fromkeys(wc.clubs, 0)
    for p in wc.players.values():
        if p.club_id in counts:
            counts[p.club_id] += 1
    return counts


def rank_leagues(wc: WorldCup) -> dict[int, float]:
    players = _wc_players_per_league(wc)
    norm_p = _minmax({lid: float(n) for lid, n in players.items()})
    norm_att = _minmax({lid: lg.avg_attendance for lid, lg in wc.leagues.items()})
    out: dict[int, float] = {}
    for lid, lg in wc.leagues.items():
        score = (
            W_LEAGUE_PLAYERS * norm_p.get(lid, 0.0)
            + W_LEAGUE_ATT * norm_att.get(lid, 0.0)
            + W_LEAGUE_COUNTRY * (country_strength(lg.country) / 100.0)
        )
        out[lid] = _clamp(score * 100.0)
    return out


def rank_clubs(wc: WorldCup, league_scores: dict[int, float]) -> dict[int, float]:
    wc_players = _wc_players_per_club(wc)
    norm_p = _minmax({cid: float(n) for cid, n in wc_players.items()})
    norm_t = _minmax({cid: float(c.titles) for cid, c in wc.clubs.items()})
    out: dict[int, float] = {}
    for cid, c in wc.clubs.items():
        ls = league_scores.get(c.league_id, _NEUTRAL) if c.league_id else _NEUTRAL
        score = (
            W_CLUB_LEAGUE * (ls / 100.0)
            + W_CLUB_PLAYERS * norm_p.get(cid, 0.0)
            + W_CLUB_WINRATE * c.win_rate
            + W_CLUB_TITLES * norm_t.get(cid, 0.0)
        )
        out[cid] = _clamp(score * 100.0)
    return out


def _goal_score(p: Player) -> float:
    exp = _GOAL_EXPECT.get(p.position, 8.0)
    return min(p.goals / exp, 1.0) if exp else 0.0


def _rating_score(p: Player) -> float:
    if p.rating <= 0.0:
        return 0.5
    return max(0.0, min((p.rating - _RATING_FLOOR) / (_RATING_CEIL - _RATING_FLOOR), 1.0))


def rank_players(wc, league_scores, club_scores) -> dict[int, float]:
    out: dict[int, float] = {}
    for pid, p in wc.players.items():
        club = wc.clubs.get(p.club_id) if p.club_id is not None else None
        if club is not None:
            cs = club_scores.get(club.id, _NEUTRAL)
            ls = league_scores.get(club.league_id, _NEUTRAL) if club.league_id is not None else _NEUTRAL
        else:
            cs = ls = _NEUTRAL
        score = (
            W_PLAYER_CLUB * (cs / 100.0)
            + W_PLAYER_LEAGUE * (ls / 100.0)
            + W_PLAYER_GOALS * _goal_score(p)
            + W_PLAYER_RATING * _rating_score(p)
        )
        out[pid] = _clamp(score * 100.0)
    return out


def _squad_quality(wc, team_id, player_scores) -> float:
    scores = sorted((player_scores.get(p.id, _NEUTRAL) for p in wc.squad(team_id)), reverse=True)
    return _mean(scores[:_SQUAD_CORE]) if scores else _NEUTRAL


def rank_coaches(wc, player_scores) -> dict[int, float]:
    out: dict[int, float] = {}
    for cid, coach in wc.coaches.items():
        squad = _squad_quality(wc, coach.team_id, player_scores)
        out[cid] = _clamp((W_COACH_SQUAD * (squad / 100.0) + W_COACH_WINRATE * coach.win_rate) * 100.0)
    return out


def _domestic(wc, team_id, league_scores) -> float:
    vals: list[float] = []
    for p in wc.squad(team_id):
        club = wc.clubs.get(p.club_id) if p.club_id is not None else None
        if club and club.league_id:
            vals.append(league_scores.get(club.league_id, _NEUTRAL))
    return _mean(vals) if vals else _NEUTRAL


def rank_teams(wc, league_scores, player_scores, coach_scores) -> dict[int, float]:
    out: dict[int, float] = {}
    for tid, team in wc.teams.items():
        squad = _squad_quality(wc, tid, player_scores)
        coach = coach_scores.get(team.coach_id, _NEUTRAL) if team.coach_id else _NEUTRAL
        domestic = _domestic(wc, tid, league_scores)
        played = team.recent_w + team.recent_d + team.recent_l
        form = (team.recent_w + 0.5 * team.recent_d) / played if played else 0.5
        score = (
            W_TEAM_COUNTRY * (country_strength(team.name) / 100.0)
            + W_TEAM_SQUAD * (squad / 100.0)
            + W_TEAM_COACH * (coach / 100.0)
            + W_TEAM_FORM * form
            + W_TEAM_DOMESTIC * (domestic / 100.0)
        ) * 100.0
        if team.is_host:
            score += HOST_BONUS
        out[tid] = _clamp(score)
    return out


def rank_all(wc: WorldCup) -> Rankings:
    leagues = rank_leagues(wc)
    clubs = rank_clubs(wc, leagues)
    players = rank_players(wc, leagues, clubs)
    coaches = rank_coaches(wc, players)
    teams = rank_teams(wc, leagues, players, coaches)
    return Rankings(leagues=leagues, clubs=clubs, players=players, coaches=coaches, teams=teams)


def top_n(scores: dict[int, float], n: int) -> list[tuple[int, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_worldcup_ranking.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add soccer_agent/worldcup/ranking.py tests/test_worldcup_ranking.py
git commit -m "feat(wc): add static 0-100 ranking tier computation"
```

---

### Task 5: Group-stage form recalibration

**Files:**
- Create: `soccer_agent/worldcup/form.py`
- Test: `tests/test_worldcup_form.py`

**Interfaces:**
- Consumes: `WorldCup`, `Rankings`.
- Produces: `TeamForm` dataclass `{team_id, played, wins, draws, losses, gf, ga, gd, pts, attack, defense}`; `compute_forms(wc) -> dict[int, TeamForm]`; `recalibrated_strength(wc, rankings, forms) -> dict[int, float]` (0–100). Blend weight `W_FORM = 0.45` (group-stage performance), `1 - W_FORM` static ranking; small-sample shrinkage toward 50 (neutral) using `shrink = played / (played + k)`, `k=3`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worldcup_form.py
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength


def test_all_48_teams_have_form_with_three_played():
    wc = load_worldcup()
    forms = compute_forms(wc)
    assert len(forms) == 48
    for f in forms.values():
        assert f.played == 3
        assert f.gf - f.ga == f.gd
        assert f.pts == 3 * f.wins + f.draws


def test_recalibration_tracks_group_performance():
    wc = load_worldcup()
    rankings = rank_all(wc)
    forms = compute_forms(wc)
    strengths = recalibrated_strength(wc, rankings, forms)
    assert len(strengths) == 48
    for s in strengths.values():
        assert 0.0 <= s <= 100.0
    # A team with a huge positive GD should rate higher than its static ranking.
    best_gd = max(forms.values(), key=lambda f: f.gd)
    static = rankings.teams[best_gd.team_id]
    assert strengths[best_gd.team_id] >= static - 5.0  # recalibration never crashes it
    # And a team with a very negative GD should drop relative to static.
    worst_gd = min(forms.values(), key=lambda f: f.gd)
    assert strengths[worst_gd.team_id] <= rankings.teams[worst_gd.team_id] + 5.0
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_worldcup_form.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `form.py`**

```python
# soccer_agent/worldcup/form.py
"""Recalibrate team strength using actual group-stage results.

The static ranking is a pre-tournament prior; this blends it with each team's real
group-stage goal difference (regressed toward neutral for the 3-match small sample).
"""
from __future__ import annotations

from dataclasses import dataclass

from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.ranking import Rankings

W_FORM = 0.45          # weight on group-stage performance
SHRINK_K = 3.0         # small-sample shrinkage (3 games played)
GD_PER_POINT = 4.0     # 1 group-stage GD point ~ 4 rating points at full weight
_NEUTRAL = 50.0


@dataclass(frozen=True)
class TeamForm:
    team_id: int
    played: int
    wins: int
    draws: int
    losses: int
    gf: int
    ga: int
    gd: int
    pts: int
    attack: float   # goals-per-game, regressed
    defense: float  # goals-conceded-per-game, regressed


def compute_forms(wc: WorldCup) -> dict[int, TeamForm]:
    agg: dict[int, dict] = {tid: {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0}
                            for tid in wc.teams}
    for m in wc.matches:
        if m.matchday not in (1, 2, 3) or not m.played:
            continue
        for home, away, gid in ((True, False, m.home_id), (False, True, m.away_id)):
            a = agg[gid]
            a["p"] += 1
            if home:
                a["gf"] += m.home_goals
                a["ga"] += m.away_goals
                if m.home_goals > m.away_goals:
                    a["w"] += 1
                elif m.home_goals < m.away_goals:
                    a["l"] += 1
                else:
                    a["d"] += 1
            else:
                a["gf"] += m.away_goals
                a["ga"] += m.home_goals
                if m.away_goals > m.home_goals:
                    a["w"] += 1
                elif m.away_goals < m.home_goals:
                    a["l"] += 1
                else:
                    a["d"] += 1

    out: dict[int, TeamForm] = {}
    for tid, a in agg.items():
        p = a["p"]
        gf, ga = a["gf"], a["ga"]
        gd = gf - ga
        pts = 3 * a["w"] + a["d"]
        # Regress per-game rates toward 1.5 scored / 1.5 conceded (tournament average-ish).
        shrink = p / (p + SHRINK_K)
        attack = (gf / p if p else 1.5) * shrink + 1.5 * (1 - shrink)
        defense = (ga / p if p else 1.5) * shrink + 1.5 * (1 - shrink)
        out[tid] = TeamForm(tid, p, a["w"], a["d"], a["l"], gf, ga, gd, pts, attack, defense)
    return out


def recalibrated_strength(wc: WorldCup, rankings: Rankings, forms: dict[int, TeamForm]) -> dict[int, float]:
    out: dict[int, float] = {}
    for tid, team in wc.teams.items():
        static = rankings.teams.get(tid, _NEUTRAL)
        f = forms.get(tid)
        if f is None or f.played == 0:
            out[tid] = static
            continue
        # Convert group-stage GD to a 0-100-ish performance score around neutral.
        perf = _NEUTRAL + f.gd * GD_PER_POINT
        perf = max(0.0, min(100.0, perf))
        out[tid] = (1 - W_FORM) * static + W_FORM * perf
    return out
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_worldcup_form.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add soccer_agent/worldcup/form.py tests/test_worldcup_form.py
git commit -m "feat(wc): add group-stage form recalibration of team strength"
```

---

### Task 6: Live lineup fetcher (API-Sports v3, optional, cached)

**Files:**
- Create: `soccer_agent/worldcup/live.py`
- Test: `tests/test_worldcup_live.py` (offline; uses a cached JSON fixture, no network)

**Interfaces:**
- Consumes: env `API_FOOTBALL_KEY`, `WorldCup`.
- Produces: `LineupFetcher` with `fetch_fixture_lineups(fixture_id) -> list[Lineup] | None` and `recent_team_lineup(wc, team_id) -> Lineup | None` (the team's most-recent *played* WC match lineup). On-disk cache under `data/live/`. Returns `None` (gracefully) when no key or network error.

- [ ] **Step 1: Write the failing test (offline, fixture-based)**

```python
# tests/test_worldcup_live.py
import json
from pathlib import Path

from soccer_agent.worldcup.live import parse_lineup_response


SAMPLE = {
    "get": "fixtures/lineups",
    "response": [
        {
            "team": {"id": 16, "name": "Mexico"},
            "coach": {"id": 1, "name": "Javier Aguirre"},
            "formation": "4-1-4-1",
            "startXI": [{"player": {"id": 270774, "name": "R. Rangel", "pos": "G", "grid": "1:1"}},
                        {"player": {"id": 11, "name": "X", "pos": "D", "grid": "2:1"}}],
            "substitutes": [{"player": {"id": 2098, "name": "G. Ochoa", "pos": "G", "grid": None}}],
        },
        {
            "team": {"id": 1531, "name": "South Africa"},
            "coach": {"id": 2, "name": "H. Broos"},
            "formation": "4-3-3",
            "startXI": [{"player": {"id": 50, "name": "R. Williams", "pos": "G", "grid": "1:1"}}],
            "substitutes": [],
        },
    ],
}


def test_parse_lineup_response():
    lineups = parse_lineup_response(SAMPLE, fixture_id=1489369)
    assert len(lineups) == 2
    mex = next(lu for lu in lineups if lu.team_id == 16)
    assert mex.formation == "4-1-4-1"
    assert mex.start_ids == (270774, 11)
    assert mex.sub_ids == (2098,)


def test_parse_empty_response_returns_empty():
    assert parse_lineup_response({"response": []}, fixture_id=1) == []


def test_fetcher_without_key_returns_none(tmp_path, monkeypatch):
    from soccer_agent.worldcup.live import LineupFetcher
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    f = LineupFetcher(cache_dir=tmp_path)
    assert f.fetch_fixture_lineups(1489369) is None
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_worldcup_live.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `live.py`**

```python
# soccer_agent/worldcup/live.py
"""Optional API-Sports v3 lineup fetcher with an on-disk cache.

Never raises on network/key problems: returns None so callers fall back to projected
lineups. The key is read only from the environment (git-ignored .env).
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from soccer_agent.worldcup.entities import Lineup, WorldCup

BASE_URL = "https://v3.football.api-sports.io"
_CACHE_NAME = "lineups_fixture={}.json"


def parse_lineup_response(payload: dict, fixture_id: int) -> list[Lineup]:
    """Parse an API-Sports /fixtures/lineups response into Lineup objects."""
    out: list[Lineup] = []
    for side in payload.get("response", []):
        team = side.get("team") or {}
        start = tuple(p["player"]["id"] for p in side.get("startXI", []) if p.get("player", {}).get("id"))
        subs = tuple(p["player"]["id"] for p in side.get("substitutes", []) if p.get("player", {}).get("id"))
        out.append(Lineup(
            fixture_id=fixture_id,
            team_id=int(team.get("id", 0)),
            formation=str(side.get("formation") or "4-3-3"),
            start_ids=start,
            sub_ids=subs,
        ))
    return out


class LineupFetcher:
    def __init__(self, cache_dir: str | Path | None = None, timeout: float = 10.0):
        repo_root = Path(__file__).resolve().parents[2]
        self.cache_dir = Path(cache_dir) if cache_dir else repo_root / "data" / "live"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    @property
    def _key(self) -> str | None:
        return os.getenv("API_FOOTBALL_KEY")

    def fetch_fixture_lineups(self, fixture_id: int) -> list[Lineup] | None:
        """Return lineups for a fixture (cached on disk after first fetch). None if unavailable."""
        cache_path = self.cache_dir / _CACHE_NAME.format(fixture_id)
        if cache_path.exists():
            return parse_lineup_response(json.loads(cache_path.read_text()), fixture_id)
        key = self._key
        if not key:
            return None
        url = f"{BASE_URL}/fixtures/lineups?fixture={fixture_id}"
        req = urllib.request.Request(url, headers={"x-apisports-key": key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError):
            return None
        cache_path.write_text(json.dumps(payload))
        return parse_lineup_response(payload, fixture_id)

    def recent_team_lineup(self, wc: WorldCup, team_id: int) -> Lineup | None:
        """Most-recent *played* WC match lineup for a team, or None."""
        played = sorted(
            (m for m in wc.matches if m.played and team_id in (m.home_id, m.away_id)),
            key=lambda m: m.kickoff,
        )
        for m in reversed(played):
            lineups = self.fetch_fixture_lineups(m.fixture_id)
            if not lineups:
                continue
            for lu in lineups:
                if lu.team_id == team_id and lu.start_ids:
                    return lu
        return None
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_worldcup_live.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add soccer_agent/worldcup/live.py tests/test_worldcup_live.py
git commit -m "feat(wc): add optional API-Sports lineup fetcher with disk cache"
```

---

### Task 7: Lineup projection (curated formations + squad projection + live/prior)

**Files:**
- Create: `soccer_agent/worldcup/lineup.py`
- Test: `tests/test_worldcup_lineup.py`

**Interfaces:**
- Consumes: `WorldCup`, `Rankings`, `LineupFetcher | None`.
- Produces: `ProjectedLineup` dataclass `{team_id, formation, start_ids, sub_ids, source, source_matchday}`; `project_lineup(wc, rankings, team_id, fixture_id, fetcher=None) -> ProjectedLineup`. `formation_slots(formation) -> (def, mid, fwd)`. Source is `"live"` (fetched recent lineup), `"prior"` (an attached dataset lineup from an earlier matchday), or `"projected"` (curated formation + squad ratings).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worldcup_lineup.py
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.lineup import project_lineup, formation_slots


def test_formation_slots():
    assert formation_slots("4-3-3") == (4, 3, 3)
    assert formation_slots("4-2-3-1") == (4, 5, 1)
    assert formation_slots("3-5-2") == (3, 5, 2)
    assert formation_slots("nonsense") == (4, 3, 3)


def test_projected_lineup_has_eleven_starters_seven_subs():
    wc = load_worldcup()
    rankings = rank_all(wc)
    any_team = next(iter(wc.teams.values()))
    lu = project_lineup(wc, rankings, any_team.id, fixture_id=0)
    assert len(lu.start_ids) == 11
    assert len(lu.sub_ids) == 7
    assert lu.formation in {"4-3-3", "4-2-3-1", "4-1-4-1", "3-5-2", "4-4-2", "3-4-3", "5-3-2"}
    assert lu.source == "projected"  # no fetcher, empty dataset lineups


def test_projected_starters_match_formation_shape():
    wc = load_worldcup()
    rankings = rank_all(wc)
    any_team = next(iter(wc.teams.values()))
    lu = project_lineup(wc, rankings, any_team.id, fixture_id=0)
    starters = [wc.players[pid] for pid in lu.start_ids]
    from collections import Counter
    pos = Counter(p.position for p in starters)
    d, m, f = formation_slots(lu.formation)
    assert pos["Goalkeeper"] == 1
    assert pos["Defender"] == d
    assert pos["Midfielder"] + pos["Attacker"] == m + f  # allow M/F flex
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_worldcup_lineup.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `lineup.py`**

```python
# soccer_agent/worldcup/lineup.py
"""Project each team's formation, starting XI, and subs for an upcoming match.

Precedence: a live-fetched recent lineup ("live") > an attached dataset lineup from an
earlier matchday ("prior") > a curated-formation + squad-rating projection ("projected").
"""
from __future__ import annotations

from dataclasses import dataclass

from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.ranking import Rankings

DEFAULT_FORMATION = "4-3-3"
SUB_COUNT = 7
_NEUTRAL = 50.0
_POS_GROUP = {"Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Attacker": "FWD"}

# Curated real coach formations (override the squad-derived default where well-known).
FORMATIONS: dict[str, str] = {
    "Argentina": "4-3-3", "France": "4-2-3-1", "Brazil": "4-3-3", "England": "4-2-3-1",
    "Spain": "4-3-3", "Germany": "4-2-3-1", "Portugal": "4-3-3", "Netherlands": "3-4-3",
    "Belgium": "3-4-3", "Croatia": "4-3-3", "Mexico": "4-1-4-1", "USA": "4-2-3-1",
    "Norway": "4-4-2", "Morocco": "4-3-3", "Japan": "4-2-3-1", "Colombia": "4-2-3-1",
    "Uruguay": "4-3-3", "Switzerland": "3-4-2-1", "Senegal": "4-3-3", "Ecuador": "4-2-3-1",
    "Australia": "4-4-2", "South Korea": "4-2-3-1", "Iran": "4-3-3", "Saudi Arabia": "4-2-3-1",
    "Canada": "4-4-2", "Türkiye": "4-2-3-1", "Austria": "4-2-3-1", "Sweden": "4-3-3",
    "Czech Republic": "4-2-3-1", "Ivory Coast": "4-3-3", "Egypt": "4-2-3-1",
    "Scotland": "5-3-2", "Paraguay": "4-3-3", "Ghana": "4-2-3-1", "Tunisia": "4-3-3",
    "Algeria": "4-2-3-1", "Congo DR": "4-3-3", "Iraq": "4-3-3", "Jordan": "4-4-2",
    "Qatar": "3-5-2", "Uzbekistan": "4-2-3-1", "Panama": "4-4-2", "Cape Verde Islands": "4-3-3",
    "Curaçao": "4-3-3", "Haiti": "4-4-2", "New Zealand": "4-4-2", "Bosnia & Herzegovina": "4-2-3-1",
    "South Africa": "4-3-3",
}


@dataclass(frozen=True)
class ProjectedLineup:
    team_id: int
    formation: str
    start_ids: tuple[int, ...]
    sub_ids: tuple[int, ...]
    source: str  # "live" | "prior" | "projected"
    source_matchday: int | None = None


def formation_slots(formation: str) -> tuple[int, int, int]:
    """(defenders, midfielders, forwards) from e.g. '4-2-3-1'. Falls back to 4-3-3."""
    try:
        nums = [int(part) for part in formation.split("-")]
    except ValueError:
        nums = []
    if len(nums) < 2:
        return (4, 3, 3)
    return (nums[0], sum(nums[1:-1]), nums[-1])


def _group(pos: str) -> str:
    return _POS_GROUP.get(pos, "MID")


def _curated_or_default(wc: WorldCup, team_id: int) -> str:
    return FORMATIONS.get(wc.teams[team_id].name, DEFAULT_FORMATION)


def _project_xi(wc: WorldCup, rankings: Rankings, team_id: int, formation: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    squad = sorted(
        wc.squad(team_id),
        key=lambda p: rankings.players.get(p.id, _NEUTRAL),
        reverse=True,
    )
    d, m, f = formation_slots(formation)
    need = {"GK": 1, "DEF": d, "MID": m, "FWD": f}
    by_group: dict[str, list] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in squad:
        by_group[_group(p.position)].append(p)

    chosen: list[int] = []
    chosen_set: set[int] = set()
    for group, count in need.items():
        for p in by_group[group][:count]:
            chosen.append(p.id)
            chosen_set.add(p.id)
    # Backfill to 11 from best remaining if a group is short.
    for p in squad:
        if len(chosen) >= 11:
            break
        if p.id not in chosen_set:
            chosen.append(p.id)
            chosen_set.add(p.id)

    start = tuple(chosen[:11])
    start_set = set(start)
    subs = tuple(p.id for p in squad if p.id not in start_set)[:SUB_COUNT]
    return start, subs


def project_lineup(
    wc: WorldCup,
    rankings: Rankings,
    team_id: int,
    fixture_id: int,
    fetcher=None,
) -> ProjectedLineup:
    # 1) Live-fetched recent lineup (most recent played WC match), if a fetcher is provided.
    if fetcher is not None:
        live = fetcher.recent_team_lineup(wc, team_id)
        if live is not None and len(live.start_ids) >= 11:
            # Find the matchday of that recent fixture for provenance.
            md = next((m.matchday for m in wc.matches if m.fixture_id == live.fixture_id), None)
            return ProjectedLineup(team_id, live.formation, live.start_ids[:11], live.sub_ids[:SUB_COUNT], "live", md)

    # 2) Dataset-attached prior lineup from an earlier matchday (dataset lineups are usually empty).
    target = next((m for m in wc.matches if m.fixture_id == fixture_id), None)
    target_md = target.matchday if target else 99
    prior = None
    prior_md = -1
    for lu in wc.lineups:
        if lu.team_id != team_id:
            continue
        md = next((m.matchday for m in wc.matches if m.fixture_id == lu.fixture_id), -1)
        if md < target_md and md > prior_md:
            prior, prior_md = lu, md
    if prior is not None and len(prior.start_ids) >= 11:
        return ProjectedLineup(team_id, prior.formation, prior.start_ids[:11], prior.sub_ids[:SUB_COUNT], "prior", prior_md)

    # 3) Curated formation + squad-rating projection.
    formation = _curated_or_default(wc, team_id)
    start, subs = _project_xi(wc, rankings, team_id, formation)
    return ProjectedLineup(team_id, formation, start, subs, "projected", None)
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_worldcup_lineup.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add soccer_agent/worldcup/lineup.py tests/test_worldcup_lineup.py
git commit -m "feat(wc): add lineup projection with curated formations and live/prior fallback"
```

---

### Task 8: Poisson + Dixon-Coles prediction model

**Files:**
- Create: `soccer_agent/worldcup/predict.py`
- Test: `tests/test_worldcup_predict.py`

**Interfaces:**
- Consumes: `WorldCup`, `Rankings`, `strengths: dict[int,float]` (recalibrated), `ProjectedLineup`.
- Produces: `MatchPrediction` dataclass (fixture_id, matchday, group, kickoff, venue, home_id, away_id, home_name, away_name, lambda_home, lambda_away, score_home, score_away, p_home, p_draw, p_away, rationale, home_adjustment, away_adjustment) with `to_dict()`; `predict_one(wc, rankings, strengths, fixture_id, home_lu, away_lu) -> MatchPrediction`; `top_scorelines(lh, la, n) -> list[tuple[int,int,float]]`; `scoreline_matrix(lh, la) -> list[list[float]]`; `effective_rating(wc, rankings, strengths, team_id, lineup, is_home, venue) -> tuple[float, float]` (effective, adjustment).

Constants: `BASE_MATCH_GOALS=2.6`, `SUPREMACY_PER_10=0.62`, `LAMBDA_FLOOR=0.18`, `MAX_GOALS=8`, `DRAW_RHO=-0.15`, `HOST_HOME_FIELD=4.0`, `TRAVEL_PENALTY={CONCACAF:0,CONMEBOL:0.5,UEFA:1.0,CAF:1.5,AFC:2.0,OFC:2.5}`, `WEATHER_PENALTY=0.8`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worldcup_predict.py
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.lineup import project_lineup
from soccer_agent.worldcup.predict import predict_one, top_scorelines, scoreline_matrix


def _strengths():
    wc = load_worldcup()
    r = rank_all(wc)
    f = compute_forms(wc)
    return wc, r, recalibrated_strength(wc, r, f)


def test_probs_sum_to_one():
    wc, r, s = _strengths()
    m = next(m for m in wc.matches if m.matchday == 0)
    hlu = project_lineup(wc, r, m.home_id, m.fixture_id)
    alu = project_lineup(wc, r, m.away_id, m.fixture_id)
    pred = predict_one(wc, r, s, m.fixture_id, hlu, alu)
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-9
    assert pred.lambda_home >= 0.18 and pred.lambda_away >= 0.18


def test_modal_score_is_matrix_argmax():
    lh, la = 1.5, 1.2
    mat = scoreline_matrix(lh, la)
    pred = predict_one  # noqa: F841 (just to ensure import path works)
    tops = top_scorelines(lh, la, 3)
    best_h, best_a, best_p = tops[0]
    assert abs(best_p - max(max(row) for row in mat)) < 1e-9


def test_stronger_team_favored():
    wc, r, s = _strengths()
    m = next(m for m in wc.matches if m.matchday == 0)
    hlu = project_lineup(wc, r, m.home_id, m.fixture_id)
    alu = project_lineup(wc, r, m.away_id, m.fixture_id)
    pred = predict_one(wc, r, s, m.fixture_id, hlu, alu)
    if s[m.home_id] - s[m.away_id] > 10:
        assert pred.p_home > pred.p_away
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_worldcup_predict.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `predict.py`**

```python
# soccer_agent/worldcup/predict.py
"""Independent-Poisson scoreline model with Dixon-Coles low-score correction.

Effective rating = blend of recalibrated team strength and projected-XI mean player rating,
adjusted for host-nation home field, inter-confederation travel, and hot-venue weather.
The rating gap becomes a goal supremacy that splits a baseline match total into two Poisson
means; the scoreline matrix yields the modal exact score and W/D/L probabilities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.lineup import ProjectedLineup
from soccer_agent.worldcup.ranking import Rankings

BASE_MATCH_GOALS = 2.6
SUPREMACY_PER_10 = 0.62
LAMBDA_FLOOR = 0.18
MAX_GOALS = 8
DRAW_RHO = -0.15
HOST_HOME_FIELD = 4.0
TRAVEL_PENALTY = {"CONCACAF": 0.0, "CONMEBOL": 0.5, "UEFA": 1.0, "CAF": 1.5, "AFC": 2.0, "OFC": 2.5}
WEATHER_PENALTY = 0.8
_HOT_VENUE_HINTS = ("Miami", "Houston", "Dallas", "Arlington", "Atlanta", "Monterrey", "Guadalajara", "Kansas City")
_HEAT_SENSITIVE = {"UEFA"}
_NEUTRAL = 50.0
W_XI = 0.5  # weight on projected-XI mean rating vs team strength


@dataclass(frozen=True)
class MatchPrediction:
    fixture_id: int
    matchday: int
    group: str
    kickoff: datetime
    venue: str
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    lambda_home: float
    lambda_away: float
    score_home: int
    score_away: int
    p_home: float
    p_draw: float
    p_away: float
    rationale: str
    home_adjustment: float = 0.0
    away_adjustment: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id, "matchday": self.matchday, "group": self.group,
            "kickoff": self.kickoff.isoformat(), "venue": self.venue,
            "home_id": self.home_id, "away_id": self.away_id,
            "home_name": self.home_name, "away_name": self.away_name,
            "lambda_home": round(self.lambda_home, 3), "lambda_away": round(self.lambda_away, 3),
            "score_home": self.score_home, "score_away": self.score_away,
            "p_home": round(self.p_home, 4), "p_draw": round(self.p_draw, 4), "p_away": round(self.p_away, 4),
            "rationale": self.rationale,
            "home_adjustment": round(self.home_adjustment, 3), "away_adjustment": round(self.away_adjustment, 3),
        }


def _poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def scoreline_matrix(lh: float, la: float) -> list[list[float]]:
    """9x9 (0..MAX_GOALS) scoreline probability matrix with Dixon-Coles low-score correction."""
    mat = [[_poisson(i, lh) * _poisson(j, la) for j in range(MAX_GOALS + 1)] for i in range(MAX_GOALS + 1)]
    # Dixon-Coles: adjust 0-0, 1-0, 0-1, 1-1.
    tau = lambda i, j: 1.0 - DRAW_RHO * _poisson(i, lh) * _poisson(j, la) if (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)] else 1.0  # noqa: E731
    adj = [[mat[i][j] * tau(i, j) for j in range(MAX_GOALS + 1)] for i in range(MAX_GOALS + 1)]
    total = sum(sum(row) for row in adj)
    return [[v / total for v in row] for row in adj]


def top_scorelines(lh: float, la: float, n: int) -> list[tuple[int, int, float]]:
    mat = scoreline_matrix(lh, la)
    cells = [(i, j, mat[i][j]) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)]
    cells.sort(key=lambda c: c[2], reverse=True)
    return [(i, j, p) for i, j, p in cells[:n]]


def _xi_mean_rating(wc: WorldCup, rankings: Rankings, lineup: ProjectedLineup) -> float:
    if not lineup.start_ids:
        return _NEUTRAL
    vals = [rankings.players.get(pid, _NEUTRAL) for pid in lineup.start_ids]
    return sum(vals) / len(vals)


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


def predict_one(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float],
    fixture_id: int, home_lu: ProjectedLineup, away_lu: ProjectedLineup,
) -> MatchPrediction:
    m = next((x for x in wc.matches if x.fixture_id == fixture_id), None)
    if m is None:
        raise ValueError(f"fixture {fixture_id} not found")
    eff_h, adj_h = effective_rating(wc, rankings, strengths, m.home_id, home_lu, True, m.venue)
    eff_a, adj_a = effective_rating(wc, rankings, strengths, m.away_id, away_lu, False, m.venue)

    supremacy = (eff_h - eff_a) / 10.0 * SUPREMACY_PER_10
    total = BASE_MATCH_GOALS
    lh = max(LAMBDA_FLOOR, total / 2.0 + supremacy / 2.0)
    la = max(LAMBDA_FLOOR, total / 2.0 - supremacy / 2.0)

    mat = scoreline_matrix(lh, la)
    p_home = sum(mat[i][j] for i in range(MAX_GOALS + 1) for j in range(i))
    p_away = sum(mat[i][j] for i in range(MAX_GOALS + 1) for j in range(i + 1, MAX_GOALS + 1))
    p_draw = sum(mat[i][i] for i in range(MAX_GOALS + 1))
    # modal exact score:
    best = max(((i, j) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)), key=lambda ij: mat[ij[0]][ij[1]])
    sh, sa = best
    rationale = (
        f"Eff {eff_h:.1f} vs {eff_a:.1f} -> supremacy {supremacy:+.2f}; "
        f"xG {lh:.2f}-{la:.2f}; adj {adj_h:+.1f}/{adj_a:+.1f}."
    )
    return MatchPrediction(
        fixture_id=m.fixture_id, matchday=m.matchday, group=m.group, kickoff=m.kickoff,
        venue=m.venue, home_id=m.home_id, away_id=m.away_id,
        home_name=wc.teams[m.home_id].name, away_name=wc.teams[m.away_id].name,
        lambda_home=lh, lambda_away=la, score_home=sh, score_away=sa,
        p_home=p_home, p_draw=p_draw, p_away=p_away, rationale=rationale,
        home_adjustment=adj_h, away_adjustment=adj_a,
    )
```

- [ ] **Step 4: Run tests — expect PASS**

Run: `pytest tests/test_worldcup_predict.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add soccer_agent/worldcup/predict.py tests/test_worldcup_predict.py
git commit -m "feat(wc): add Poisson + Dixon-Coles prediction model"
```

---

### Task 9: Standings + bracket + Monte-Carlo simulation

**Files:**
- Create: `soccer_agent/worldcup/standings.py`
- Create: `soccer_agent/worldcup/bracket.py`
- Create: `soccer_agent/worldcup/simulate.py`
- Test: `tests/test_worldcup_bracket.py`

**Interfaces:**
- Consumes: `WorldCup`, `Rankings`, `strengths`, `LineupFetcher | None`.
- Produces:
  - `standings.group_standings(wc) -> dict[str, list[StandingRow]]` where `StandingRow{team_id, name, played, wins, draws, losses, gf, ga, gd, pts}` sorted by FIFA tiebreakers.
  - `bracket.build_bracket(wc) -> Bracket` with `r32: list[int]` (fixture_ids sorted) and a pairing tree `pairs: list[tuple[int,int]]` (R16 pairs of R32 indices), plus `slots` for QF/SF/Final derived by binary-tree reduction.
  - `simulate.simulate_bracket(wc, rankings, strengths, fetcher=None, n=10000) -> BracketSim` with `champion: dict[int,float]` (sums to 1.0), `advancement: dict[int, dict[str,float]]` (per team per round), `r32_predictions: list[MatchPrediction]`, `modal_path: list[dict]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worldcup_bracket.py
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.standings import group_standings
from soccer_agent.worldcup.bracket import build_bracket
from soccer_agent.worldcup.simulate import simulate_bracket


def _setup():
    wc = load_worldcup()
    r = rank_all(wc)
    f = compute_forms(wc)
    return wc, r, recalibrated_strength(wc, r, f)


def test_group_standings_twelve_groups_four_each():
    wc, _, _ = _setup()
    gs = group_standings(wc)
    assert len(gs) == 12
    for rows in gs.values():
        assert len(rows) == 4
        # sorted by points desc
        assert rows[0].pts >= rows[-1].pts


def test_bracket_has_sixteen_r32_and_tree():
    wc, _, _ = _setup()
    b = build_bracket(wc)
    assert len(b.r32) == 16
    assert len(b.pairs) == 8  # 8 R16 ties


def test_simulation_champion_mass_sums_to_one():
    wc, r, s = _setup()
    sim = simulate_bracket(wc, r, s, fetcher=None, n=500)
    assert len(sim.r32_predictions) == 16
    total = sum(sim.champion.values())
    assert abs(total - 1.0) < 1e-6
    # exactly one champion team has the max
    assert max(sim.champion.values()) > 0.0
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_worldcup_bracket.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `standings.py`**

```python
# soccer_agent/worldcup/standings.py
"""Final group standings from results, with FIFA tiebreakers (pts -> GD -> GF -> head-to-head)."""
from __future__ import annotations

from dataclasses import dataclass

from soccer_agent.worldcup.entities import WorldCup


@dataclass(frozen=True)
class StandingRow:
    team_id: int
    name: str
    played: int
    wins: int
    draws: int
    losses: int
    gf: int
    ga: int
    gd: int
    pts: int


def _h2h_rank(rows: list[StandingRow], h2h_pts: dict[int, int]) -> list[StandingRow]:
    # Stable secondary sort key using head-to-head points when available.
    return sorted(rows, key=lambda r: (r.pts, r.gd, r.gf, h2h_pts.get(r.team_id, 0), r.name), reverse=True)


def group_standings(wc: WorldCup) -> dict[str, list[StandingRow]]:
    groups = wc.groups()
    out: dict[str, list[StandingRow]] = {}
    for gname, teams in groups.items():
        agg: dict[int, dict] = {t.id: {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0} for t in teams}
        matches = [m for m in wc.matches if m.group == gname and m.played]
        for m in matches:
            ha, ga = m.home_goals, m.away_goals
            agg[m.home_id]["p"] += 1; agg[m.away_id]["p"] += 1
            agg[m.home_id]["gf"] += ha; agg[m.home_id]["ga"] += ga
            agg[m.away_id]["gf"] += ga; agg[m.away_id]["ga"] += ha
            if ha > ga:
                agg[m.home_id]["w"] += 1; agg[m.away_id]["l"] += 1
            elif ha < ga:
                agg[m.away_id]["w"] += 1; agg[m.home_id]["l"] += 1
            else:
                agg[m.home_id]["d"] += 1; agg[m.away_id]["d"] += 1
        rows = [StandingRow(
            tid, wc.teams[tid].name, a["p"], a["w"], a["d"], a["l"], a["gf"], a["ga"],
            a["gf"] - a["ga"], 3 * a["w"] + a["d"],
        ) for tid, a in agg.items()]
        out[gname] = _h2h_rank(rows, {})  # head-to-head omitted for simplicity; pts/GD/GF suffice
    return out
```

- [ ] **Step 4: Implement `bracket.py`**

```python
# soccer_agent/worldcup/bracket.py
"""Round-of-32 fixtures (from the dataset) + an approximated R16->Final binary tree.

The dataset carries the 16 real R32 pairings but no R16 slot map, so R32 matches are
paired into R16 by sorted fixture_id (match 1 vs 2, 3 vs 4, ...). This pairing is an
approximation; R32 itself is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

from soccer_agent.worldcup.entities import WorldCup


@dataclass(frozen=True)
class Bracket:
    r32: tuple[int, ...]            # fixture_ids, sorted
    pairs: tuple[tuple[int, int], ...]  # 8 R16 pairs of R32 indices (into r32)


def build_bracket(wc: WorldCup) -> Bracket:
    r32 = tuple(sorted(m.fixture_id for m in wc.matches if m.matchday == 0))
    pairs = tuple((r32[i], r32[i + 1]) for i in range(0, len(r32), 2))
    return Bracket(r32=r32, pairs=pairs)
```

- [ ] **Step 5: Implement `simulate.py`**

```python
# soccer_agent/worldcup/simulate.py
"""Monte-Carlo simulation of the knockout bracket to a champion.

Each R32 match is predicted once (deterministic) for the printed card; the bracket is then
walked by sampling match outcomes from the Poisson scoreline matrix. Knockout ties that are
drawn after 90' go to extra time (lambdas * 4/3) and, if still level, a shootout whose win
prob is shifted by the rating edge (capped at +/-0.15).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from soccer_agent.worldcup.bracket import Bracket, build_bracket
from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.lineup import project_lineup
from soccer_agent.worldcup.predict import MatchPrediction, predict_one, scoreline_matrix
from soccer_agent.worldcup.ranking import Rankings

ET_FACTOR = 4.0 / 3.0
PEN_EDGE_PER_10 = 0.03
PEN_EDGE_CAP = 0.15


@dataclass
class BracketSim:
    r32_predictions: list[MatchPrediction] = field(default_factory=list)
    champion: dict[int, float] = field(default_factory=dict)
    advancement: dict[int, dict[str, float]] = field(default_factory=dict)
    modal_path: list[dict[str, Any]] = field(default_factory=list)


def _sample_winner(lh: float, la: float, eff_h: float, eff_a: float, rng: random.Random) -> int:
    """Return 1 if home wins the tie, 2 if away wins (after ET + pens if needed)."""
    mat = scoreline_matrix(lh, la)
    flat = [(i, j, mat[i][j]) for i in range(len(mat)) for j in range(len(mat))]
    r = rng.random()
    cum = 0.0
    sh = sa = 0
    for i, j, p in flat:
        cum += p
        if r <= cum:
            sh, sa = i, j
            break
    if sh != sa:
        return 1 if sh > sa else 2
    # Extra time.
    et = scoreline_matrix(lh * ET_FACTOR, la * ET_FACTOR)
    flat = [(i, j, et[i][j]) for i in range(len(et)) for j in range(len(et))]
    r = rng.random()
    cum = 0.0
    for i, j, p in flat:
        cum += p
        if r <= cum:
            sh, sa = i, j
            break
    if sh != sa:
        return 1 if sh > sa else 2
    # Shootout: shift 0.5 by rating edge.
    edge = (eff_h - eff_a) / 10.0 * PEN_EDGE_PER_10
    edge = max(-PEN_EDGE_CAP, min(PEN_EDGE_CAP, edge))
    return 1 if rng.random() < (0.5 + edge) else 2


def simulate_bracket(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float],
    fetcher=None, n: int = 10000, seed: int = 2026,
) -> BracketSim:
    bracket: Bracket = build_bracket(wc)
    rng = random.Random(seed)

    # Predict each R32 match deterministically for the cards.
    r32_preds: list[MatchPrediction] = []
    r32_map: dict[int, MatchPrediction] = {}
    for fid in bracket.r32:
        m = next(x for x in wc.matches if x.fixture_id == fid)
        hlu = project_lineup(wc, rankings, m.home_id, fid, fetcher)
        alu = project_lineup(wc, rankings, m.away_id, fid, fetcher)
        pred = predict_one(wc, rankings, strengths, fid, hlu, alu)
        r32_preds.append(pred)
        r32_map[fid] = pred

    champion: dict[int, float] = {tid: 0.0 for tid in wc.teams}
    rounds = ["R32", "R16", "QF", "SF", "Final"]

    for _ in range(n):
        # current_winners: list of team_ids at each slot, seeded from R32 fixtures.
        slots: list[int] = []
        for fid in bracket.r32:
            m = next(x for x in wc.matches if x.fixture_id == fid)
            pred = r32_map[fid]
            eff_h = pred.home_adjustment + strengths.get(m.home_id, 50.0)
            eff_a = pred.away_adjustment + strengths.get(m.away_id, 50.0)
            w = _sample_winner(pred.lambda_home, pred.lambda_away, eff_h, eff_a, rng)
            slots.append(m.home_id if w == 1 else m.away_id)
        # R16 -> Final: pair adjacent slots.
        round_idx = 1
        while len(slots) > 1:
            nxt: list[int] = []
            for i in range(0, len(slots), 2):
                home_id, away_id = slots[i], slots[i + 1]
                lh = max(0.18, 1.3 + (strengths.get(home_id, 50.0) - strengths.get(away_id, 50.0)) / 10.0 * 0.62 / 2)
                la = max(0.18, 2.6 - lh)
                eff_h = strengths.get(home_id, 50.0)
                eff_a = strengths.get(away_id, 50.0)
                w = _sample_winner(lh, la, eff_h, eff_a, rng)
                nxt.append(home_id if w == 1 else away_id)
            slots = nxt
            round_idx += 1
        champion[slots[0]] += 1.0

    for tid in champion:
        champion[tid] /= n

    return BracketSim(r32_predictions=r32_preds, champion=champion, advancement={}, modal_path=[])
```

- [ ] **Step 6: Run tests — expect PASS**

Run: `pytest tests/test_worldcup_bracket.py -v`
Expected: 3 passed.

- [ ] **Step 7: Commit**

```bash
git add soccer_agent/worldcup/standings.py soccer_agent/worldcup/bracket.py soccer_agent/worldcup/simulate.py tests/test_worldcup_bracket.py
git commit -m "feat(wc): add standings, bracket tree, and Monte-Carlo simulation"
```

---

### Task 10: Match card + PDF renderer

**Files:**
- Create: `soccer_agent/worldcup/card.py`
- Create: `soccer_agent/worldcup/cardpdf.py`
- Test: `tests/test_worldcup_card.py`

**Interfaces:**
- Consumes: `WorldCup`, `Rankings`, `strengths`, `LineupFetcher | None`.
- Produces: `PlayerLine{player_id,name,position,rating}`, `TeamCard{team_id,name,coach_name,coach_record,formation,starters,subs,source,source_matchday}`, `MatchCard{fixture_id,group,kickoff,venue,home,away,prediction,top_scorelines}` (all with `to_dict()`); `build_card(wc, rankings, strengths, home_id, away_id, fetcher=None, fixture_id=None) -> MatchCard`; `render_card_pdf(card, path)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worldcup_card.py
import json

import pytest

from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.card import build_card


def _setup():
    wc = load_worldcup()
    r = rank_all(wc)
    f = compute_forms(wc)
    return wc, r, recalibrated_strength(wc, r, f)


def test_build_card_structure():
    wc, r, s = _setup()
    m = next(m for m in wc.matches if m.matchday == 0)
    card = build_card(wc, r, s, m.home_id, m.away_id, fixture_id=m.fixture_id)
    assert card.home.name and card.away.name
    assert len(card.home.starters) == 11
    assert len(card.home.subs) == 7
    assert card.home.coach_name is not None
    assert card.prediction is not None
    d = card.to_dict()
    json.dumps(d)  # serializable


def test_render_card_pdf_skips_without_reportlab(tmp_path):
    pytest.importorskip("reportlab")  # skip if not installed
    from soccer_agent.worldcup.cardpdf import render_card_pdf
    wc, r, s = _setup()
    m = next(m for m in wc.matches if m.matchday == 0)
    card = build_card(wc, r, s, m.home_id, m.away_id, fixture_id=m.fixture_id)
    out = tmp_path / "x.pdf"
    render_card_pdf(card, out)
    assert out.exists() and out.stat().st_size > 100
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_worldcup_card.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `card.py`**

```python
# soccer_agent/worldcup/card.py
"""Assemble a single-match preview card: lineups, coaches, and a lineup-aware prediction."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.lineup import project_lineup
from soccer_agent.worldcup.predict import MatchPrediction, predict_one, top_scorelines
from soccer_agent.worldcup.ranking import Rankings

_NEUTRAL = 50.0


@dataclass(frozen=True)
class PlayerLine:
    player_id: int
    name: str
    position: str
    rating: float

    def to_dict(self) -> dict[str, Any]:
        return {"player_id": self.player_id, "name": self.name, "position": self.position, "rating": self.rating}


@dataclass(frozen=True)
class TeamCard:
    team_id: int
    name: str
    coach_name: Optional[str]
    coach_record: Optional[tuple[int, int, int]]
    formation: str
    starters: tuple[PlayerLine, ...]
    subs: tuple[PlayerLine, ...]
    source: str
    source_matchday: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id, "name": self.name,
            "coach_name": self.coach_name,
            "coach_record": list(self.coach_record) if self.coach_record else None,
            "formation": self.formation,
            "starters": [p.to_dict() for p in self.starters],
            "subs": [p.to_dict() for p in self.subs],
            "source": self.source, "source_matchday": self.source_matchday,
        }


@dataclass(frozen=True)
class MatchCard:
    fixture_id: Optional[int]
    group: str
    kickoff: Optional[datetime]
    venue: str
    home: TeamCard
    away: TeamCard
    prediction: MatchPrediction
    top_scorelines: tuple[tuple[int, int, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id, "group": self.group,
            "kickoff": self.kickoff.isoformat() if self.kickoff else None,
            "venue": self.venue,
            "home": self.home.to_dict(), "away": self.away.to_dict(),
            "prediction": self.prediction.to_dict(),
            "top_scorelines": [list(s) for s in self.top_scorelines],
        }


def _player_line(wc: WorldCup, rankings: Rankings, pid: int) -> PlayerLine:
    p = wc.players.get(pid)
    rating = round(rankings.players.get(pid, _NEUTRAL), 1)
    if p is None:
        return PlayerLine(pid, f"#{pid}", "?", rating)
    return PlayerLine(pid, p.name, p.position, rating)


def _team_card(wc: WorldCup, rankings: Rankings, team_id: int, lineup) -> TeamCard:
    team = wc.teams[team_id]
    coach = wc.coaches.get(team.coach_id) if team.coach_id is not None else None
    return TeamCard(
        team_id=team_id, name=team.name,
        coach_name=coach.name if coach else None,
        coach_record=(coach.wins, coach.draws, coach.losses) if coach else None,
        formation=lineup.formation,
        starters=tuple(_player_line(wc, rankings, pid) for pid in lineup.start_ids),
        subs=tuple(_player_line(wc, rankings, pid) for pid in lineup.sub_ids),
        source=lineup.source, source_matchday=lineup.source_matchday,
    )


def build_card(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float],
    home_id: int, away_id: int, fetcher=None, fixture_id: int | None = None,
) -> MatchCard:
    # Prefer a real dataset fixture for kickoff/venue/group; else synthesize a neutral card.
    m = None
    if fixture_id is not None:
        m = next((x for x in wc.matches if x.fixture_id == fixture_id), None)
    if m is None:
        m = next((x for x in wc.matches if x.matchday == 0 and {x.home_id, x.away_id} == {home_id, away_id}), None)
    fid = m.fixture_id if m else None
    group = m.group if m and m.group else "Knockout"
    kickoff = m.kickoff if m else None
    venue = m.venue if m else "TBD"

    hlu = project_lineup(wc, rankings, home_id, fid or 0, fetcher)
    alu = project_lineup(wc, rankings, away_id, fid or 0, fetcher)

    if fid is not None:
        pred = predict_one(wc, rankings, strengths, fid, hlu, alu)
    else:
        # Synthesize a prediction with a transient fixture entry is complex; reuse an R32 fixture's
        # home/away by swapping ids is fragile. Instead, require a fixture for prediction.
        raise ValueError("build_card requires a fixture_id (use an R32 fixture)")

    tops = tuple(top_scorelines(pred.lambda_home, pred.lambda_away, 3))
    return MatchCard(
        fixture_id=fid, group=group, kickoff=kickoff, venue=venue,
        home=_team_card(wc, rankings, home_id, hlu),
        away=_team_card(wc, rankings, away_id, alu),
        prediction=pred, top_scorelines=tops,
    )
```

- [ ] **Step 4: Implement `cardpdf.py`**

```python
# soccer_agent/worldcup/cardpdf.py
"""Render a MatchCard to a one-page A4 PDF. reportlab is imported lazily."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soccer_agent.worldcup.card import MatchCard, TeamCard

_INSTALL_HINT = "PDF output requires reportlab; install with: pip install 'soccer-agent[pdf]'"


def render_card_pdf(card: "MatchCard", path: str | Path) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover
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

    kickoff = card.kickoff.strftime("%Y-%m-%d %H:%M %Z").strip() if card.kickoff else "TBD"
    line(f"{card.home.name}  vs  {card.away.name}", size=16, gap=8)
    line(f"{card.group}  ·  {kickoff}  ·  {card.venue}", size=9, gap=8)

    pred = card.prediction
    line(
        f"Prediction: {pred.home_name} {pred.score_home}-{pred.score_away} {pred.away_name}"
        f"   (W {pred.p_home:.0%} / D {pred.p_draw:.0%} / L {pred.p_away:.0%})",
        size=12, gap=6,
    )
    line(f"Expected goals: {pred.lambda_home:.2f} - {pred.lambda_away:.2f}", size=9)
    tops = ", ".join(f"{h}-{a} ({p:.0%})" for h, a, p in card.top_scorelines)
    line(f"Most likely scorelines: {tops}", size=9, gap=7)
    line(pred.rationale, size=8, gap=8)

    def team_block(team: "TeamCard", x: float) -> None:
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
        cursor = top

    block_top = cursor
    team_block(card.home, left)
    cursor = block_top
    team_block(card.away, left + (right - left) / 2.0)

    pdf.showPage()
    pdf.save()
```

- [ ] **Step 5: Install reportlab and run tests — expect PASS**

```bash
pip install reportlab>=4.0
pytest tests/test_worldcup_card.py -v
```
Expected: 2 passed (PDF test runs; if reportlab somehow missing it skips).

- [ ] **Step 6: Commit**

```bash
git add soccer_agent/worldcup/card.py soccer_agent/worldcup/cardpdf.py tests/test_worldcup_card.py
git commit -m "feat(wc): add match preview card and PDF renderer"
```

---

### Task 11: CLI + generate batch outputs

**Files:**
- Create: `soccer_agent/worldcup/cli.py`
- Create: `soccer_agent/worldcup/__main__.py`
- Test: `tests/test_worldcup_cli.py`

**Interfaces:**
- Produces: `python -m soccer_agent.worldcup predict` writes `predictions/worldcup-2026-predictions-after1st-group.{md,json}`; `python -m soccer_agent.worldcup card "Home" "Away"` writes `predictions/<Home>-vs-<Away>.{pdf,json}`; `python -m soccer_agent.worldcup bracket` prints champion odds. `cli.main(argv) -> int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_worldcup_cli.py
import json
from pathlib import Path

from soccer_agent.worldcup.cli import main


def test_predict_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(["predict"])
    assert code == 0
    md = tmp_path / "predictions" / "worldcup-2026-predictions-after1st-group.md"
    js = tmp_path / "predictions" / "worldcup-2026-predictions-after1st-group.json"
    assert md.exists() and js.exists()
    data = json.loads(js.read_text())
    assert "standings" in data and "r32" in data and "bracket" in data
    assert len(data["r32"]) == 16


def test_card_writes_pdf_and_json(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("reportlab")
    wc = load_worldcup()
    m = next(m for m in wc.matches if m.matchday == 0)  # first real R32 fixture
    home = wc.teams[m.home_id].name
    away = wc.teams[m.away_id].name
    monkeypatch.chdir(tmp_path)
    code = main(["card", home, away])
    assert code == 0
    pdfs = list((tmp_path / "predictions").glob("*.pdf"))
    assert pdfs, "expected a PDF card"
```

- [ ] **Step 2: Run test — expect FAIL**

Run: `pytest tests/test_worldcup_cli.py -v`
Expected: FAIL, module not found.

- [ ] **Step 3: Implement `cli.py`**

```python
# soccer_agent/worldcup/cli.py
"""Command-line interface for the World Cup predictor.

Commands:
  predict            -> predictions/worldcup-2026-predictions-after1st-group.{md,json}
  card "Home" "Away" -> predictions/<Home>-vs-<Away>.{pdf,json}
  bracket            -> print champion + advancement odds to stdout
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from soccer_agent.worldcup.card import build_card
from soccer_agent.worldcup.cardpdf import render_card_pdf
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.live import LineupFetcher
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.simulate import simulate_bracket
from soccer_agent.worldcup.standings import group_standings

PRED_DIR = Path("predictions")


def _engine():
    wc = load_worldcup()
    rankings = rank_all(wc)
    forms = compute_forms(wc)
    strengths = recalibrated_strength(wc, rankings, forms)
    fetcher = LineupFetcher() if os.getenv("API_FOOTBALL_KEY") else None
    return wc, rankings, strengths, fetcher


def _team_by_name(wc, name: str):
    name_l = name.lower().strip()
    for t in wc.teams.values():
        if t.name.lower() == name_l or t.name.lower().startswith(name_l):
            return t
    raise SystemExit(f"team not found: {name}")


def _write_predictions(wc, rankings, strengths, fetcher) -> int:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    sim = simulate_bracket(wc, rankings, strengths, fetcher=fetcher, n=10000)
    gs = group_standings(wc)

    # Standings dict
    standings_json = {g: [r.__dict__ for r in rows] for g, rows in gs.items()}
    r32_json = [p.to_dict() for p in sim.r32_predictions]
    champ_sorted = sorted(sim.champion.items(), key=lambda kv: kv[1], reverse=True)[:10]
    bracket_json = {
        "champion_top10": [{"team": wc.teams[t].name, "probability": round(p, 4)} for t, p in champ_sorted],
        "method": "Monte-Carlo 10000 iters; R32 fixtures exact, R16+ bracket pairing approximated by sorted fixture_id.",
    }
    payload = {"standings": standings_json, "r32": r32_json, "bracket": bracket_json}
    (PRED_DIR / "worldcup-2026-predictions-after1st-group.json").write_text(json.dumps(payload, indent=2))

    # Markdown
    lines = ["# FIFA 2026 World Cup — Predictions (after group stage)", ""]
    lines.append("Group stage complete. Below: final group standings, all 16 Round-of-32 "
                 "predictions, and a Monte-Carlo bracket simulation to the champion.")
    lines.append("")
    lines.append("## Group standings")
    for g, rows in gs.items():
        lines.append(f"\n### {g}\n")
        lines.append("| Team | P | W | D | L | GF | GA | GD | Pts |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(f"| {r.name} | {r.played} | {r.wins} | {r.draws} | {r.losses} | {r.gf} | {r.ga} | {r.gd} | {r.pts} |")
    lines.append("\n## Round of 32\n")
    for p in sim.r32_predictions:
        ko = p.kickoff.strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"- `{ko}`  **{p.home_name} {p.score_home}-{p.score_away} {p.away_name}**  "
                     f"(W {p.p_home:.0%} / D {p.p_draw:.0%} / L {p.p_away:.0%})  — {p.rationale}")
    lines.append("\n## Bracket simulation (Monte-Carlo, 10000 iters)\n")
    lines.append("R32 pairings are the real fixtures; R16→Final pairing is approximated "
                 "(sorted by fixture_id).\n")
    lines.append("**Champion probabilities (top 10):**\n")
    for t, p in champ_sorted:
        lines.append(f"- {wc.teams[t].name}: {p:.1%}")
    (PRED_DIR / "worldcup-2026-predictions-after1st-group.md").write_text("\n".join(lines))
    return 0


def _write_card(wc, rankings, strengths, fetcher, home_name: str, away_name: str) -> int:
    home = _team_by_name(wc, home_name)
    away = _team_by_name(wc, away_name)
    # Find an R32 fixture matching these two teams (either order).
    m = next((x for x in wc.matches if x.matchday == 0 and {x.home_id, x.away_id} == {home.id, away.id}), None)
    if m is None:
        raise SystemExit(f"no R32 fixture between {home.name} and {away.name}")
    card = build_card(wc, rankings, strengths, home.id, away.id, fetcher=fetcher, fixture_id=m.fixture_id)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{home.name}-vs-{away.name}"
    (PRED_DIR / f"{stem}.json").write_text(json.dumps(card.to_dict(), indent=2))
    try:
        render_card_pdf(card, PRED_DIR / f"{stem}.pdf")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print("usage: python -m soccer_agent.worldcup {predict|card|bracket} [...]", file=sys.stderr)
        return 2
    cmd = argv[0]
    wc, rankings, strengths, fetcher = _engine()
    if cmd == "predict":
        return _write_predictions(wc, rankings, strengths, fetcher)
    if cmd == "card":
        if len(argv) < 3:
            print("usage: card \"Home\" \"Away\"", file=sys.stderr)
            return 2
        return _write_card(wc, rankings, strengths, fetcher, argv[1], argv[2])
    if cmd == "bracket":
        sim = simulate_bracket(wc, rankings, strengths, fetcher=fetcher, n=10000)
        for t, p in sorted(sim.champion.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"{wc.teams[t].name}: {p:.1%}")
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2
```

- [ ] **Step 4: Implement `__main__.py`**

```python
# soccer_agent/worldcup/__main__.py
import sys

from soccer_agent.worldcup.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests — expect PASS**

Run: `pytest tests/test_worldcup_cli.py -v`
Expected: 2 passed (PDF test runs if reportlab installed).

- [ ] **Step 6: Commit**

```bash
git add soccer_agent/worldcup/cli.py soccer_agent/worldcup/__main__.py tests/test_worldcup_cli.py
git commit -m "feat(wc): add CLI and generate after-group predictions + match cards"
```

---

### Task 12: Final integration — run the pipeline end-to-end and verify outputs

**Files:**
- Run the CLI; verify outputs; run the full suite.

- [ ] **Step 1: Run the full test suite**

```bash
pytest -q
```
Expected: all green.

- [ ] **Step 2: Generate the batch predictions**

```bash
python -m soccer_agent.worldcup predict
```
Expected: exit 0; `predictions/worldcup-2026-predictions-after1st-group.{md,json}` written with 16 R32 predictions + champion odds.

- [ ] **Step 3: Generate a sample match card PDF**

Derive the first real R32 pairing from the dataset, then render its card:

```bash
PAIR=$(python -c "from soccer_agent.worldcup.dataset import load_worldcup as l; wc=l(); m=[x for x in wc.matches if x.matchday==0][0]; print(f'{wc.teams[m.home_id].name}|{wc.teams[m.away_id].name}')")
HOME="${PAIR%|*}"; AWAY="${PAIR#*|}"
python -m soccer_agent.worldcup card "$HOME" "$AWAY"
```
Expected: exit 0; `predictions/<Home>-vs-<Away>.pdf` + `.json` written; PDF opens with coach, formation, starting XI, subs, prediction.

- [ ] **Step 4: Sanity-check the JSON contents**

```bash
python -c "import json; d=json.load(open('predictions/worldcup-2026-predictions-after1st-group.json')); print(len(d['r32']), 'r32; top champ:', d['bracket']['champion_top10'][0])"
```
Expected: `16 r32; top champ: <team> <prob>`.

- [ ] **Step 5: Lint**

```bash
ruff check soccer_agent/worldcup tests/test_worldcup_*.py
```
Expected: clean (fix any E/F/I/N/W issues).

- [ ] **Step 6: Confirm .env is NOT staged**

```bash
git status --porcelain | grep '\.env' && echo "ABORT" || echo "env-safe"
git add predictions docs/superpowers/plans/2026-06-28-worldcup-2026-knockout-predictor.md
git commit -m "data(wc): generate after-group predictions and sample match card"
```

- [ ] **Step 7: Final commit of any remaining test/plan artifacts**

```bash
git add -A
git status --porcelain | grep '\.env' && echo "ABORT" || git commit -m "chore(wc): finalize predictor" || true
```
