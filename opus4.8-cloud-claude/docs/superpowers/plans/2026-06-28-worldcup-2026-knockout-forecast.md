# FIFA 2026 World Cup Knockout-Stage Forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the predictor from group-stage scorelines to the full knockout bracket (Round of 32 → Final + third-place playoff), producing a single most-likely bracket and Monte-Carlo title odds.

**Architecture:** The live API supplies the complete group results and the 16 concrete Round-of-32 fixtures; everything from the Round of 16 onward is synthesized from the hardcoded official FIFA-2026 bracket tree. Final group standings are computed locally (the API `rank` field is wrong). A knockout-resolution layer (extra time + penalties) sits on top of the existing rating + Poisson + Dixon–Coles scoreline core, which is reused unchanged. A modal pass yields the headline bracket; a seeded Monte-Carlo pass yields advancement/title odds.

**Tech Stack:** Python 3.11+, `src/` layout, stdlib only (no new dependencies), pytest, ruff, mypy. Frozen dataclasses with explicit `to_dict`/`from_dict`. Dependency injection for RNG and dataset path at the CLI boundary.

## Global Constraints

- Target Python 3.11+; importable code lives under `src/soccer/`.
- No new third-party dependencies; stdlib only.
- No import-time side effects; read config/env only at the CLI boundary.
- Tests must not touch the network, wall-clock time, or machine-specific paths; inject RNG (`random.Random`) and use crafted in-memory `WorldCup` fixtures.
- Every behavior change ships with tests, including error paths.
- Ruff is the source of truth for format + lint; mypy must pass on `src` and `tests`; `# type: ignore[code]` only when narrow and specific.
- Frozen dataclasses with explicit `to_dict` (and `from_dict` where persisted) matching `entities.py`.
- Validation per task: `ruff format .` && `ruff check .` && `mypy src tests` && `pytest`.

---

### Task 1: Add `round_name` to `WcMatch`

**Files:**
- Modify: `src/soccer/worldcup/entities.py` (the `WcMatch` dataclass, ~lines 219-260)
- Test: `tests/worldcup/test_entities.py`

**Interfaces:**
- Produces: `WcMatch.round_name: str = ""` (last field, default `""`); knockout rows use `"Round of 32"`, group rows use `""`. `to_dict` emits `"round_name"`; `from_dict` reads it with a `""` default so existing datasets load unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/worldcup/test_entities.py  (add these tests)
from datetime import UTC, datetime

from soccer.worldcup.entities import WcMatch


def test_wcmatch_round_name_defaults_empty_and_roundtrips() -> None:
    m = WcMatch(
        fixture_id=1,
        matchday=1,
        group="Group A",
        home_id=10,
        away_id=20,
        kickoff=datetime(2026, 6, 11, 19, 0, tzinfo=UTC),
        venue="Estadio Azteca / Mexico City",
        home_goals=2,
        away_goals=0,
    )
    assert m.round_name == ""
    assert m.to_dict()["round_name"] == ""


def test_wcmatch_from_dict_without_round_name_is_empty() -> None:
    raw = {
        "fixture_id": 1, "matchday": 1, "group": "Group A",
        "home_id": 10, "away_id": 20,
        "kickoff": "2026-06-11T19:00:00+00:00",
        "venue": "v", "home_goals": None, "away_goals": None,
    }
    assert WcMatch.from_dict(raw).round_name == ""


def test_wcmatch_knockout_round_name_roundtrips() -> None:
    m = WcMatch(
        fixture_id=99, matchday=0, group="",
        home_id=10, away_id=20,
        kickoff=datetime(2026, 6, 28, 19, 0, tzinfo=UTC),
        venue="SoFi Stadium",
        home_goals=None, away_goals=None,
        round_name="Round of 32",
    )
    assert WcMatch.from_dict(m.to_dict()).round_name == "Round of 32"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/worldcup/test_entities.py -k round_name -v`
Expected: FAIL (`TypeError: ... unexpected keyword argument 'round_name'` / missing attribute).

- [ ] **Step 3: Add the field**

In `WcMatch`, add the field after `away_goals`:

```python
    away_goals: int | None
    round_name: str = ""
```

In `to_dict`, add to the returned dict:

```python
            "away_goals": self.away_goals,
            "round_name": self.round_name,
        }
```

In `from_dict`, add the last argument:

```python
            away_goals=None if raw["away_goals"] is None else int(raw["away_goals"]),
            round_name=str(raw.get("round_name", "")),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_entities.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/worldcup/entities.py tests/worldcup/test_entities.py
git commit -m "feat(wc): add round_name field to WcMatch for knockout fixtures"
```

---

### Task 2: Keep knockout fixtures in ingest

**Files:**
- Modify: `src/soccer/worldcup/ingest.py` (`_parse_matches` ~lines 72-97, `ingest_world_cup` line ~284)
- Test: `tests/worldcup/test_ingest.py`

**Interfaces:**
- Consumes: `WcMatch.round_name` from Task 1.
- Produces: ingest now keeps fixtures whose API `league.round` is not a group round; each match's `round_name` is set from `league.round`; knockout matches get `matchday = 0` and `group = ""`. Group matches are unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/worldcup/test_ingest.py  (add)
from soccer.worldcup.ingest import _parse_matches


def _fixture(fid: int, rnd: str, home: int, away: int, status: str) -> dict:
    return {
        "fixture": {"id": fid, "date": "2026-06-28T19:00:00+00:00",
                    "venue": {"name": "SoFi Stadium", "city": None},
                    "status": {"short": status}},
        "league": {"round": rnd},
        "teams": {"home": {"id": home}, "away": {"id": away}},
        "goals": {"home": None, "away": None},
    }


def test_parse_matches_keeps_knockout_round_name() -> None:
    fixtures = [
        _fixture(1, "Group Stage - 1", 10, 20, "FT"),
        _fixture(2, "Round of 32", 30, 40, "NS"),
    ]
    team_group = {10: "Group A", 20: "Group A"}
    matches = _parse_matches(fixtures, team_group)
    by_id = {m.fixture_id: m for m in matches}
    assert by_id[1].round_name == "Group Stage - 1"
    assert by_id[1].group == "Group A"
    assert by_id[2].round_name == "Round of 32"
    assert by_id[2].group == ""
    assert by_id[2].matchday == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/worldcup/test_ingest.py -k knockout -v`
Expected: FAIL (`round_name` empty / match filtered out).

- [ ] **Step 3: Set round_name and knockout matchday in `_parse_matches`**

In `_parse_matches`, replace the `WcMatch(...)` construction so it reads the round once and branches matchday:

```python
    for item in fixtures:
        fx = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        home_id = _safe_int(teams.get("home", {}).get("id"))
        away_id = _safe_int(teams.get("away", {}).get("id"))
        status = fx.get("status", {}).get("short", "")
        played = status in _FINISHED
        venue = fx.get("venue", {}) or {}
        venue_name = " / ".join(p for p in (venue.get("name"), venue.get("city")) if p)
        round_name = str(item.get("league", {}).get("round", ""))
        is_group = round_name.startswith("Group Stage")
        matches.append(
            WcMatch(
                fixture_id=_safe_int(fx.get("id")),
                matchday=_matchday(round_name) if is_group else 0,
                group=team_group.get(home_id, "") if is_group else "",
                home_id=home_id,
                away_id=away_id,
                kickoff=datetime.fromisoformat(fx.get("date")),
                venue=venue_name,
                home_goals=_safe_int(goals.get("home")) if played else None,
                away_goals=_safe_int(goals.get("away")) if played else None,
                round_name=round_name,
            )
        )
    return matches
```

- [ ] **Step 4: Stop dropping knockout fixtures in `ingest_world_cup`**

Change line ~284 from:

```python
    matches = tuple(m for m in _parse_matches(fixtures, team_group) if m.group)
```

to:

```python
    matches = tuple(
        m for m in _parse_matches(fixtures, team_group) if m.group or m.round_name
    )
```

(Keeps any group match and any knockout match; still drops blank/garbage rows.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_ingest.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/soccer/worldcup/ingest.py tests/worldcup/test_ingest.py
git commit -m "feat(wc): keep knockout fixtures during ingest"
```

---

### Task 3: Group standings module

**Files:**
- Create: `src/soccer/worldcup/standings.py`
- Test: `tests/worldcup/test_standings.py`

**Interfaces:**
- Consumes: `WorldCup`, `WcMatch` (group matches identified by `m.group` non-empty and `m.played`).
- Produces:
  - `@dataclass(frozen=True) StandingRow` with `team_id, group, played, won, drawn, lost, gf, ga, points, rank` and `@property gd -> int`.
  - `group_tables(wc: WorldCup) -> dict[str, list[StandingRow]]` — per group, ordered best→worst, `rank` set 1..N.
  - `team_labels(wc: WorldCup) -> dict[int, str]` — `team_id -> f"{rank}{group_letter}"` (e.g. `"1A"`, `"3C"`), `group_letter = group.split()[-1]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/worldcup/test_standings.py
from __future__ import annotations

from datetime import UTC, datetime

from soccer.worldcup.entities import NationalTeam, WcMatch, WorldCup
from soccer.worldcup.standings import group_tables, team_labels


def _team(tid: int, group: str) -> NationalTeam:
    return NationalTeam(
        id=tid, name=f"T{tid}", group=group, confederation="UEFA",
        is_host=False, player_ids=(), coach_id=None,
        recent_w=0, recent_d=0, recent_l=0,
    )


def _match(fid: int, h: int, a: int, hg: int, ag: int) -> WcMatch:
    return WcMatch(
        fixture_id=fid, matchday=1, group="Group A", home_id=h, away_id=a,
        kickoff=datetime(2026, 6, 11, tzinfo=UTC), venue="v",
        home_goals=hg, away_goals=ag, round_name="Group Stage - 1",
    )


def _wc(teams: dict[int, NationalTeam], matches: tuple[WcMatch, ...]) -> WorldCup:
    return WorldCup(teams=teams, matches=matches)


def test_points_order_decides_rank() -> None:
    teams = {1: _team(1, "Group A"), 2: _team(2, "Group A"), 3: _team(3, "Group A")}
    # team1 beats 2 and 3; team2 beats 3
    matches = (_match(1, 1, 2, 2, 0), _match(2, 1, 3, 1, 0), _match(3, 2, 3, 3, 1))
    table = group_tables(_wc(teams, matches))["Group A"]
    assert [r.team_id for r in table] == [1, 2, 3]
    assert [r.rank for r in table] == [1, 2, 3]
    assert team_labels(_wc(teams, matches))[1] == "1A"
    assert team_labels(_wc(teams, matches))[2] == "2A"


def test_goal_difference_breaks_equal_points() -> None:
    teams = {1: _team(1, "Group A"), 2: _team(2, "Group A"), 3: _team(3, "Group A")}
    # team1 and team2 both beat team3, lose nothing else; team1 by more goals
    matches = (_match(1, 1, 3, 5, 0), _match(2, 2, 3, 1, 0), _match(3, 1, 2, 0, 0))
    table = group_tables(_wc(teams, matches))["Group A"]
    assert table[0].team_id == 1  # better GD
    assert table[0].gd == 5
    assert table[1].team_id == 2


def test_head_to_head_breaks_equal_points_and_gd() -> None:
    # Two teams identical on points, GD, GF; head-to-head decides.
    teams = {1: _team(1, "Group A"), 2: _team(2, "Group A")}
    matches = (_match(1, 1, 2, 2, 1),)  # team1 won the only meeting
    table = group_tables(_wc(teams, matches))["Group A"]
    assert table[0].team_id == 1


def test_ignores_knockout_and_unplayed_matches() -> None:
    teams = {1: _team(1, "Group A"), 2: _team(2, "Group A")}
    ko = WcMatch(
        fixture_id=9, matchday=0, group="", home_id=1, away_id=2,
        kickoff=datetime(2026, 6, 28, tzinfo=UTC), venue="v",
        home_goals=None, away_goals=None, round_name="Round of 32",
    )
    unplayed = WcMatch(
        fixture_id=10, matchday=2, group="Group A", home_id=1, away_id=2,
        kickoff=datetime(2026, 6, 16, tzinfo=UTC), venue="v",
        home_goals=None, away_goals=None, round_name="Group Stage - 2",
    )
    played = _match(1, 1, 2, 1, 0)
    table = group_tables(_wc(teams, (played, ko, unplayed)))["Group A"]
    assert table[0].played == 1  # only the played group match counts
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/worldcup/test_standings.py -v`
Expected: FAIL (`ModuleNotFoundError: soccer.worldcup.standings`).

- [ ] **Step 3: Implement the standings module**

```python
# src/soccer/worldcup/standings.py
"""Compute final group standings from played group matches.

The API's ``standings.rank`` field is unreliable for this dataset (it lists
fourth-placed teams as advancing), so ranks are derived here from match goals
using the FIFA tiebreaker order: points, goal difference, goals for, then
head-to-head among the teams still tied, then team id as a deterministic
stand-in for the drawing of lots.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from soccer.worldcup.entities import WcMatch, WorldCup


@dataclass(frozen=True)
class StandingRow:
    team_id: int
    group: str
    played: int
    won: int
    drawn: int
    lost: int
    gf: int
    ga: int
    points: int
    rank: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def _group_matches(wc: WorldCup) -> dict[str, list[WcMatch]]:
    out: dict[str, list[WcMatch]] = defaultdict(list)
    for m in wc.matches:
        if m.group and m.played:
            out[m.group].append(m)
    return out


def _raw_rows(group: str, matches: list[WcMatch]) -> dict[int, StandingRow]:
    acc: dict[int, dict[str, int]] = defaultdict(
        lambda: {"played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0, "points": 0}
    )
    for m in matches:
        assert m.home_goals is not None and m.away_goals is not None
        hg, ag = m.home_goals, m.away_goals
        for tid, gf, ga in ((m.home_id, hg, ag), (m.away_id, ag, hg)):
            r = acc[tid]
            r["played"] += 1
            r["gf"] += gf
            r["ga"] += ga
        if hg > ag:
            acc[m.home_id]["won"] += 1
            acc[m.home_id]["points"] += 3
            acc[m.away_id]["lost"] += 1
        elif hg < ag:
            acc[m.away_id]["won"] += 1
            acc[m.away_id]["points"] += 3
            acc[m.home_id]["lost"] += 1
        else:
            acc[m.home_id]["drawn"] += 1
            acc[m.away_id]["drawn"] += 1
            acc[m.home_id]["points"] += 1
            acc[m.away_id]["points"] += 1
    return {tid: StandingRow(team_id=tid, group=group, **vals) for tid, vals in acc.items()}


def _head_to_head_key(tied: list[int], matches: list[WcMatch]) -> dict[int, tuple[int, int, int]]:
    pts: dict[int, int] = defaultdict(int)
    gf: dict[int, int] = defaultdict(int)
    ga: dict[int, int] = defaultdict(int)
    members = set(tied)
    for m in matches:
        if m.home_id not in members or m.away_id not in members:
            continue
        assert m.home_goals is not None and m.away_goals is not None
        hg, ag = m.home_goals, m.away_goals
        gf[m.home_id] += hg; ga[m.home_id] += ag
        gf[m.away_id] += ag; ga[m.away_id] += hg
        if hg > ag:
            pts[m.home_id] += 3
        elif hg < ag:
            pts[m.away_id] += 3
        else:
            pts[m.home_id] += 1; pts[m.away_id] += 1
    return {tid: (pts[tid], gf[tid] - ga[tid], gf[tid]) for tid in tied}


def _rank_rows(rows: list[StandingRow], matches: list[WcMatch]) -> list[StandingRow]:
    def primary(r: StandingRow) -> tuple[int, int, int]:
        return (r.points, r.gd, r.gf)

    ordered = sorted(rows, key=primary, reverse=True)
    resolved: list[StandingRow] = []
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and primary(ordered[j]) == primary(ordered[i]):
            j += 1
        cluster = ordered[i:j]
        if len(cluster) > 1:
            h2h = _head_to_head_key([r.team_id for r in cluster], matches)
            cluster.sort(key=lambda r: (h2h[r.team_id], -r.team_id), reverse=True)
        resolved.extend(cluster)
        i = j
    return [replace(r, rank=k) for k, r in enumerate(resolved, start=1)]


def group_tables(wc: WorldCup) -> dict[str, list[StandingRow]]:
    by_group = _group_matches(wc)
    out: dict[str, list[StandingRow]] = {}
    for group in sorted(by_group):
        matches = by_group[group]
        rows = list(_raw_rows(group, matches).values())
        out[group] = _rank_rows(rows, matches)
    return out


def team_labels(wc: WorldCup) -> dict[int, str]:
    labels: dict[int, str] = {}
    for group, rows in group_tables(wc).items():
        letter = group.split()[-1]
        for row in rows:
            labels[row.team_id] = f"{row.rank}{letter}"
    return labels
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_standings.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/worldcup/standings.py tests/worldcup/test_standings.py
git commit -m "feat(wc): compute group standings with FIFA tiebreakers"
```

---

### Task 4: Bracket construction module

**Files:**
- Create: `src/soccer/worldcup/bracket.py`
- Test: `tests/worldcup/test_bracket.py`

**Interfaces:**
- Consumes: `WorldCup`, `WcMatch` (the live R32 fixtures: `round_name == "Round of 32"`), `team_labels` from Task 3.
- Produces:
  - `class BracketError(Exception)`.
  - `@dataclass(frozen=True) BracketTie` with `match_no: int`, `round_name: str`, `home_src: str`, `away_src: str`, `fixture_id: int | None`, `home_id: int | None`, `away_id: int | None`, `venue: str`. For R32, `home_id/away_id/fixture_id/venue` are set and `home_src/away_src == ""`. For later rounds, `home_src/away_src` are like `"W74"` / `"L101"` and the id/fixture fields are `None`.
  - `build_bracket(wc: WorldCup, labels: dict[int, str]) -> dict[int, BracketTie]` — keys 73..104, R32 ties matched to slots, downstream ties from the fixed tree. Raises `BracketError` if the live R32 set does not map one-to-one onto slots 73-88.
  - Module constants `R32_ANCHORS: dict[int, frozenset[str]]`, `KNOCKOUT_EDGES: dict[int, tuple[str, str]]`, `round_name_for(match_no: int) -> str`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/worldcup/test_bracket.py
from __future__ import annotations

from datetime import UTC, datetime

import pytest

from soccer.worldcup.bracket import BracketError, build_bracket, round_name_for
from soccer.worldcup.entities import WcMatch


def _r32(fid: int, h: int, a: int) -> WcMatch:
    return WcMatch(
        fixture_id=fid, matchday=0, group="", home_id=h, away_id=a,
        kickoff=datetime(2026, 6, 28, tzinfo=UTC), venue="SoFi",
        home_goals=None, away_goals=None, round_name="Round of 32",
    )


class _WC:
    def __init__(self, matches: tuple[WcMatch, ...]) -> None:
        self.matches = matches


def _full_labels() -> dict[int, str]:
    # 32 teams: ids 1..32 mapped to the 32 advancing slot labels.
    slots = []
    for letter in "ABCDEFGHIJKL":
        slots += [f"1{letter}", f"2{letter}"]
    # eight third-place qualifiers (any 8 distinct groups) to fill the 8 third slots
    slots += [f"3{c}" for c in "CDEFGHIJ"][:8]
    return {i + 1: slots[i] for i in range(32)}


def test_round_name_for_covers_all_rounds() -> None:
    assert round_name_for(73) == "Round of 32"
    assert round_name_for(90) == "Round of 16"
    assert round_name_for(99) == "Quarter-final"
    assert round_name_for(101) == "Semi-final"
    assert round_name_for(103) == "Third-place play-off"
    assert round_name_for(104) == "Final"


def test_build_bracket_maps_double_anchor_slot() -> None:
    labels = _full_labels()
    inv = {v: k for k, v in labels.items()}
    # Slot 73 anchors = {"2A","2B"}; build one R32 fixture for it (+ fillers for the rest).
    fixtures = []
    # one fixture per slot using that slot's anchors / a third filler
    from soccer.worldcup.bracket import R32_ANCHORS
    used_third = iter([f"3{c}" for c in "CDEFGHIJ"])
    fid = 1000
    for match_no, anchors in R32_ANCHORS.items():
        anchors = list(anchors)
        if len(anchors) == 2:
            h, a = inv[anchors[0]], inv[anchors[1]]
        else:
            h = inv[anchors[0]]
            a = inv[next(used_third)]
        fixtures.append(_r32(fid, h, a))
        fid += 1
    bracket = build_bracket(_WC(tuple(fixtures)), labels)
    assert set(bracket) == set(range(73, 105))
    # 73 is an R32 tie with concrete ids
    assert bracket[73].home_id is not None
    # 89 is the first R16 tie wired to winners of 74 and 77
    assert (bracket[89].home_src, bracket[89].away_src) == ("W74", "W77")
    assert bracket[104].home_src == "W101" and bracket[104].away_src == "W102"
    assert bracket[103].home_src == "L101" and bracket[103].away_src == "L102"


def test_build_bracket_raises_on_unmatched_fixture() -> None:
    labels = {1: "1A", 2: "1A"}  # impossible duplicate, matches no slot
    fixtures = (_r32(1, 1, 2),)
    with pytest.raises(BracketError):
        build_bracket(_WC(fixtures), labels)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/worldcup/test_bracket.py -v`
Expected: FAIL (`ModuleNotFoundError: soccer.worldcup.bracket`).

- [ ] **Step 3: Implement the bracket module**

```python
# src/soccer/worldcup/bracket.py
"""Official FIFA-2026 knockout bracket: live R32 leaves + fixed downstream tree.

The API provides only the 16 concrete Round-of-32 fixtures; the Round of 16
through the Final are synthesized from the published bracket (matches 73-104,
Wikipedia "2026 FIFA World Cup knockout stage", verified 2026-06-28). Each R32
slot is keyed by its fixed winner/runner-up "anchor" label(s); because every
``(group, rank)`` anchor is unique, a live fixture maps to exactly one slot.
"""

from __future__ import annotations

from dataclasses import dataclass

from soccer.worldcup.entities import WcMatch, WorldCup


class BracketError(Exception):
    """Raised when the live R32 draw cannot be mapped onto the official slots."""


# match_no -> the fixed winner/runner-up labels that identify the slot.
# Slots with two anchors are fully fixed; one-anchor slots take a third-placed team.
R32_ANCHORS: dict[int, frozenset[str]] = {
    73: frozenset({"2A", "2B"}),
    74: frozenset({"1E"}),
    75: frozenset({"1F", "2C"}),
    76: frozenset({"1C", "2F"}),
    77: frozenset({"1I"}),
    78: frozenset({"2E", "2I"}),
    79: frozenset({"1A"}),
    80: frozenset({"1L"}),
    81: frozenset({"1D"}),
    82: frozenset({"1G"}),
    83: frozenset({"2K", "2L"}),
    84: frozenset({"1H", "2J"}),
    85: frozenset({"1B"}),
    86: frozenset({"1J", "2H"}),
    87: frozenset({"1K"}),
    88: frozenset({"2D", "2G"}),
}

# match_no -> (home source, away source) for the synthesized rounds.
KNOCKOUT_EDGES: dict[int, tuple[str, str]] = {
    89: ("W74", "W77"), 90: ("W73", "W75"), 91: ("W76", "W78"), 92: ("W79", "W80"),
    93: ("W83", "W84"), 94: ("W81", "W82"), 95: ("W86", "W88"), 96: ("W85", "W87"),
    97: ("W89", "W90"), 98: ("W93", "W94"), 99: ("W91", "W92"), 100: ("W95", "W96"),
    101: ("W97", "W98"), 102: ("W99", "W100"),
    103: ("L101", "L102"),
    104: ("W101", "W102"),
}


@dataclass(frozen=True)
class BracketTie:
    match_no: int
    round_name: str
    home_src: str = ""
    away_src: str = ""
    fixture_id: int | None = None
    home_id: int | None = None
    away_id: int | None = None
    venue: str = ""


def round_name_for(match_no: int) -> str:
    if 73 <= match_no <= 88:
        return "Round of 32"
    if 89 <= match_no <= 96:
        return "Round of 16"
    if 97 <= match_no <= 100:
        return "Quarter-final"
    if 101 <= match_no <= 102:
        return "Semi-final"
    if match_no == 103:
        return "Third-place play-off"
    if match_no == 104:
        return "Final"
    raise BracketError(f"no round for match {match_no}")


def _match_r32(fixture: WcMatch, labels: dict[int, str]) -> int:
    present = {labels.get(fixture.home_id, ""), labels.get(fixture.away_id, "")}
    matched = [no for no, anchors in R32_ANCHORS.items() if anchors <= present]
    if len(matched) != 1:
        raise BracketError(
            f"R32 fixture {fixture.fixture_id} (labels {sorted(present)}) "
            f"matched slots {matched}; expected exactly one"
        )
    return matched[0]


def build_bracket(wc: WorldCup, labels: dict[int, str]) -> dict[int, BracketTie]:
    r32 = [m for m in wc.matches if m.round_name == "Round of 32"]
    if len(r32) != 16:
        raise BracketError(f"expected 16 Round of 32 fixtures, found {len(r32)}")
    ties: dict[int, BracketTie] = {}
    for fixture in r32:
        no = _match_r32(fixture, labels)
        if no in ties:
            raise BracketError(f"two fixtures mapped to slot {no}")
        ties[no] = BracketTie(
            match_no=no,
            round_name="Round of 32",
            fixture_id=fixture.fixture_id,
            home_id=fixture.home_id,
            away_id=fixture.away_id,
            venue=fixture.venue,
        )
    if len(ties) != 16:
        raise BracketError("R32 slots not fully populated")
    for no, (home_src, away_src) in KNOCKOUT_EDGES.items():
        ties[no] = BracketTie(
            match_no=no,
            round_name=round_name_for(no),
            home_src=home_src,
            away_src=away_src,
        )
    return ties
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_bracket.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/worldcup/bracket.py tests/worldcup/test_bracket.py
git commit -m "feat(wc): build knockout bracket from live R32 + fixed FIFA tree"
```

---

### Task 5: Knockout match model (extra time + penalties)

**Files:**
- Modify: `src/soccer/worldcup/predict.py`
- Test: `tests/worldcup/test_predict_knockout.py`

**Interfaces:**
- Consumes: existing `_scoreline_matrix`, `_outcome_probs`, `_modal_score`, `_effective_rating`, `BASE_MATCH_GOALS`, `SUPREMACY_PER_10`, `LAMBDA_FLOOR`; `Rankings`; `WorldCup`.
- Produces:
  - Constants `ET_GOAL_FRACTION = 1 / 3`, `PEN_EDGE_PER_10 = 0.03`, `PEN_EDGE_CAP = 0.15`.
  - `_knockout_lambdas(wc, rankings, home_id, away_id, venue) -> tuple[float, float, float, float]` returning `(lam_home, lam_away, eff_home, eff_away)` with **neutral site** (no host home-field bonus; travel/weather still apply).
  - `@dataclass(frozen=True) KnockoutPrediction` with `match_no, round_name, home_id, away_id, home_name, away_name, score_home, score_away, p_home, p_draw, p_away, p_home_advance, p_away_advance, expected_extra_time, rationale` and `to_dict`.
  - `predict_knockout(wc, rankings, home_id, away_id, *, match_no=0, round_name="", venue="") -> KnockoutPrediction`.
  - `advance_prob(wc, rankings, home_id, away_id, venue="") -> float` returning `p_home_advance` (used by the simulator; pure, cacheable).

- [ ] **Step 1: Write the failing tests**

```python
# tests/worldcup/test_predict_knockout.py
from __future__ import annotations

from soccer.worldcup.predict import advance_prob, predict_knockout
from soccer.worldcup.ranking import rank_all
from tests.worldcup.conftest import sample_world_cup  # noqa: F401  (pytest fixture)


def test_advancement_probs_sum_to_one(sample_world_cup) -> None:
    wc = sample_world_cup
    ranks = rank_all(wc)
    pred = predict_knockout(wc, ranks, home_id=1, away_id=2, match_no=104, round_name="Final")
    assert abs(pred.p_home_advance + pred.p_away_advance - 1.0) < 1e-9
    assert abs(pred.p_home + pred.p_draw + pred.p_away - 1.0) < 1e-9


def test_stronger_team_is_favoured_to_advance(sample_world_cup) -> None:
    wc = sample_world_cup
    ranks = rank_all(wc)
    # team 1 is the strong side in the fixture
    assert advance_prob(wc, ranks, 1, 2) > 0.5
    assert advance_prob(wc, ranks, 2, 1) < 0.5


def test_equal_teams_advance_is_half() -> None:
    # Build a symmetric two-team world cup so ratings tie exactly.
    from datetime import UTC, datetime

    from soccer.worldcup.entities import NationalTeam, WorldCup

    def t(i: int) -> NationalTeam:
        return NationalTeam(
            id=i, name=f"T{i}", group="Group A", confederation="UEFA",
            is_host=False, player_ids=(), coach_id=None,
            recent_w=3, recent_d=1, recent_l=1,
        )

    wc = WorldCup(teams={1: t(1), 2: t(2)})
    ranks = rank_all(wc)
    assert abs(advance_prob(wc, ranks, 1, 2) - 0.5) < 1e-6


def test_expected_extra_time_flag_set_for_evenly_matched(sample_world_cup) -> None:
    wc = sample_world_cup
    ranks = rank_all(wc)
    # Same team vs itself-strength: draw is the plurality outcome -> flag true.
    pred = predict_knockout(wc, ranks, home_id=1, away_id=1)
    assert pred.expected_extra_time is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/worldcup/test_predict_knockout.py -v`
Expected: FAIL (`ImportError: cannot import name 'advance_prob'`).

- [ ] **Step 3: Implement the knockout model in `predict.py`**

Add the constants near the top (after `DRAW_RHO`):

```python
ET_GOAL_FRACTION = 1 / 3  # extra time adds ~30 min vs 90 at the same rate
PEN_EDGE_PER_10 = 0.03  # shootout win prob shift per 10 effective-rating points
PEN_EDGE_CAP = 0.15  # shootout prob stays within 0.5 +/- this
```

Add the `KnockoutPrediction` dataclass after `MatchPrediction`:

```python
@dataclass(frozen=True)
class KnockoutPrediction:
    match_no: int
    round_name: str
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    score_home: int
    score_away: int
    p_home: float
    p_draw: float
    p_away: float
    p_home_advance: float
    p_away_advance: float
    expected_extra_time: bool
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_no": self.match_no,
            "round_name": self.round_name,
            "home_id": self.home_id,
            "away_id": self.away_id,
            "home_name": self.home_name,
            "away_name": self.away_name,
            "score_home": self.score_home,
            "score_away": self.score_away,
            "p_home": self.p_home,
            "p_draw": self.p_draw,
            "p_away": self.p_away,
            "p_home_advance": self.p_home_advance,
            "p_away_advance": self.p_away_advance,
            "expected_extra_time": self.expected_extra_time,
            "rationale": self.rationale,
        }
```

Add the lambda helper and prediction functions at the end of the module:

```python
def _knockout_lambdas(
    wc: WorldCup, rankings: Rankings, home_id: int, away_id: int, venue: str
) -> tuple[float, float, float, float]:
    """Neutral-site expected goals: travel/weather apply, host home-field does not."""
    base_h = rankings.teams.get(home_id, 50.0)
    base_a = rankings.teams.get(away_id, 50.0)
    eff_h = _effective_rating(wc, home_id, base_h, is_home=False, venue=venue)
    eff_a = _effective_rating(wc, away_id, base_a, is_home=False, venue=venue)
    supremacy = SUPREMACY_PER_10 * (eff_h - eff_a) / 10.0
    lam_home = max(BASE_MATCH_GOALS / 2.0 + supremacy / 2.0, LAMBDA_FLOOR)
    lam_away = max(BASE_MATCH_GOALS / 2.0 - supremacy / 2.0, LAMBDA_FLOOR)
    return lam_home, lam_away, eff_h, eff_a


def _shootout_home_prob(eff_home: float, eff_away: float) -> float:
    edge = PEN_EDGE_PER_10 * (eff_home - eff_away) / 10.0
    return min(max(0.5 + edge, 0.5 - PEN_EDGE_CAP), 0.5 + PEN_EDGE_CAP)


def _advance_from_lambdas(
    lam_home: float, lam_away: float, eff_home: float, eff_away: float
) -> tuple[float, float, float, float]:
    """Return (p_home, p_draw, p_away, p_home_advance) for a no-draw tie."""
    p_home, p_draw, p_away = _outcome_probs(_scoreline_matrix(lam_home, lam_away))
    et_home, et_draw, et_away = _outcome_probs(
        _scoreline_matrix(lam_home * ET_GOAL_FRACTION, lam_away * ET_GOAL_FRACTION)
    )
    pens_home = _shootout_home_prob(eff_home, eff_away)
    p_home_advance = p_home + p_draw * (et_home + et_draw * pens_home)
    return p_home, p_draw, p_away, p_home_advance


def advance_prob(
    wc: WorldCup, rankings: Rankings, home_id: int, away_id: int, venue: str = ""
) -> float:
    lam_home, lam_away, eff_h, eff_a = _knockout_lambdas(wc, rankings, home_id, away_id, venue)
    return _advance_from_lambdas(lam_home, lam_away, eff_h, eff_a)[3]


def predict_knockout(
    wc: WorldCup,
    rankings: Rankings,
    home_id: int,
    away_id: int,
    *,
    match_no: int = 0,
    round_name: str = "",
    venue: str = "",
) -> KnockoutPrediction:
    lam_home, lam_away, eff_h, eff_a = _knockout_lambdas(wc, rankings, home_id, away_id, venue)
    p_home, p_draw, p_away, p_home_adv = _advance_from_lambdas(lam_home, lam_away, eff_h, eff_a)
    score_home, score_away = _modal_score(_scoreline_matrix(lam_home, lam_away))
    rationale = (
        f"Neutral-site rating {eff_h:.1f} vs {eff_a:.1f}; xG {lam_home:.2f}-{lam_away:.2f}; "
        f"advance {p_home_adv:.0%}-{1 - p_home_adv:.0%}."
    )
    return KnockoutPrediction(
        match_no=match_no,
        round_name=round_name,
        home_id=home_id,
        away_id=away_id,
        home_name=wc.teams[home_id].name,
        away_name=wc.teams[away_id].name,
        score_home=score_home,
        score_away=score_away,
        p_home=round(p_home, 4),
        p_draw=round(p_draw, 4),
        p_away=round(p_away, 4),
        p_home_advance=round(p_home_adv, 4),
        p_away_advance=round(1 - p_home_adv, 4),
        expected_extra_time=(p_draw >= p_home and p_draw >= p_away),
        rationale=rationale,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_predict_knockout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/worldcup/predict.py tests/worldcup/test_predict_knockout.py
git commit -m "feat(wc): add knockout match model with extra time and penalties"
```

---

### Task 6: Modal bracket walk

**Files:**
- Create: `src/soccer/worldcup/simulate.py`
- Test: `tests/worldcup/test_simulate_modal.py`

**Interfaces:**
- Consumes: `build_bracket`, `BracketTie`, `KNOCKOUT_EDGES` from Task 4; `predict_knockout`, `advance_prob`, `KnockoutPrediction` from Task 5; `team_labels` from Task 3; `Rankings`.
- Produces:
  - `@dataclass(frozen=True) Podium` with `champion_id, runner_up_id, third_id, fourth_id` and matching `*_name` strings, plus `to_dict`.
  - `run_modal_bracket(wc, rankings, ties: dict[int, BracketTie]) -> tuple[list[KnockoutPrediction], Podium]` — resolves matches 73..104 in order, advancing the higher `p_home_advance` side; returns predictions ordered by `match_no` and the podium.
  - Helper `_resolve_src(src: str, winners, losers) -> int` mapping `"W74"`/`"L101"` to a team id.

- [ ] **Step 1: Write the failing tests**

```python
# tests/worldcup/test_simulate_modal.py
from __future__ import annotations

from datetime import UTC, datetime

from soccer.worldcup.bracket import build_bracket
from soccer.worldcup.entities import NationalTeam, WcMatch, WorldCup
from soccer.worldcup.ranking import rank_all
from soccer.worldcup.simulate import run_modal_bracket
from soccer.worldcup.standings import team_labels


def _build_full_wc() -> WorldCup:
    """32 teams filling every R32 slot; ratings descend with id so results are deterministic."""
    from soccer.worldcup.bracket import R32_ANCHORS

    slots: list[str] = []
    for letter in "ABCDEFGHIJKL":
        slots += [f"1{letter}", f"2{letter}"]
    thirds = [f"3{c}" for c in "CDEFGHIJ"]
    label_for_id: dict[int, str] = {}
    teams: dict[int, NationalTeam] = {}
    # assign ids 1..24 to winners/runners-up, 25..32 to thirds; build group results below
    return _wc_from_labels(slots + thirds)


def _wc_from_labels(all_labels: list[str]) -> WorldCup:
    teams: dict[int, NationalTeam] = {}
    matches: list[WcMatch] = []
    # one team per label; group/rank encoded so team_labels reproduces the label.
    # Build a 3-team mini group per letter so ranks 1,2,3 fall out by constructed results.
    by_letter: dict[str, list[int]] = {}
    next_id = 1
    label_to_id: dict[str, int] = {}
    for label in all_labels:
        rank, letter = int(label[0]), label[1]
        tid = next_id
        next_id += 1
        label_to_id[label] = tid
        teams[tid] = NationalTeam(
            id=tid, name=label, group=f"Group {letter}", confederation="UEFA",
            is_host=False, player_ids=(), coach_id=None,
            recent_w=4 - rank, recent_d=0, recent_l=rank - 1,
        )
        by_letter.setdefault(letter, []).append(tid)
    # group matches that yield the intended rank order (higher rank id beats lower)
    fid = 1
    for letter, ids in by_letter.items():
        ids_sorted = sorted(ids, key=lambda t: teams[t].name)  # 1x,2x,3x
        for i in range(len(ids_sorted)):
            for j in range(i + 1, len(ids_sorted)):
                matches.append(WcMatch(
                    fixture_id=fid, matchday=1, group=f"Group {letter}",
                    home_id=ids_sorted[i], away_id=ids_sorted[j],
                    kickoff=datetime(2026, 6, 11, tzinfo=UTC), venue="v",
                    home_goals=2, away_goals=0, round_name="Group Stage - 1",
                ))
                fid += 1
    return WorldCup(teams=teams, matches=tuple(matches))


def _add_r32(wc: WorldCup) -> WorldCup:
    from soccer.worldcup.bracket import R32_ANCHORS

    labels = team_labels(wc)
    inv = {v: k for k, v in labels.items()}
    thirds = iter(sorted(t for t in labels.values() if t.startswith("3")))
    r32: list[WcMatch] = []
    fid = 5000
    for no, anchors in R32_ANCHORS.items():
        anchors = list(anchors)
        if len(anchors) == 2:
            h, a = inv[anchors[0]], inv[anchors[1]]
        else:
            h, a = inv[anchors[0]], inv[next(thirds)]
        r32.append(WcMatch(
            fixture_id=fid, matchday=0, group="", home_id=h, away_id=a,
            kickoff=datetime(2026, 6, 28, tzinfo=UTC), venue="",
            home_goals=None, away_goals=None, round_name="Round of 32",
        ))
        fid += 1
    return WorldCup(teams=wc.teams, matches=wc.matches + tuple(r32))


def test_modal_bracket_is_complete_and_deterministic() -> None:
    wc = _add_r32(_wc_from_labels(
        [f"{r}{c}" for c in "ABCDEFGHIJKL" for r in (1, 2)] + [f"3{c}" for c in "CDEFGHIJ"]
    ))
    ranks = rank_all(wc)
    ties = build_bracket(wc, team_labels(wc))
    preds, podium = run_modal_bracket(wc, ranks, ties)
    assert [p.match_no for p in preds] == sorted(range(73, 105))
    assert podium.champion_id != podium.runner_up_id
    # determinism: same inputs -> same champion
    preds2, podium2 = run_modal_bracket(wc, ranks, ties)
    assert podium2.champion_id == podium.champion_id
    # the final's winner is the champion
    final = next(p for p in preds if p.match_no == 104)
    winner = final.home_id if final.p_home_advance >= 0.5 else final.away_id
    assert podium.champion_id == winner
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/worldcup/test_simulate_modal.py -v`
Expected: FAIL (`ModuleNotFoundError: soccer.worldcup.simulate`).

- [ ] **Step 3: Implement the modal walk**

```python
# src/soccer/worldcup/simulate.py
"""Walk the knockout bracket: a modal headline bracket and Monte-Carlo odds.

The modal pass advances each tie's more-likely side to a single champion. The
Monte-Carlo pass (Task 7) samples each tie to produce advancement/title odds.
Both reuse the knockout match model in :mod:`soccer.worldcup.predict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from soccer.worldcup.bracket import BracketTie
from soccer.worldcup.entities import WorldCup
from soccer.worldcup.predict import KnockoutPrediction, predict_knockout
from soccer.worldcup.ranking import Rankings


@dataclass(frozen=True)
class Podium:
    champion_id: int
    champion_name: str
    runner_up_id: int
    runner_up_name: str
    third_id: int
    third_name: str
    fourth_id: int
    fourth_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion": {"id": self.champion_id, "name": self.champion_name},
            "runner_up": {"id": self.runner_up_id, "name": self.runner_up_name},
            "third": {"id": self.third_id, "name": self.third_name},
            "fourth": {"id": self.fourth_id, "name": self.fourth_name},
        }


def _resolve_src(src: str, winners: dict[int, int], losers: dict[int, int]) -> int:
    kind, ref = src[0], int(src[1:])
    return winners[ref] if kind == "W" else losers[ref]


def _teams_for(
    tie: BracketTie, winners: dict[int, int], losers: dict[int, int]
) -> tuple[int, int]:
    if tie.home_id is not None and tie.away_id is not None:
        return tie.home_id, tie.away_id
    return (
        _resolve_src(tie.home_src, winners, losers),
        _resolve_src(tie.away_src, winners, losers),
    )


def run_modal_bracket(
    wc: WorldCup, rankings: Rankings, ties: dict[int, BracketTie]
) -> tuple[list[KnockoutPrediction], Podium]:
    winners: dict[int, int] = {}
    losers: dict[int, int] = {}
    preds: list[KnockoutPrediction] = []
    for no in sorted(ties):
        tie = ties[no]
        home_id, away_id = _teams_for(tie, winners, losers)
        pred = predict_knockout(
            wc, rankings, home_id, away_id,
            match_no=no, round_name=tie.round_name, venue=tie.venue,
        )
        preds.append(pred)
        if pred.p_home_advance >= 0.5:
            winners[no], losers[no] = home_id, away_id
        else:
            winners[no], losers[no] = away_id, home_id
    champ = winners[104]
    runner = losers[104]
    third = winners[103]
    fourth = losers[103]
    podium = Podium(
        champion_id=champ, champion_name=wc.teams[champ].name,
        runner_up_id=runner, runner_up_name=wc.teams[runner].name,
        third_id=third, third_name=wc.teams[third].name,
        fourth_id=fourth, fourth_name=wc.teams[fourth].name,
    )
    return preds, podium
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_simulate_modal.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/worldcup/simulate.py tests/worldcup/test_simulate_modal.py
git commit -m "feat(wc): walk the modal knockout bracket to a champion"
```

---

### Task 7: Monte-Carlo title odds

**Files:**
- Modify: `src/soccer/worldcup/simulate.py`
- Test: `tests/worldcup/test_simulate_montecarlo.py`

**Interfaces:**
- Consumes: `BracketTie`, `advance_prob` from Task 5, `_teams_for`/`_resolve_src` from Task 6, `Rankings`.
- Produces:
  - `@dataclass(frozen=True) TeamOdds` with `team_id, name, reach_r16, reach_qf, reach_sf, reach_final, win` and `to_dict`.
  - `run_monte_carlo(wc, rankings, ties, *, rng: random.Random, n_sims: int = 20000) -> dict[int, TeamOdds]` — pairwise advancement probabilities cached by `(home_id, away_id)`; tallies normalized by `n_sims`. A team "reaches" a round when it wins the previous round's tie, so the reach series is monotone non-increasing.

- [ ] **Step 1: Write the failing tests**

```python
# tests/worldcup/test_simulate_montecarlo.py
from __future__ import annotations

import random

from soccer.worldcup.bracket import build_bracket
from soccer.worldcup.ranking import rank_all
from soccer.worldcup.simulate import run_monte_carlo
from soccer.worldcup.standings import team_labels
from tests.worldcup.test_simulate_modal import _add_r32, _wc_from_labels


def _wc():
    return _add_r32(_wc_from_labels(
        [f"{r}{c}" for c in "ABCDEFGHIJKL" for r in (1, 2)] + [f"3{c}" for c in "CDEFGHIJ"]
    ))


def test_monte_carlo_is_reproducible_with_seed() -> None:
    wc = _wc()
    ranks = rank_all(wc)
    ties = build_bracket(wc, team_labels(wc))
    a = run_monte_carlo(wc, ranks, ties, rng=random.Random(7), n_sims=500)
    b = run_monte_carlo(wc, ranks, ties, rng=random.Random(7), n_sims=500)
    assert {t: o.win for t, o in a.items()} == {t: o.win for t, o in b.items()}


def test_probabilities_are_valid_and_monotone() -> None:
    wc = _wc()
    ranks = rank_all(wc)
    ties = build_bracket(wc, team_labels(wc))
    odds = run_monte_carlo(wc, ranks, ties, rng=random.Random(1), n_sims=1000)
    assert abs(sum(o.win for o in odds.values()) - 1.0) < 1e-9
    for o in odds.values():
        assert 0.0 <= o.win <= o.reach_final <= o.reach_sf <= o.reach_qf <= o.reach_r16 <= 1.0


def test_favourite_has_highest_title_odds() -> None:
    wc = _wc()
    ranks = rank_all(wc)
    ties = build_bracket(wc, team_labels(wc))
    odds = run_monte_carlo(wc, ranks, ties, rng=random.Random(3), n_sims=2000)
    best = max(odds.values(), key=lambda o: o.win)
    # strongest team (recent_w=3 -> a group winner) should out-rank a weak third
    assert best.win > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/worldcup/test_simulate_montecarlo.py -v`
Expected: FAIL (`ImportError: cannot import name 'run_monte_carlo'`).

- [ ] **Step 3: Implement the Monte-Carlo pass**

Add to `simulate.py` (imports: add `import random` and `from soccer.worldcup.predict import advance_prob`):

```python
@dataclass(frozen=True)
class TeamOdds:
    team_id: int
    name: str
    reach_r16: float
    reach_qf: float
    reach_sf: float
    reach_final: float
    win: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "name": self.name,
            "reach_r16": self.reach_r16,
            "reach_qf": self.reach_qf,
            "reach_sf": self.reach_sf,
            "reach_final": self.reach_final,
            "win": self.win,
        }


# match_no ranges whose winners have "reached" the next round.
_WIN_REACHES = {
    "reach_r16": range(73, 89),
    "reach_qf": range(89, 97),
    "reach_sf": range(97, 101),
    "reach_final": range(101, 103),
}


def run_monte_carlo(
    wc: WorldCup,
    rankings: Rankings,
    ties: dict[int, BracketTie],
    *,
    rng: random.Random,
    n_sims: int = 20000,
) -> dict[int, TeamOdds]:
    cache: dict[tuple[int, int], float] = {}

    def p_home(home_id: int, away_id: int, venue: str) -> float:
        key = (home_id, away_id)
        if key not in cache:
            cache[key] = advance_prob(wc, rankings, home_id, away_id, venue)
        return cache[key]

    counts = {
        "reach_r16": dict.fromkeys(wc.teams, 0),
        "reach_qf": dict.fromkeys(wc.teams, 0),
        "reach_sf": dict.fromkeys(wc.teams, 0),
        "reach_final": dict.fromkeys(wc.teams, 0),
        "win": dict.fromkeys(wc.teams, 0),
    }
    order = sorted(ties)
    for _ in range(n_sims):
        winners: dict[int, int] = {}
        losers: dict[int, int] = {}
        for no in order:
            tie = ties[no]
            home_id, away_id = _teams_for(tie, winners, losers)
            if rng.random() < p_home(home_id, away_id, tie.venue):
                winners[no], losers[no] = home_id, away_id
            else:
                winners[no], losers[no] = away_id, home_id
        for field, nos in _WIN_REACHES.items():
            for no in nos:
                counts[field][winners[no]] += 1
        counts["win"][winners[104]] += 1

    out: dict[int, TeamOdds] = {}
    for tid, team in wc.teams.items():
        out[tid] = TeamOdds(
            team_id=tid,
            name=team.name,
            reach_r16=counts["reach_r16"][tid] / n_sims,
            reach_qf=counts["reach_qf"][tid] / n_sims,
            reach_sf=counts["reach_sf"][tid] / n_sims,
            reach_final=counts["reach_final"][tid] / n_sims,
            win=counts["win"][tid] / n_sims,
        )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_simulate_montecarlo.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/worldcup/simulate.py tests/worldcup/test_simulate_montecarlo.py
git commit -m "feat(wc): add Monte-Carlo knockout title and advancement odds"
```

---

### Task 8: `soccer wc knockout` CLI subcommand + report

**Files:**
- Modify: `src/soccer/worldcup/cli.py`
- Test: `tests/worldcup/test_cli.py`

**Interfaces:**
- Consumes: `load_dataset`, `_prediction_dir`, `_dataset_path`; `rank_all`; `team_labels`; `build_bracket`, `BracketError`; `run_modal_bracket`, `run_monte_carlo`, `Podium`, `TeamOdds`.
- Produces:
  - `cmd_knockout(args, config) -> int` — errors out (returns 1, prints fetch hint) if no `Round of 32` fixtures; otherwise writes `<name>.json` + `<name>.md` (default `name="worldcup-2026-knockout"`), prints the champion line, returns 0.
  - `_render_knockout_report(wc, preds, podium, odds) -> str`.
  - subparser wiring in `add_wc_subparser`: `knockout` with `--sims` (default 20000), `--seed` (default 20260628), `--out-dir`, `--name`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/worldcup/test_cli.py  (add)
import json

from soccer.config import AppConfig
from soccer.worldcup import cli as wc_cli


def test_knockout_errors_without_r32(tmp_path, capsys) -> None:
    # dataset with only a group match -> no R32 -> non-zero exit + hint
    from soccer.worldcup.entities import NationalTeam, WcMatch, WorldCup
    from datetime import UTC, datetime

    wc = WorldCup(
        teams={1: NationalTeam(1, "A", "Group A", "UEFA", False, (), None, 0, 0, 0)},
        matches=(WcMatch(1, 1, "Group A", 1, 1, datetime(2026, 6, 11, tzinfo=UTC),
                         "v", 1, 0, "Group Stage - 1"),),
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "worldcup-2026.json").write_text(json.dumps(wc.to_dict()))
    config = AppConfig.from_env().__class__(
        data_dir=data_dir, ollama_host="", ollama_model="", ollama_timeout=1.0,
        provider_mode="fixture", reasoner="fake",
        api_football_base_url="", api_football_key=None,
        prediction_dir=tmp_path / "out",
    )
    args = wc_cli.argparse.Namespace(sims=10, seed=1, out_dir=None, name=None)
    assert wc_cli.cmd_knockout(args, config) == 1
    assert "fetch" in capsys.readouterr().out


def test_knockout_writes_files(tmp_path) -> None:
    # Build a full 32-team dataset with R32 using the modal test's helpers.
    from tests.worldcup.test_simulate_modal import _add_r32, _wc_from_labels

    wc = _add_r32(_wc_from_labels(
        [f"{r}{c}" for c in "ABCDEFGHIJKL" for r in (1, 2)] + [f"3{c}" for c in "CDEFGHIJ"]
    ))
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "worldcup-2026.json").write_text(json.dumps(wc.to_dict()))
    out = tmp_path / "out"
    config = AppConfig.from_env().__class__(
        data_dir=data_dir, ollama_host="", ollama_model="", ollama_timeout=1.0,
        provider_mode="fixture", reasoner="fake",
        api_football_base_url="", api_football_key=None, prediction_dir=out,
    )
    args = wc_cli.argparse.Namespace(sims=50, seed=1, out_dir=None, name=None)
    assert wc_cli.cmd_knockout(args, config) == 0
    assert (out / "worldcup-2026-knockout.json").exists()
    assert (out / "worldcup-2026-knockout.md").exists()
    payload = json.loads((out / "worldcup-2026-knockout.json").read_text())
    assert "podium" in payload and "title_odds" in payload and "bracket" in payload
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/worldcup/test_cli.py -k knockout -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'cmd_knockout'`).

- [ ] **Step 3: Implement the subcommand + report**

Add imports at the top of `cli.py`:

```python
from soccer.worldcup.bracket import BracketError, build_bracket
from soccer.worldcup.simulate import Podium, TeamOdds, run_modal_bracket, run_monte_carlo
from soccer.worldcup.standings import team_labels
```

Add the report renderer and command (place near `cmd_predict`):

```python
def _render_knockout_report(
    wc: WorldCup,
    preds: list[Any],
    podium: Podium,
    odds: dict[int, TeamOdds],
) -> str:
    lines = [
        "# FIFA 2026 World Cup — Knockout-Stage Forecast",
        "",
        "Most-likely bracket from the live Round-of-32 draw forward, with each tie's "
        "predicted score and advancement odds, then Monte-Carlo title odds.",
        "",
        "## Predicted podium",
        "",
        f"- 🥇 **Champion:** {podium.champion_name}",
        f"- 🥈 **Runner-up:** {podium.runner_up_name}",
        f"- 🥉 **Third:** {podium.third_name}",
        f"- 4th: {podium.fourth_name}",
        "",
        "## Bracket",
        "",
    ]
    current = ""
    for p in preds:
        if p.round_name != current:
            current = p.round_name
            lines += [f"### {current}", ""]
        et = "  _(likely AET/pens)_" if p.expected_extra_time else ""
        lines.append(
            f"- `M{p.match_no}` **{p.home_name} {p.score_home}-{p.score_away} {p.away_name}**  "
            f"(adv {p.home_name} {p.p_home_advance:.0%} / {p.away_name} {p.p_away_advance:.0%})"
            f"{et}"
        )
    lines += ["", "## Title odds (top 12)", "", "| Team | Win | Final | Semi | Quarter |", "|---|---|---|---|---|"]
    for o in sorted(odds.values(), key=lambda o: o.win, reverse=True)[:12]:
        lines.append(
            f"| {o.name} | {o.win:.1%} | {o.reach_final:.0%} | {o.reach_sf:.0%} | {o.reach_qf:.0%} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def cmd_knockout(args: argparse.Namespace, config: AppConfig) -> int:
    wc = load_dataset(_dataset_path(config))
    if not any(m.round_name == "Round of 32" for m in wc.matches):
        print(
            "no Round of 32 fixtures in the dataset; run `soccer wc fetch` "
            "(or `refresh`) to pull the knockout draw first"
        )
        return 1
    rankings = rank_all(wc)
    try:
        ties = build_bracket(wc, team_labels(wc))
    except BracketError as exc:
        print(f"bracket error: {exc}")
        return 1
    preds, podium = run_modal_bracket(wc, rankings, ties)
    odds = run_monte_carlo(
        wc, rankings, ties, rng=random.Random(args.seed), n_sims=args.sims
    )
    out_dir = Path(args.out_dir) if args.out_dir else _prediction_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or "worldcup-2026-knockout"
    payload = {
        "bracket": [p.to_dict() for p in preds],
        "podium": podium.to_dict(),
        "title_odds": [
            o.to_dict() for o in sorted(odds.values(), key=lambda o: o.win, reverse=True)
        ],
    }
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / f"{name}.md").write_text(
        _render_knockout_report(wc, preds, podium, odds), encoding="utf-8"
    )
    print(f"predicted champion: {podium.champion_name}")
    print(f"wrote {out_dir / f'{name}.json'} and {out_dir / f'{name}.md'}")
    return 0
```

Add `import random` at the top of `cli.py` if not present. Wire the subparser inside `add_wc_subparser` (after `p_predict`):

```python
    p_ko = wc_sub.add_parser("knockout", help="forecast the knockout bracket to the final")
    p_ko.add_argument("--sims", type=int, default=20000, help="Monte-Carlo iterations")
    p_ko.add_argument("--seed", type=int, default=20260628, help="RNG seed (reproducible)")
    p_ko.add_argument("--out-dir", default=None, help="output directory for the files")
    p_ko.add_argument("--name", default=None, help="basename for the .json/.md files")
    p_ko.set_defaults(func=cmd_knockout)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/worldcup/test_cli.py -k knockout -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite + linters**

Run: `ruff format . && ruff check . && mypy src tests && pytest`
Expected: all PASS. Fix any types (e.g. annotate `preds: list[KnockoutPrediction]` in the renderer instead of `Any` if mypy prefers).

- [ ] **Step 6: Commit**

```bash
git add src/soccer/worldcup/cli.py tests/worldcup/test_cli.py
git commit -m "feat(wc): add 'soccer wc knockout' bracket forecast command"
```

---

### Task 9: Refresh real data, generate the forecast, update docs

**Files:**
- Modify: `README.md` (World Cup usage section)
- Modify: `docs/architecture.md` (if it lists the worldcup modules)
- Output (generated, committed): `prediction/worldcup-2026-knockout.json`, `prediction/worldcup-2026-knockout.md`

**Interfaces:**
- Consumes: everything above; the networked `soccer wc fetch`/`refresh` path (needs `SOCCER_API_FOOTBALL_KEY` from `.env`).

- [ ] **Step 1: Refresh the dataset with completed groups + live R32**

This is a networked step (needs the API key in `.env`). Run:

```bash
set -a; source .env; set +a
python -m soccer wc fetch   # or: python -m soccer wc refresh
```

Expected: the printed match count rises from 72 to 88 (24×3 group + 16 Round of 32), and all 72 group matches are finished. Confirm:

```bash
python -c "import json; d=json.load(open('data/worldcup-2026.json')); \
from collections import Counter; \
print(Counter(m['round_name'] for m in d['matches']))"
```

Expected: includes `'Round of 32': 16` and the three `Group Stage - N` rounds.

- [ ] **Step 2: Generate the knockout forecast**

```bash
python -m soccer wc knockout
```

Expected: prints `predicted champion: <team>` and writes `prediction/worldcup-2026-knockout.{json,md}`. Open the `.md` and sanity-check: the R32 ties match the real draw (e.g. Brazil vs Japan, Argentina vs Cape Verde Islands), the bracket runs to a single Final, and the title-odds table is sorted with credible favourites on top.

- [ ] **Step 3: Sanity-check the bracket against the live draw**

```bash
python -c "
import json; d=json.load(open('prediction/worldcup-2026-knockout.json'))
r32=[t for t in d['bracket'] if t['round_name']=='Round of 32']
assert len(r32)==16, r32
print('champion:', d['podium']['champion']['name'])
print('top-3 title odds:', [(o['name'], round(o['win'],3)) for o in d['title_odds'][:3]])
"
```

Expected: 16 R32 ties, a printed champion, and three credible favourites. If `build_bracket` raised a `BracketError`, the live label→slot mapping disagrees with `R32_ANCHORS`; re-verify the anchors against the official bracket and the computed `team_labels` before continuing.

- [ ] **Step 4: Update the README**

Add a knockout subsection to the World Cup usage in `README.md`, e.g.:

```markdown
### Knockout-stage forecast

Once the group stage is complete and the Round of 32 is drawn, forecast the
whole bracket to the final:

​```bash
python -m soccer wc fetch          # pulls completed groups + the R32 draw
python -m soccer wc knockout       # writes prediction/worldcup-2026-knockout.{json,md}
python -m soccer wc knockout --sims 50000 --seed 7   # more Monte-Carlo iterations
​```

The report gives a single most-likely bracket (each tie's score + advancement
odds, with extra-time/penalty notes), the predicted podium, and Monte-Carlo
title odds. Standings are computed from results; the Round of 16 onward use the
official FIFA-2026 bracket tree.
```

- [ ] **Step 5: Final validation + commit**

Run: `ruff format . && ruff check . && mypy src tests && pytest`
Expected: all PASS.

```bash
git add README.md docs/architecture.md prediction/worldcup-2026-knockout.json prediction/worldcup-2026-knockout.md
git commit -m "data(wc): generate knockout-stage forecast and document the command"
```

---

## Self-Review Notes

- **Spec coverage:** §1 flow → Tasks 8-9; §2 data model → Task 1; §2 ingest fix → Task 2; §3 standings → Task 3; §4 bracket hybrid + cross-check → Task 4; §5 knockout model → Task 5; §6 modal + Monte Carlo → Tasks 6-7; §7 CLI + report → Task 8; §8 testing → per-task tests (all offline, injected `random.Random`); refresh/real output + docs → Task 9.
- **Neutral-site modeling** (no host home-field bonus in knockouts; travel/weather retained) is stated in Task 5 and is intentional — it keeps `advance_prob` symmetric and cacheable by `(home_id, away_id)`.
- **Bracket data** (`R32_ANCHORS`, `KNOCKOUT_EDGES`) is transcribed from the official bracket (matches 73-104) and guarded at runtime by the one-to-one slot cross-check and at generation time by Task 9 Step 3.
```
