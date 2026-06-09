# Soccer Prediction Agent — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-tool soccer match-prediction agent that assembles a typed dossier from pluggable data tools, reasons with a local Ollama model (behind a swappable interface), logs a 1X2 prediction with rationale and confidence, settles results on demand, self-evaluates, and scores itself against the bookmaker via an offline eval harness.

**Architecture:** Deterministic pipeline (`PredictionAgent`) over a `ToolRegistry` of single-method `Protocol` providers (fixture + HTTP adapters), assembling a `MatchDossier`, then one call to a `Reasoner` (Ollama or deterministic fake). Append-only JSONL persistence. On-demand `settle` and an offline `harness` using fixture scenarios with known results.

**Tech Stack:** Python 3.11+, standard library only at runtime (HTTP via `urllib` behind an injected callable). Dev: ruff, mypy, pytest, pytest-cov, pre-commit. `src/` layout, package `soccer`.

---

## File Structure

```text
src/soccer/
  __init__.py
  models.py            # enums, dataclasses, prob helpers, serialization
  config.py            # AppConfig (env reads at boundary)
  tools/
    __init__.py
    base.py            # ToolError, Tool view
    fixtures.py        # JSON fixture loader
    form.py injuries.py head_to_head.py weather.py venue.py odds.py results.py
  registry.py          # ToolRegistry
  dossier.py           # build_dossier
  reasoning/
    __init__.py
    base.py            # Reasoner Protocol, ReasonResult, ReasonerError
    fake.py            # DeterministicReasoner
    prompt.py          # dossier → prompt + JSON parsing helpers
    ollama.py          # OllamaReasoner (injected post_json)
  agent.py             # PredictionAgent
  store.py             # PredictionStore (JSONL)
  evaluation.py        # score(), metrics, calibration
  settle.py            # settle()
  harness.py           # Scenario, EvalReport, run_scenario
  scenarios.py         # build UCL + WC-final fixture scenarios
  cli.py               # argparse subcommands
  __main__.py
tests/                 # mirrors src/soccer
fixtures/              # JSON fixture data for scenarios
```

---

### Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `Makefile`, `.gitignore`, `.env.example`, `src/soccer/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Initialise git**

```bash
cd /Users/houyang/gitHub/ai-coding/soccer-forcast/opus4.8-cloud-claude
git init
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "soccer"
version = "0.1.0"
description = "Multi-tool soccer match-prediction agent"
requires-python = ">=3.11"
dependencies = []

[project.optional-dependencies]
dev = ["ruff>=0.5", "mypy>=1.10", "pytest>=8", "pytest-cov>=5", "pre-commit>=3"]

[project.scripts]
soccer = "soccer.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
src = ["src", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.mypy]
python_version = "3.11"
strict = true
files = ["src", "tests"]

[tool.pytest.ini_options]
addopts = "-q"
testpaths = ["tests"]

[tool.coverage.run]
source = ["soccer"]
branch = true
```

- [ ] **Step 3: Write `Makefile`**

```makefile
.PHONY: format lint typecheck test coverage check
format:
	ruff format .
lint:
	ruff check .
typecheck:
	mypy src tests
test:
	pytest
coverage:
	pytest --cov=soccer --cov-report=term-missing
check: lint typecheck coverage
```

- [ ] **Step 4: Write `.gitignore`**

```gitignore
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
data/
```

- [ ] **Step 5: Write `.env.example`**

```bash
# Data directory for JSONL prediction/result/evaluation logs
SOCCER_DATA_DIR=./data
# Ollama reasoner
SOCCER_OLLAMA_HOST=http://localhost:11434
SOCCER_OLLAMA_MODEL=gemma4:12b-mlx
SOCCER_OLLAMA_TIMEOUT=60
# Provider mode: fixture | http
SOCCER_PROVIDER_MODE=fixture
# Reasoner: fake | ollama
SOCCER_REASONER=fake
```

- [ ] **Step 6: Create empty package files**

Create `src/soccer/__init__.py` containing `__version__ = "0.1.0"`. Create empty `tests/__init__.py` and empty `tests/conftest.py`.

- [ ] **Step 7: Install and verify tooling**

Run:
```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
ruff check . && mypy src tests && pytest
```
Expected: ruff/mypy pass; pytest reports "no tests ran" (exit 5 is acceptable here).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: scaffold soccer prediction agent project"
```

---

### Task 1: Core enums and probability helpers (`models.py` part 1)

**Files:**
- Create: `src/soccer/models.py`
- Test: `tests/test_models_probs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_probs.py
import pytest
from soccer.models import Outcome, normalize_probs, validate_probs

def test_normalize_scales_to_one():
    out = normalize_probs({Outcome.HOME: 2.0, Outcome.DRAW: 1.0, Outcome.AWAY: 1.0})
    assert out[Outcome.HOME] == pytest.approx(0.5)
    assert sum(out.values()) == pytest.approx(1.0)

def test_normalize_rejects_nonpositive_total():
    with pytest.raises(ValueError):
        normalize_probs({Outcome.HOME: 0.0, Outcome.DRAW: 0.0, Outcome.AWAY: 0.0})

def test_validate_requires_all_three_outcomes():
    with pytest.raises(ValueError):
        validate_probs({Outcome.HOME: 0.5, Outcome.DRAW: 0.5})

def test_validate_requires_sum_one():
    with pytest.raises(ValueError):
        validate_probs({Outcome.HOME: 0.5, Outcome.DRAW: 0.4, Outcome.AWAY: 0.4})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_probs.py -v`
Expected: FAIL with ImportError (cannot import name from soccer.models).

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/models.py
from __future__ import annotations

from enum import Enum


class Outcome(str, Enum):
    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"


class MatchOutcome(str, Enum):
    WIN = "W"
    DRAW = "D"
    LOSS = "L"


def normalize_probs(probs: dict[Outcome, float]) -> dict[Outcome, float]:
    total = sum(probs.values())
    if total <= 0:
        raise ValueError("probability total must be positive")
    return {k: v / total for k, v in probs.items()}


def validate_probs(probs: dict[Outcome, float]) -> dict[Outcome, float]:
    if set(probs) != set(Outcome):
        raise ValueError(f"probs must cover all outcomes, got {set(probs)}")
    if any(v < 0 for v in probs.values()):
        raise ValueError("probabilities must be non-negative")
    if abs(sum(probs.values()) - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1.0, got {sum(probs.values())}")
    return probs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models_probs.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/models.py tests/test_models_probs.py
git commit -m "feat: add Outcome enum and probability helpers"
```

---

### Task 2: Dossier dataclasses (`models.py` part 2)

**Files:**
- Modify: `src/soccer/models.py`
- Test: `tests/test_models_dossier.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_dossier.py
from datetime import datetime, timezone
from soccer.models import (
    MatchRef, TeamForm, MatchOutcome, OddsSnapshot, Outcome, MatchDossier,
)

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)

def test_odds_implied_probs_normalised():
    odds = OddsSnapshot(bookmaker="b", home=2.0, draw=4.0, away=4.0,
                        as_of=KICK, source="fixture")
    p = odds.implied_probs
    assert sum(p.values()) == 0.0 + 1.0  # normalised to 1
    assert p[Outcome.HOME] > p[Outcome.DRAW]

def test_dossier_holds_optional_pieces():
    ref = MatchRef(id="m1", competition="UCL", home="A", away="B",
                   kickoff=KICK, venue_id="v1", season="2025-26")
    form = TeamForm(team="A", last_n=(MatchOutcome.WIN,), gf=3, ga=1, points=3,
                    streak="W1", as_of=KICK, source="fixture")
    d = MatchDossier(match=ref, form={"home": form, "away": None},
                     injuries={"home": None, "away": None}, h2h=None,
                     weather=None, venue=None, odds=None, missing=("odds",))
    assert d.form["home"].team == "A"
    assert "odds" in d.missing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_dossier.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

Append to `src/soccer/models.py`:

```python
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MatchRef:
    id: str
    competition: str
    home: str
    away: str
    kickoff: datetime
    venue_id: str
    season: str


@dataclass(frozen=True)
class PlayerStatus:
    name: str
    status: str  # "out" | "doubtful"
    reason: str


@dataclass(frozen=True)
class TeamForm:
    team: str
    last_n: tuple[MatchOutcome, ...]
    gf: int
    ga: int
    points: int
    streak: str
    as_of: datetime
    source: str


@dataclass(frozen=True)
class InjuryReport:
    team: str
    out: tuple[PlayerStatus, ...]
    doubtful: tuple[PlayerStatus, ...]
    as_of: datetime
    source: str


@dataclass(frozen=True)
class PastMeeting:
    date: datetime
    home: str
    away: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True)
class H2HRecord:
    home: str
    away: str
    meetings: tuple[PastMeeting, ...]
    home_wins: int
    draws: int
    away_wins: int
    source: str


@dataclass(frozen=True)
class WeatherReport:
    venue_id: str
    kickoff: datetime
    temp_c: float
    wind_kph: float
    precip_mm: float
    condition: str
    source: str


@dataclass(frozen=True)
class VenueInfo:
    venue_id: str
    name: str
    city: str
    surface: str
    capacity: int
    altitude_m: int
    home_advantage_hint: float
    source: str


@dataclass(frozen=True)
class OddsSnapshot:
    bookmaker: str
    home: float
    draw: float
    away: float
    as_of: datetime
    source: str

    @property
    def implied_probs(self) -> dict[Outcome, float]:
        raw = {
            Outcome.HOME: 1.0 / self.home,
            Outcome.DRAW: 1.0 / self.draw,
            Outcome.AWAY: 1.0 / self.away,
        }
        return normalize_probs(raw)


@dataclass(frozen=True)
class MatchDossier:
    match: MatchRef
    form: dict[str, TeamForm | None]
    injuries: dict[str, InjuryReport | None]
    h2h: H2HRecord | None
    weather: WeatherReport | None
    venue: VenueInfo | None
    odds: OddsSnapshot | None
    missing: tuple[str, ...]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models_dossier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/models.py tests/test_models_dossier.py
git commit -m "feat: add dossier dataclasses"
```

---

### Task 3: Prediction / result / evaluation models + serialization (`models.py` part 3)

**Files:**
- Modify: `src/soccer/models.py`
- Test: `tests/test_models_lifecycle.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models_lifecycle.py
from datetime import datetime, timezone
from soccer.models import (
    MatchRef, Outcome, Prediction, MatchResult, Evaluation,
    prediction_to_dict, prediction_from_dict,
    result_to_dict, result_from_dict,
    evaluation_to_dict, evaluation_from_dict,
)

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)
REF = MatchRef(id="m1", competition="UCL", home="A", away="B",
               kickoff=KICK, venue_id="v1", season="2025-26")

def _pred() -> Prediction:
    return Prediction(
        id="abc123", match_ref=REF, created_at=KICK,
        probs={Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2},
        pick=Outcome.HOME, confidence=0.5, rationale="strong home form",
        market_probs={Outcome.HOME: 0.45, Outcome.DRAW: 0.3, Outcome.AWAY: 0.25},
        dossier_digest="deadbeef", reasoner_name="fake")

def test_prediction_round_trip():
    p = _pred()
    assert prediction_from_dict(prediction_to_dict(p)) == p

def test_result_outcome_property():
    r = MatchResult(match_id="m1", home_goals=2, away_goals=1,
                    status="finished", source="fixture")
    assert r.outcome is Outcome.HOME

def test_result_round_trip():
    r = MatchResult(match_id="m1", home_goals=1, away_goals=1,
                    status="finished", source="fixture")
    assert result_from_dict(result_to_dict(r)) == r
    assert r.outcome is Outcome.DRAW

def test_evaluation_round_trip():
    r = MatchResult(match_id="m1", home_goals=0, away_goals=2,
                    status="finished", source="fixture")
    e = Evaluation(prediction_id="abc123", result=r, correct=False,
                   brier=0.5, log_loss=1.2, beat_market=False,
                   self_critique="overrated home", evaluated_at=KICK)
    assert evaluation_from_dict(evaluation_to_dict(e)) == e
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models_lifecycle.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

Append to `src/soccer/models.py`:

```python
import hashlib
from typing import Any


@dataclass(frozen=True)
class Prediction:
    id: str
    match_ref: MatchRef
    created_at: datetime
    probs: dict[Outcome, float]
    pick: Outcome
    confidence: float
    rationale: str
    market_probs: dict[Outcome, float] | None
    dossier_digest: str
    reasoner_name: str

    def __post_init__(self) -> None:
        validate_probs(self.probs)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class MatchResult:
    match_id: str
    home_goals: int
    away_goals: int
    status: str
    source: str

    @property
    def outcome(self) -> Outcome:
        if self.home_goals > self.away_goals:
            return Outcome.HOME
        if self.home_goals < self.away_goals:
            return Outcome.AWAY
        return Outcome.DRAW


@dataclass(frozen=True)
class Evaluation:
    prediction_id: str
    result: MatchResult
    correct: bool
    brier: float
    log_loss: float
    beat_market: bool
    self_critique: str
    evaluated_at: datetime


def make_prediction_id(match_id: str, created_at: datetime) -> str:
    raw = f"{match_id}:{created_at.isoformat()}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _probs_to_dict(probs: dict[Outcome, float] | None) -> dict[str, float] | None:
    if probs is None:
        return None
    return {k.value: v for k, v in probs.items()}


def _probs_from_dict(raw: dict[str, float] | None) -> dict[Outcome, float] | None:
    if raw is None:
        return None
    return {Outcome(k): v for k, v in raw.items()}


def _ref_to_dict(ref: MatchRef) -> dict[str, Any]:
    return {
        "id": ref.id, "competition": ref.competition, "home": ref.home,
        "away": ref.away, "kickoff": ref.kickoff.isoformat(),
        "venue_id": ref.venue_id, "season": ref.season,
    }


def _ref_from_dict(raw: dict[str, Any]) -> MatchRef:
    return MatchRef(
        id=raw["id"], competition=raw["competition"], home=raw["home"],
        away=raw["away"], kickoff=datetime.fromisoformat(raw["kickoff"]),
        venue_id=raw["venue_id"], season=raw["season"],
    )


def prediction_to_dict(p: Prediction) -> dict[str, Any]:
    return {
        "id": p.id, "match_ref": _ref_to_dict(p.match_ref),
        "created_at": p.created_at.isoformat(), "probs": _probs_to_dict(p.probs),
        "pick": p.pick.value, "confidence": p.confidence, "rationale": p.rationale,
        "market_probs": _probs_to_dict(p.market_probs),
        "dossier_digest": p.dossier_digest, "reasoner_name": p.reasoner_name,
    }


def prediction_from_dict(raw: dict[str, Any]) -> Prediction:
    return Prediction(
        id=raw["id"], match_ref=_ref_from_dict(raw["match_ref"]),
        created_at=datetime.fromisoformat(raw["created_at"]),
        probs=_probs_from_dict(raw["probs"]),  # type: ignore[arg-type]
        pick=Outcome(raw["pick"]), confidence=raw["confidence"],
        rationale=raw["rationale"], market_probs=_probs_from_dict(raw["market_probs"]),
        dossier_digest=raw["dossier_digest"], reasoner_name=raw["reasoner_name"],
    )


def result_to_dict(r: MatchResult) -> dict[str, Any]:
    return {
        "match_id": r.match_id, "home_goals": r.home_goals,
        "away_goals": r.away_goals, "status": r.status, "source": r.source,
    }


def result_from_dict(raw: dict[str, Any]) -> MatchResult:
    return MatchResult(
        match_id=raw["match_id"], home_goals=raw["home_goals"],
        away_goals=raw["away_goals"], status=raw["status"], source=raw["source"],
    )


def evaluation_to_dict(e: Evaluation) -> dict[str, Any]:
    return {
        "prediction_id": e.prediction_id, "result": result_to_dict(e.result),
        "correct": e.correct, "brier": e.brier, "log_loss": e.log_loss,
        "beat_market": e.beat_market, "self_critique": e.self_critique,
        "evaluated_at": e.evaluated_at.isoformat(),
    }


def evaluation_from_dict(raw: dict[str, Any]) -> Evaluation:
    return Evaluation(
        prediction_id=raw["prediction_id"], result=result_from_dict(raw["result"]),
        correct=raw["correct"], brier=raw["brier"], log_loss=raw["log_loss"],
        beat_market=raw["beat_market"], self_critique=raw["self_critique"],
        evaluated_at=datetime.fromisoformat(raw["evaluated_at"]),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_models_lifecycle.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/models.py tests/test_models_lifecycle.py
git commit -m "feat: add prediction/result/evaluation models and serialization"
```

---

### Task 4: Tool base + fixture loader (`tools/base.py`, `tools/fixtures.py`)

**Files:**
- Create: `src/soccer/tools/__init__.py`, `src/soccer/tools/base.py`, `src/soccer/tools/fixtures.py`
- Test: `tests/tools/__init__.py`, `tests/tools/test_fixtures.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_fixtures.py
import json
import pytest
from soccer.tools.base import ToolError
from soccer.tools.fixtures import FixtureStore

def test_fixture_store_reads_section(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"form": {"A": {"team": "A"}}}))
    store = FixtureStore(path)
    assert store.get("form", "A") == {"team": "A"}

def test_fixture_store_missing_key_raises_toolerror(tmp_path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"form": {}}))
    store = FixtureStore(path)
    with pytest.raises(ToolError):
        store.get("form", "Z")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_fixtures.py -v`
Expected: FAIL with ImportError. (Create empty `tests/tools/__init__.py` first.)

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/tools/__init__.py
# (empty marker)
```

```python
# src/soccer/tools/base.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ToolError(Exception):
    """Raised by a provider when data cannot be obtained."""


@dataclass(frozen=True)
class Tool:
    """Uniform view of a capability for a future model-driven selection loop."""

    name: str
    description: str
    call: Callable[..., Any]
```

```python
# src/soccer/tools/fixtures.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from soccer.tools.base import ToolError


class FixtureStore:
    """Loads a single JSON file of the form {section: {key: payload}}."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        try:
            self._data: dict[str, Any] = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolError(f"cannot load fixture {self._path}: {exc}") from exc

    def get(self, section: str, key: str) -> Any:
        try:
            return self._data[section][key]
        except KeyError as exc:
            raise ToolError(f"fixture missing {section}/{key}") from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_fixtures.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/tools/ tests/tools/
git commit -m "feat: add tool base types and fixture store"
```

---

### Task 5: Provider protocols + fixture providers (`tools/*.py`)

**Files:**
- Create: `src/soccer/tools/form.py`, `injuries.py`, `head_to_head.py`, `weather.py`, `venue.py`, `odds.py`, `results.py`
- Test: `tests/tools/test_providers.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_providers.py
import json
from datetime import datetime, timezone
import pytest
from soccer.models import MatchRef, Outcome
from soccer.tools.base import ToolError
from soccer.tools.fixtures import FixtureStore
from soccer.tools.form import FixtureFormProvider
from soccer.tools.injuries import FixtureInjuryProvider
from soccer.tools.head_to_head import FixtureH2HProvider
from soccer.tools.weather import FixtureWeatherProvider
from soccer.tools.venue import FixtureVenueProvider
from soccer.tools.odds import FixtureOddsProvider
from soccer.tools.results import FixtureResultProvider

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)
REF = MatchRef(id="m1", competition="UCL", home="A", away="B",
               kickoff=KICK, venue_id="v1", season="2025-26")

@pytest.fixture
def store(tmp_path):
    payload = {
        "form": {"A": {"team": "A", "last_n": ["W", "W", "D"], "gf": 7, "ga": 2,
                       "points": 21, "streak": "W2"}},
        "injuries": {"A": {"team": "A",
                           "out": [{"name": "P1", "status": "out", "reason": "knee"}],
                           "doubtful": []}},
        "h2h": {"A|B": {"home": "A", "away": "B",
                        "meetings": [{"date": "2025-01-01T00:00:00+00:00",
                                      "home": "A", "away": "B",
                                      "home_goals": 2, "away_goals": 1}],
                        "home_wins": 1, "draws": 0, "away_wins": 0}},
        "weather": {"v1": {"venue_id": "v1", "temp_c": 12.0, "wind_kph": 9.0,
                           "precip_mm": 0.0, "condition": "clear"}},
        "venue": {"v1": {"venue_id": "v1", "name": "Stad", "city": "X",
                         "surface": "grass", "capacity": 60000, "altitude_m": 50,
                         "home_advantage_hint": 0.1}},
        "odds": {"m1": {"bookmaker": "b", "home": 2.0, "draw": 3.5, "away": 3.8}},
        "results": {"m1": {"home_goals": 2, "away_goals": 1, "status": "finished"}},
    }
    path = tmp_path / "ucl.json"
    path.write_text(json.dumps(payload))
    return FixtureStore(path)

def test_form_provider(store):
    form = FixtureFormProvider(store).get_form("A", KICK)
    assert form.points == 21 and form.last_n[0].value == "W"

def test_injury_provider(store):
    rep = FixtureInjuryProvider(store).get_injuries("A", KICK)
    assert rep.out[0].name == "P1"

def test_h2h_provider(store):
    rec = FixtureH2HProvider(store).get_h2h("A", "B")
    assert rec.home_wins == 1 and rec.meetings[0].home_goals == 2

def test_weather_provider(store):
    w = FixtureWeatherProvider(store).get_weather("v1", KICK)
    assert w.condition == "clear"

def test_venue_provider(store):
    v = FixtureVenueProvider(store).get_venue("v1")
    assert v.capacity == 60000

def test_odds_provider(store):
    o = FixtureOddsProvider(store).get_odds(REF)
    assert o.implied_probs[Outcome.HOME] > o.implied_probs[Outcome.AWAY]

def test_result_provider_returns_none_when_absent(store):
    none_ref = MatchRef(id="zzz", competition="UCL", home="A", away="B",
                        kickoff=KICK, venue_id="v1", season="2025-26")
    assert FixtureResultProvider(store).get_result(none_ref) is None

def test_result_provider_returns_result(store):
    r = FixtureResultProvider(store).get_result(REF)
    assert r is not None and r.outcome is Outcome.HOME
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_providers.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/tools/form.py
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from soccer.models import MatchOutcome, TeamForm
from soccer.tools.fixtures import FixtureStore


class FormProvider(Protocol):
    def get_form(self, team: str, as_of: datetime) -> TeamForm: ...


class FixtureFormProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_form(self, team: str, as_of: datetime) -> TeamForm:
        raw = self._store.get("form", team)
        return TeamForm(
            team=raw["team"],
            last_n=tuple(MatchOutcome(x) for x in raw["last_n"]),
            gf=raw["gf"], ga=raw["ga"], points=raw["points"],
            streak=raw["streak"], as_of=as_of, source="fixture",
        )
```

```python
# src/soccer/tools/injuries.py
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from soccer.models import InjuryReport, PlayerStatus
from soccer.tools.fixtures import FixtureStore


class InjuryProvider(Protocol):
    def get_injuries(self, team: str, as_of: datetime) -> InjuryReport: ...


def _players(items: list[dict[str, str]]) -> tuple[PlayerStatus, ...]:
    return tuple(PlayerStatus(name=i["name"], status=i["status"], reason=i["reason"])
                 for i in items)


class FixtureInjuryProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_injuries(self, team: str, as_of: datetime) -> InjuryReport:
        raw = self._store.get("injuries", team)
        return InjuryReport(
            team=raw["team"], out=_players(raw["out"]),
            doubtful=_players(raw["doubtful"]), as_of=as_of, source="fixture",
        )
```

```python
# src/soccer/tools/head_to_head.py
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from soccer.models import H2HRecord, PastMeeting
from soccer.tools.fixtures import FixtureStore


class H2HProvider(Protocol):
    def get_h2h(self, home: str, away: str) -> H2HRecord: ...


class FixtureH2HProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_h2h(self, home: str, away: str) -> H2HRecord:
        raw = self._store.get("h2h", f"{home}|{away}")
        meetings = tuple(
            PastMeeting(date=datetime.fromisoformat(m["date"]), home=m["home"],
                        away=m["away"], home_goals=m["home_goals"],
                        away_goals=m["away_goals"])
            for m in raw["meetings"]
        )
        return H2HRecord(home=raw["home"], away=raw["away"], meetings=meetings,
                         home_wins=raw["home_wins"], draws=raw["draws"],
                         away_wins=raw["away_wins"], source="fixture")
```

```python
# src/soccer/tools/weather.py
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from soccer.models import WeatherReport
from soccer.tools.fixtures import FixtureStore


class WeatherProvider(Protocol):
    def get_weather(self, venue_id: str, kickoff: datetime) -> WeatherReport: ...


class FixtureWeatherProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_weather(self, venue_id: str, kickoff: datetime) -> WeatherReport:
        raw = self._store.get("weather", venue_id)
        return WeatherReport(
            venue_id=raw["venue_id"], kickoff=kickoff, temp_c=raw["temp_c"],
            wind_kph=raw["wind_kph"], precip_mm=raw["precip_mm"],
            condition=raw["condition"], source="fixture",
        )
```

```python
# src/soccer/tools/venue.py
from __future__ import annotations

from typing import Protocol

from soccer.models import VenueInfo
from soccer.tools.fixtures import FixtureStore


class VenueProvider(Protocol):
    def get_venue(self, venue_id: str) -> VenueInfo: ...


class FixtureVenueProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_venue(self, venue_id: str) -> VenueInfo:
        raw = self._store.get("venue", venue_id)
        return VenueInfo(
            venue_id=raw["venue_id"], name=raw["name"], city=raw["city"],
            surface=raw["surface"], capacity=raw["capacity"],
            altitude_m=raw["altitude_m"],
            home_advantage_hint=raw["home_advantage_hint"], source="fixture",
        )
```

```python
# src/soccer/tools/odds.py
from __future__ import annotations

from typing import Protocol

from soccer.models import MatchRef, OddsSnapshot
from soccer.tools.fixtures import FixtureStore


class OddsProvider(Protocol):
    def get_odds(self, match: MatchRef) -> OddsSnapshot: ...


class FixtureOddsProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_odds(self, match: MatchRef) -> OddsSnapshot:
        raw = self._store.get("odds", match.id)
        return OddsSnapshot(
            bookmaker=raw["bookmaker"], home=raw["home"], draw=raw["draw"],
            away=raw["away"], as_of=match.kickoff, source="fixture",
        )
```

```python
# src/soccer/tools/results.py
from __future__ import annotations

from typing import Protocol

from soccer.models import MatchRef, MatchResult
from soccer.tools.base import ToolError
from soccer.tools.fixtures import FixtureStore


class ResultProvider(Protocol):
    def get_result(self, match: MatchRef) -> MatchResult | None: ...


class FixtureResultProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_result(self, match: MatchRef) -> MatchResult | None:
        try:
            raw = self._store.get("results", match.id)
        except ToolError:
            return None
        return MatchResult(
            match_id=match.id, home_goals=raw["home_goals"],
            away_goals=raw["away_goals"], status=raw["status"], source="fixture",
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_providers.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/tools/ tests/tools/test_providers.py
git commit -m "feat: add provider protocols and fixture implementations"
```

---

### Task 6: HTTP provider stubs (`tools/http_stubs.py`)

**Files:**
- Create: `src/soccer/tools/http_stubs.py`
- Test: `tests/tools/test_http_stubs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/test_http_stubs.py
from datetime import datetime, timezone
import pytest
from soccer.models import MatchRef
from soccer.tools.http_stubs import HttpFormProvider

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)

def test_http_provider_not_implemented():
    with pytest.raises(NotImplementedError):
        HttpFormProvider(base_url="https://example.test").get_form("A", KICK)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/tools/test_http_stubs.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/tools/http_stubs.py
"""Real HTTP adapters. Wiring is sketched; endpoints are intentionally not
chosen yet (AGENTS.md: do not invent endpoints). Filling in a concrete API must
not change any caller — the Protocol signatures are identical to the fixture
providers."""
from __future__ import annotations

from datetime import datetime

from soccer.models import (
    H2HRecord, InjuryReport, MatchRef, MatchResult, OddsSnapshot, TeamForm,
    VenueInfo, WeatherReport,
)

_NOT_WIRED = "HTTP provider not wired to a concrete API yet; use provider_mode=fixture"


class HttpFormProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_form(self, team: str, as_of: datetime) -> TeamForm:
        raise NotImplementedError(_NOT_WIRED)


class HttpInjuryProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_injuries(self, team: str, as_of: datetime) -> InjuryReport:
        raise NotImplementedError(_NOT_WIRED)


class HttpH2HProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_h2h(self, home: str, away: str) -> H2HRecord:
        raise NotImplementedError(_NOT_WIRED)


class HttpWeatherProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_weather(self, venue_id: str, kickoff: datetime) -> WeatherReport:
        raise NotImplementedError(_NOT_WIRED)


class HttpVenueProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_venue(self, venue_id: str) -> VenueInfo:
        raise NotImplementedError(_NOT_WIRED)


class HttpOddsProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_odds(self, match: MatchRef) -> OddsSnapshot:
        raise NotImplementedError(_NOT_WIRED)


class HttpResultProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_result(self, match: MatchRef) -> MatchResult | None:
        raise NotImplementedError(_NOT_WIRED)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/tools/test_http_stubs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/tools/http_stubs.py tests/tools/test_http_stubs.py
git commit -m "feat: add HTTP provider stubs with stable signatures"
```

---

### Task 7: Tool registry (`registry.py`)

**Files:**
- Create: `src/soccer/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
import json
from soccer.registry import ToolRegistry, build_fixture_registry

def test_build_fixture_registry_exposes_providers(tmp_path):
    payload = {"form": {}, "injuries": {}, "h2h": {}, "weather": {},
               "venue": {}, "odds": {}, "results": {}}
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    reg = build_fixture_registry(path)
    assert reg.form is not None and reg.results is not None

def test_registry_tool_view_lists_capabilities(tmp_path):
    payload = {"form": {}, "injuries": {}, "h2h": {}, "weather": {},
               "venue": {}, "odds": {}, "results": {}}
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    reg = build_fixture_registry(path)
    names = {t.name for t in reg.as_tools()}
    assert {"form", "injuries", "h2h", "weather", "venue", "odds"} <= names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_registry.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/registry.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soccer.tools.base import Tool
from soccer.tools.fixtures import FixtureStore
from soccer.tools.form import FixtureFormProvider, FormProvider
from soccer.tools.head_to_head import FixtureH2HProvider, H2HProvider
from soccer.tools.injuries import FixtureInjuryProvider, InjuryProvider
from soccer.tools.odds import FixtureOddsProvider, OddsProvider
from soccer.tools.results import FixtureResultProvider, ResultProvider
from soccer.tools.venue import FixtureVenueProvider, VenueProvider
from soccer.tools.weather import FixtureWeatherProvider, WeatherProvider


@dataclass(frozen=True)
class ToolRegistry:
    form: FormProvider
    injuries: InjuryProvider
    h2h: H2HProvider
    weather: WeatherProvider
    venue: VenueProvider
    odds: OddsProvider
    results: ResultProvider

    def as_tools(self) -> list[Tool]:
        return [
            Tool("form", "recent team form", self.form.get_form),
            Tool("injuries", "injury/availability report", self.injuries.get_injuries),
            Tool("h2h", "head-to-head history", self.h2h.get_h2h),
            Tool("weather", "match-time weather", self.weather.get_weather),
            Tool("venue", "venue characteristics", self.venue.get_venue),
            Tool("odds", "bookmaker odds", self.odds.get_odds),
            Tool("results", "final result lookup", self.results.get_result),
        ]


def build_fixture_registry(fixture_path: Path) -> ToolRegistry:
    store = FixtureStore(fixture_path)
    return ToolRegistry(
        form=FixtureFormProvider(store),
        injuries=FixtureInjuryProvider(store),
        h2h=FixtureH2HProvider(store),
        weather=FixtureWeatherProvider(store),
        venue=FixtureVenueProvider(store),
        odds=FixtureOddsProvider(store),
        results=FixtureResultProvider(store),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_registry.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/registry.py tests/test_registry.py
git commit -m "feat: add tool registry with fixture builder and tool view"
```

---

### Task 8: Dossier builder with graceful degradation (`dossier.py`)

**Files:**
- Create: `src/soccer/dossier.py`
- Test: `tests/test_dossier.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dossier.py
import json
from datetime import datetime, timezone
from soccer.models import MatchRef
from soccer.registry import build_fixture_registry, ToolRegistry
from soccer.tools.base import ToolError
from soccer.dossier import build_dossier, dossier_digest

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)
REF = MatchRef(id="m1", competition="UCL", home="A", away="B",
               kickoff=KICK, venue_id="v1", season="2025-26")

def _full_payload():
    return {
        "form": {"A": {"team": "A", "last_n": ["W"], "gf": 5, "ga": 1, "points": 12,
                       "streak": "W1"},
                 "B": {"team": "B", "last_n": ["L"], "gf": 2, "ga": 4, "points": 5,
                       "streak": "L1"}},
        "injuries": {"A": {"team": "A", "out": [], "doubtful": []},
                     "B": {"team": "B", "out": [], "doubtful": []}},
        "h2h": {"A|B": {"home": "A", "away": "B", "meetings": [],
                        "home_wins": 0, "draws": 0, "away_wins": 0}},
        "weather": {"v1": {"venue_id": "v1", "temp_c": 10.0, "wind_kph": 5.0,
                           "precip_mm": 0.0, "condition": "clear"}},
        "venue": {"v1": {"venue_id": "v1", "name": "S", "city": "C", "surface": "grass",
                         "capacity": 50000, "altitude_m": 10,
                         "home_advantage_hint": 0.1}},
        "odds": {"m1": {"bookmaker": "b", "home": 1.8, "draw": 3.6, "away": 4.5}},
        "results": {},
    }

def test_full_dossier_has_no_missing(tmp_path):
    path = tmp_path / "f.json"
    path.write_text(json.dumps(_full_payload()))
    d = build_dossier(REF, build_fixture_registry(path))
    assert d.missing == ()
    assert d.form["home"].team == "A" and d.form["away"].team == "B"
    assert d.odds is not None

def test_missing_provider_recorded_not_fatal(tmp_path):
    payload = _full_payload()
    del payload["odds"]["m1"]  # odds lookup will raise ToolError
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    d = build_dossier(REF, build_fixture_registry(path))
    assert d.odds is None
    assert "odds" in d.missing

def test_dossier_digest_is_stable(tmp_path):
    path = tmp_path / "f.json"
    path.write_text(json.dumps(_full_payload()))
    reg = build_fixture_registry(path)
    d1 = build_dossier(REF, reg)
    d2 = build_dossier(REF, reg)
    assert dossier_digest(d1) == dossier_digest(d2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dossier.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/dossier.py
from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TypeVar

from soccer.models import MatchDossier, MatchRef
from soccer.registry import ToolRegistry
from soccer.tools.base import ToolError

T = TypeVar("T")


def _try(missing: list[str], name: str, fn: Callable[[], T]) -> T | None:
    try:
        return fn()
    except ToolError:
        missing.append(name)
        return None


def build_dossier(match: MatchRef, registry: ToolRegistry) -> MatchDossier:
    missing: list[str] = []
    form = {
        "home": _try(missing, "form:home",
                     lambda: registry.form.get_form(match.home, match.kickoff)),
        "away": _try(missing, "form:away",
                     lambda: registry.form.get_form(match.away, match.kickoff)),
    }
    injuries = {
        "home": _try(missing, "injuries:home",
                     lambda: registry.injuries.get_injuries(match.home, match.kickoff)),
        "away": _try(missing, "injuries:away",
                     lambda: registry.injuries.get_injuries(match.away, match.kickoff)),
    }
    h2h = _try(missing, "h2h", lambda: registry.h2h.get_h2h(match.home, match.away))
    weather = _try(missing, "weather",
                   lambda: registry.weather.get_weather(match.venue_id, match.kickoff))
    venue = _try(missing, "venue", lambda: registry.venue.get_venue(match.venue_id))
    odds = _try(missing, "odds", lambda: registry.odds.get_odds(match))
    return MatchDossier(match=match, form=form, injuries=injuries, h2h=h2h,
                        weather=weather, venue=venue, odds=odds,
                        missing=tuple(missing))


def dossier_digest(dossier: MatchDossier) -> str:
    parts = [
        dossier.match.id,
        f"form={[f.points if f else None for f in dossier.form.values()]}",
        f"odds={None if dossier.odds is None else (dossier.odds.home, dossier.odds.draw, dossier.odds.away)}",
        f"missing={sorted(dossier.missing)}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
```

Note: the test uses `d.missing == ()` for the full payload; `_try` appends only on `ToolError`. The second test deletes the `m1` odds key so `FixtureOddsProvider` raises `ToolError`, recorded as `"odds"`. Adjust the test's assertion `"odds" in d.missing` to match the recorded name `"odds"` (it is exact).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_dossier.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/dossier.py tests/test_dossier.py
git commit -m "feat: add dossier builder with graceful tool degradation"
```

---

### Task 9: Reasoner base + deterministic fake (`reasoning/base.py`, `reasoning/fake.py`)

**Files:**
- Create: `src/soccer/reasoning/__init__.py`, `src/soccer/reasoning/base.py`, `src/soccer/reasoning/fake.py`
- Test: `tests/reasoning/__init__.py`, `tests/reasoning/test_fake.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/reasoning/test_fake.py
import json
from datetime import datetime, timezone
import pytest
from soccer.models import MatchRef, Outcome, MatchResult, Prediction
from soccer.registry import build_fixture_registry
from soccer.dossier import build_dossier
from soccer.reasoning.fake import DeterministicReasoner

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)
REF = MatchRef(id="m1", competition="UCL", home="A", away="B",
               kickoff=KICK, venue_id="v1", season="2025-26")

def _payload(with_odds=True):
    p = {
        "form": {"A": {"team": "A", "last_n": ["W", "W"], "gf": 6, "ga": 1,
                       "points": 18, "streak": "W2"},
                 "B": {"team": "B", "last_n": ["L", "D"], "gf": 2, "ga": 5,
                       "points": 4, "streak": "L1"}},
        "injuries": {"A": {"team": "A", "out": [], "doubtful": []},
                     "B": {"team": "B", "out": [], "doubtful": []}},
        "h2h": {"A|B": {"home": "A", "away": "B", "meetings": [],
                        "home_wins": 2, "draws": 0, "away_wins": 0}},
        "weather": {"v1": {"venue_id": "v1", "temp_c": 10.0, "wind_kph": 5.0,
                           "precip_mm": 0.0, "condition": "clear"}},
        "venue": {"v1": {"venue_id": "v1", "name": "S", "city": "C",
                         "surface": "grass", "capacity": 50000, "altitude_m": 10,
                         "home_advantage_hint": 0.1}},
        "odds": {"m1": {"bookmaker": "b", "home": 1.7, "draw": 3.8, "away": 5.0}}
                if with_odds else {},
        "results": {},
    }
    return p

def _dossier(tmp_path, with_odds=True):
    path = tmp_path / "f.json"
    path.write_text(json.dumps(_payload(with_odds)))
    return build_dossier(REF, build_fixture_registry(path))

def test_fake_reasoner_is_deterministic(tmp_path):
    r = DeterministicReasoner()
    d = _dossier(tmp_path)
    a = r.predict(d)
    b = r.predict(d)
    assert a.probs == b.probs and a.confidence == b.confidence

def test_fake_reasoner_probs_valid_and_favours_strong_home(tmp_path):
    res = DeterministicReasoner().predict(_dossier(tmp_path))
    assert abs(sum(res.probs.values()) - 1.0) < 1e-6
    assert res.probs[Outcome.HOME] > res.probs[Outcome.AWAY]
    assert res.rationale

def test_fake_reasoner_without_odds_still_valid(tmp_path):
    res = DeterministicReasoner().predict(_dossier(tmp_path, with_odds=False))
    assert abs(sum(res.probs.values()) - 1.0) < 1e-6

def test_fake_self_evaluate_returns_text(tmp_path):
    r = DeterministicReasoner()
    pred = Prediction(id="x", match_ref=REF, created_at=KICK,
                      probs={Outcome.HOME: 0.6, Outcome.DRAW: 0.25, Outcome.AWAY: 0.15},
                      pick=Outcome.HOME, confidence=0.6, rationale="r",
                      market_probs=None, dossier_digest="d", reasoner_name="fake")
    result = MatchResult(match_id="m1", home_goals=0, away_goals=1,
                         status="finished", source="fixture")
    assert "AWAY" in r.self_evaluate(pred, result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/reasoning/test_fake.py -v`
Expected: FAIL with ImportError. (Create empty `tests/reasoning/__init__.py` first.)

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/reasoning/__init__.py
# (empty marker)
```

```python
# src/soccer/reasoning/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from soccer.models import MatchDossier, MatchResult, Outcome, Prediction


class ReasonerError(Exception):
    """Raised when a reasoner produces unusable output."""


@dataclass(frozen=True)
class ReasonResult:
    probs: dict[Outcome, float]
    confidence: float
    rationale: str


class Reasoner(Protocol):
    name: str

    def predict(self, dossier: MatchDossier) -> ReasonResult: ...

    def self_evaluate(self, prediction: Prediction, result: MatchResult) -> str: ...
```

```python
# src/soccer/reasoning/fake.py
from __future__ import annotations

from soccer.models import (
    MatchDossier, MatchOutcome, MatchResult, Outcome, Prediction, normalize_probs,
)
from soccer.reasoning.base import ReasonResult

_POINTS = {MatchOutcome.WIN: 3.0, MatchOutcome.DRAW: 1.0, MatchOutcome.LOSS: 0.0}


def _form_strength(dossier: MatchDossier, side: str) -> float:
    form = dossier.form.get(side)
    if form is None or not form.last_n:
        return 1.0
    return sum(_POINTS[o] for o in form.last_n) / len(form.last_n)


class DeterministicReasoner:
    """Blends market-implied odds with a fixed form/H2H adjustment. No randomness."""

    name = "fake"

    def predict(self, dossier: MatchDossier) -> ReasonResult:
        if dossier.odds is not None:
            base = dict(dossier.odds.implied_probs)
        else:
            base = {Outcome.HOME: 0.4, Outcome.DRAW: 0.3, Outcome.AWAY: 0.3}
        home_str = _form_strength(dossier, "home")
        away_str = _form_strength(dossier, "away")
        # Tilt toward the in-form side; +0.05 weight per point of form gap.
        gap = (home_str - away_str) * 0.05
        adjusted = {
            Outcome.HOME: max(base[Outcome.HOME] + gap, 1e-6),
            Outcome.DRAW: max(base[Outcome.DRAW], 1e-6),
            Outcome.AWAY: max(base[Outcome.AWAY] - gap, 1e-6),
        }
        probs = normalize_probs(adjusted)
        pick = max(probs, key=lambda k: probs[k])
        confidence = round(probs[pick], 4)
        rationale = (
            f"Market-implied base adjusted by form gap {gap:+.3f} "
            f"(home {home_str:.2f} vs away {away_str:.2f}); "
            f"missing data: {list(dossier.missing) or 'none'}."
        )
        return ReasonResult(probs=probs, confidence=confidence, rationale=rationale)

    def self_evaluate(self, prediction: Prediction, result: MatchResult) -> str:
        hit = "correct" if prediction.pick is result.outcome else "wrong"
        return (
            f"Prediction was {hit}: picked {prediction.pick.value} "
            f"(p={prediction.probs[prediction.pick]:.2f}), "
            f"actual {result.outcome.value} "
            f"({result.home_goals}-{result.away_goals})."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/reasoning/test_fake.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/reasoning/ tests/reasoning/
git commit -m "feat: add reasoner protocol and deterministic fake reasoner"
```

---

### Task 10: Prompt rendering + JSON parsing (`reasoning/prompt.py`)

**Files:**
- Create: `src/soccer/reasoning/prompt.py`
- Test: `tests/reasoning/test_prompt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/reasoning/test_prompt.py
import json
from datetime import datetime, timezone
import pytest
from soccer.models import MatchRef, Outcome
from soccer.registry import build_fixture_registry
from soccer.dossier import build_dossier
from soccer.reasoning.base import ReasonerError
from soccer.reasoning.prompt import render_prompt, parse_reason_json

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)
REF = MatchRef(id="m1", competition="UCL", home="A", away="B",
               kickoff=KICK, venue_id="v1", season="2025-26")

def _dossier(tmp_path):
    payload = {
        "form": {"A": {"team": "A", "last_n": ["W"], "gf": 3, "ga": 0, "points": 9,
                       "streak": "W1"},
                 "B": {"team": "B", "last_n": ["L"], "gf": 0, "ga": 3, "points": 1,
                       "streak": "L1"}},
        "injuries": {"A": {"team": "A", "out": [], "doubtful": []},
                     "B": {"team": "B", "out": [], "doubtful": []}},
        "h2h": {"A|B": {"home": "A", "away": "B", "meetings": [],
                        "home_wins": 1, "draws": 0, "away_wins": 0}},
        "weather": {"v1": {"venue_id": "v1", "temp_c": 9.0, "wind_kph": 4.0,
                           "precip_mm": 0.0, "condition": "clear"}},
        "venue": {"v1": {"venue_id": "v1", "name": "S", "city": "C", "surface": "grass",
                         "capacity": 1000, "altitude_m": 0, "home_advantage_hint": 0.1}},
        "odds": {"m1": {"bookmaker": "b", "home": 1.8, "draw": 3.5, "away": 4.0}},
        "results": {},
    }
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    return build_dossier(REF, build_fixture_registry(path))

def test_render_prompt_mentions_teams_and_json(tmp_path):
    text = render_prompt(_dossier(tmp_path))
    assert "A" in text and "B" in text
    assert "JSON" in text or "json" in text

def test_parse_valid_json():
    raw = '{"home": 0.5, "draw": 0.3, "away": 0.2, "confidence": 0.55, "rationale": "x"}'
    res = parse_reason_json(raw)
    assert res.probs[Outcome.HOME] == 0.5 and res.confidence == 0.55

def test_parse_renormalises_probs():
    raw = '{"home": 2, "draw": 1, "away": 1, "confidence": 0.5, "rationale": "x"}'
    res = parse_reason_json(raw)
    assert abs(sum(res.probs.values()) - 1.0) < 1e-6

def test_parse_rejects_garbage():
    with pytest.raises(ReasonerError):
        parse_reason_json("not json at all")

def test_parse_rejects_missing_keys():
    with pytest.raises(ReasonerError):
        parse_reason_json('{"home": 0.5}')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/reasoning/test_prompt.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/reasoning/prompt.py
from __future__ import annotations

import json

from soccer.models import MatchDossier, Outcome, normalize_probs
from soccer.reasoning.base import ReasonResult, ReasonerError

_INSTRUCTIONS = (
    "You are a football match analyst. Using the dossier, estimate the probability "
    "of each 1X2 outcome. Respond with ONLY a JSON object with keys: "
    '"home", "draw", "away" (numbers), "confidence" (0-1), "rationale" (string).'
)


def _form_line(dossier: MatchDossier, side: str) -> str:
    form = dossier.form.get(side)
    if form is None:
        return f"{side}: form unavailable"
    return (f"{side} ({form.team}): last={[o.value for o in form.last_n]} "
            f"pts={form.points} gf={form.gf} ga={form.ga} streak={form.streak}")


def render_prompt(dossier: MatchDossier) -> str:
    m = dossier.match
    odds = dossier.odds
    odds_line = ("odds unavailable" if odds is None
                 else f"odds H/D/A = {odds.home}/{odds.draw}/{odds.away}")
    h2h = dossier.h2h
    h2h_line = ("h2h unavailable" if h2h is None
                else f"h2h home_wins={h2h.home_wins} draws={h2h.draws} away_wins={h2h.away_wins}")
    weather = dossier.weather
    weather_line = ("weather unavailable" if weather is None
                    else f"weather {weather.condition} {weather.temp_c}C wind {weather.wind_kph}kph")
    lines = [
        _INSTRUCTIONS,
        f"Match: {m.home} (home) vs {m.away} (away), {m.competition} {m.season}.",
        _form_line(dossier, "home"),
        _form_line(dossier, "away"),
        h2h_line,
        weather_line,
        odds_line,
        f"Missing data: {list(dossier.missing) or 'none'}.",
    ]
    return "\n".join(lines)


def parse_reason_json(raw: str) -> ReasonResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReasonerError(f"reasoner did not return JSON: {exc}") from exc
    required = {"home", "draw", "away", "confidence", "rationale"}
    if not required <= set(data):
        raise ReasonerError(f"reasoner JSON missing keys: {required - set(data)}")
    try:
        raw_probs = {
            Outcome.HOME: float(data["home"]),
            Outcome.DRAW: float(data["draw"]),
            Outcome.AWAY: float(data["away"]),
        }
        confidence = float(data["confidence"])
    except (TypeError, ValueError) as exc:
        raise ReasonerError(f"reasoner returned non-numeric value: {exc}") from exc
    if any(v < 0 for v in raw_probs.values()):
        raise ReasonerError("reasoner returned negative probability")
    probs = normalize_probs(raw_probs)
    confidence = min(max(confidence, 0.0), 1.0)
    return ReasonResult(probs=probs, confidence=confidence,
                        rationale=str(data["rationale"]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/reasoning/test_prompt.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/reasoning/prompt.py tests/reasoning/test_prompt.py
git commit -m "feat: add prompt rendering and reasoner JSON parsing"
```

---

### Task 11: Ollama reasoner with injected transport (`reasoning/ollama.py`)

**Files:**
- Create: `src/soccer/reasoning/ollama.py`
- Test: `tests/reasoning/test_ollama.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/reasoning/test_ollama.py
import json
from datetime import datetime, timezone
import pytest
from soccer.models import MatchRef, Outcome, MatchResult, Prediction
from soccer.registry import build_fixture_registry
from soccer.dossier import build_dossier
from soccer.reasoning.base import ReasonerError
from soccer.reasoning.ollama import OllamaReasoner

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)
REF = MatchRef(id="m1", competition="UCL", home="A", away="B",
               kickoff=KICK, venue_id="v1", season="2025-26")

def _dossier(tmp_path):
    payload = {
        "form": {"A": {"team": "A", "last_n": ["W"], "gf": 3, "ga": 0, "points": 9,
                       "streak": "W1"},
                 "B": {"team": "B", "last_n": ["L"], "gf": 0, "ga": 3, "points": 1,
                       "streak": "L1"}},
        "injuries": {"A": {"team": "A", "out": [], "doubtful": []},
                     "B": {"team": "B", "out": [], "doubtful": []}},
        "h2h": {"A|B": {"home": "A", "away": "B", "meetings": [],
                        "home_wins": 1, "draws": 0, "away_wins": 0}},
        "weather": {"v1": {"venue_id": "v1", "temp_c": 9.0, "wind_kph": 4.0,
                           "precip_mm": 0.0, "condition": "clear"}},
        "venue": {"v1": {"venue_id": "v1", "name": "S", "city": "C", "surface": "grass",
                         "capacity": 1000, "altitude_m": 0, "home_advantage_hint": 0.1}},
        "odds": {"m1": {"bookmaker": "b", "home": 1.8, "draw": 3.5, "away": 4.0}},
        "results": {},
    }
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    return build_dossier(REF, build_fixture_registry(path))

def test_ollama_predict_parses_transport_response(tmp_path):
    captured = {}
    def fake_post(url, payload, timeout):
        captured["url"] = url
        captured["model"] = payload["model"]
        content = '{"home": 0.6, "draw": 0.25, "away": 0.15, "confidence": 0.6, "rationale": "home strong"}'
        return {"message": {"content": content}}
    r = OllamaReasoner(host="http://localhost:11434", model="gemma4:12b-mlx",
                       timeout=5, post_json=fake_post)
    res = r.predict(_dossier(tmp_path))
    assert res.probs[Outcome.HOME] == 0.6
    assert captured["model"] == "gemma4:12b-mlx"
    assert captured["url"].endswith("/api/chat")

def test_ollama_predict_raises_on_bad_json(tmp_path):
    def fake_post(url, payload, timeout):
        return {"message": {"content": "definitely not json"}}
    r = OllamaReasoner(host="http://localhost:11434", model="m", timeout=5,
                       post_json=fake_post)
    with pytest.raises(ReasonerError):
        r.predict(_dossier(tmp_path))

def test_ollama_predict_raises_on_malformed_envelope(tmp_path):
    def fake_post(url, payload, timeout):
        return {"unexpected": True}
    r = OllamaReasoner(host="http://localhost:11434", model="m", timeout=5,
                       post_json=fake_post)
    with pytest.raises(ReasonerError):
        r.predict(_dossier(tmp_path))

def test_ollama_self_evaluate(tmp_path):
    def fake_post(url, payload, timeout):
        return {"message": {"content": "I overweighted home form."}}
    r = OllamaReasoner(host="http://localhost:11434", model="m", timeout=5,
                       post_json=fake_post)
    pred = Prediction(id="x", match_ref=REF, created_at=KICK,
                      probs={Outcome.HOME: 0.6, Outcome.DRAW: 0.25, Outcome.AWAY: 0.15},
                      pick=Outcome.HOME, confidence=0.6, rationale="r",
                      market_probs=None, dossier_digest="d", reasoner_name="ollama")
    result = MatchResult(match_id="m1", home_goals=0, away_goals=2,
                         status="finished", source="fixture")
    assert "overweighted" in r.self_evaluate(pred, result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/reasoning/test_ollama.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/reasoning/ollama.py
from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from soccer.models import MatchDossier, MatchResult, Prediction
from soccer.reasoning.base import ReasonResult, ReasonerError
from soccer.reasoning.prompt import parse_reason_json, render_prompt

PostJson = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _urllib_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ReasonerError(f"ollama request failed: {exc}") from exc


class OllamaReasoner:
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: float,
                 post_json: PostJson = _urllib_post) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._post = post_json

    def _chat(self, prompt: str, *, json_format: bool) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "seed": 7},
        }
        if json_format:
            payload["format"] = "json"
        data = self._post(f"{self._host}/api/chat", payload, self._timeout)
        try:
            return str(data["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise ReasonerError(f"unexpected ollama response shape: {exc}") from exc

    def predict(self, dossier: MatchDossier) -> ReasonResult:
        content = self._chat(render_prompt(dossier), json_format=True)
        return parse_reason_json(content)

    def self_evaluate(self, prediction: Prediction, result: MatchResult) -> str:
        prompt = (
            f"You predicted {prediction.pick.value} with probabilities "
            f"{ {k.value: round(v, 3) for k, v in prediction.probs.items()} }. "
            f"The actual result was {result.outcome.value} "
            f"({result.home_goals}-{result.away_goals}). "
            "In 2-3 sentences, critique what your reasoning got wrong or right."
        )
        return self._chat(prompt, json_format=False)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/reasoning/test_ollama.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/reasoning/ollama.py tests/reasoning/test_ollama.py
git commit -m "feat: add ollama reasoner with injected transport"
```

---

### Task 12: Prediction agent (`agent.py`)

**Files:**
- Create: `src/soccer/agent.py`
- Test: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent.py
import json
from datetime import datetime, timezone
from soccer.models import MatchRef, Outcome
from soccer.registry import build_fixture_registry
from soccer.reasoning.fake import DeterministicReasoner
from soccer.agent import PredictionAgent

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)
REF = MatchRef(id="m1", competition="UCL", home="A", away="B",
               kickoff=KICK, venue_id="v1", season="2025-26")

def _registry(tmp_path):
    payload = {
        "form": {"A": {"team": "A", "last_n": ["W", "W"], "gf": 6, "ga": 1,
                       "points": 18, "streak": "W2"},
                 "B": {"team": "B", "last_n": ["L"], "gf": 1, "ga": 4, "points": 3,
                       "streak": "L1"}},
        "injuries": {"A": {"team": "A", "out": [], "doubtful": []},
                     "B": {"team": "B", "out": [], "doubtful": []}},
        "h2h": {"A|B": {"home": "A", "away": "B", "meetings": [],
                        "home_wins": 1, "draws": 0, "away_wins": 0}},
        "weather": {"v1": {"venue_id": "v1", "temp_c": 9.0, "wind_kph": 4.0,
                           "precip_mm": 0.0, "condition": "clear"}},
        "venue": {"v1": {"venue_id": "v1", "name": "S", "city": "C", "surface": "grass",
                         "capacity": 1000, "altitude_m": 0, "home_advantage_hint": 0.1}},
        "odds": {"m1": {"bookmaker": "b", "home": 1.7, "draw": 3.8, "away": 5.0}},
        "results": {},
    }
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    return build_fixture_registry(path)

def test_agent_produces_valid_prediction(tmp_path):
    agent = PredictionAgent(registry=_registry(tmp_path),
                            reasoner=DeterministicReasoner(),
                            clock=lambda: NOW)
    pred = agent.predict(REF)
    assert pred.pick in Outcome
    assert abs(sum(pred.probs.values()) - 1.0) < 1e-6
    assert pred.reasoner_name == "fake"
    assert pred.market_probs is not None  # odds present
    assert pred.created_at == NOW
    assert pred.id  # stable id assigned

def test_agent_id_is_deterministic_for_same_clock(tmp_path):
    agent = PredictionAgent(registry=_registry(tmp_path),
                            reasoner=DeterministicReasoner(), clock=lambda: NOW)
    assert agent.predict(REF).id == agent.predict(REF).id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/agent.py
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from soccer.dossier import build_dossier, dossier_digest
from soccer.models import MatchRef, Prediction, make_prediction_id
from soccer.reasoning.base import Reasoner
from soccer.registry import ToolRegistry


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PredictionAgent:
    def __init__(self, registry: ToolRegistry, reasoner: Reasoner,
                 clock: Callable[[], datetime] = _utc_now) -> None:
        self._registry = registry
        self._reasoner = reasoner
        self._clock = clock

    def predict(self, match: MatchRef) -> Prediction:
        dossier = build_dossier(match, self._registry)
        result = self._reasoner.predict(dossier)
        created_at = self._clock()
        pick = max(result.probs, key=lambda k: result.probs[k])
        market = dossier.odds.implied_probs if dossier.odds is not None else None
        return Prediction(
            id=make_prediction_id(match.id, created_at),
            match_ref=match, created_at=created_at, probs=result.probs, pick=pick,
            confidence=result.confidence, rationale=result.rationale,
            market_probs=market, dossier_digest=dossier_digest(dossier),
            reasoner_name=self._reasoner.name,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/agent.py tests/test_agent.py
git commit -m "feat: add prediction agent orchestrating dossier and reasoner"
```

---

### Task 13: Prediction store (`store.py`)

**Files:**
- Create: `src/soccer/store.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
from datetime import datetime, timezone
from soccer.models import MatchRef, Outcome, Prediction, MatchResult, Evaluation
from soccer.store import PredictionStore

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)
REF = MatchRef(id="m1", competition="UCL", home="A", away="B",
               kickoff=KICK, venue_id="v1", season="2025-26")

def _pred(pid="p1", mid="m1") -> Prediction:
    ref = MatchRef(id=mid, competition="UCL", home="A", away="B",
                   kickoff=KICK, venue_id="v1", season="2025-26")
    return Prediction(id=pid, match_ref=ref, created_at=KICK,
                      probs={Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2},
                      pick=Outcome.HOME, confidence=0.5, rationale="r",
                      market_probs=None, dossier_digest="d", reasoner_name="fake")

def _store(tmp_path) -> PredictionStore:
    return PredictionStore(predictions_path=tmp_path / "p.jsonl",
                           results_path=tmp_path / "r.jsonl",
                           evaluations_path=tmp_path / "e.jsonl")

def test_prediction_round_trip(tmp_path):
    s = _store(tmp_path)
    s.append_prediction(_pred())
    loaded = s.load_predictions()
    assert len(loaded) == 1 and loaded[0] == _pred()

def test_pending_excludes_evaluated(tmp_path):
    s = _store(tmp_path)
    s.append_prediction(_pred("p1", "m1"))
    s.append_prediction(_pred("p2", "m2"))
    result = MatchResult(match_id="m1", home_goals=1, away_goals=0,
                         status="finished", source="fixture")
    s.append_result(result)
    s.append_evaluation(Evaluation(prediction_id="p1", result=result, correct=True,
                                   brier=0.1, log_loss=0.2, beat_market=True,
                                   self_critique="ok", evaluated_at=KICK))
    pending = s.pending()
    assert [p.id for p in pending] == ["p2"]

def test_load_empty_returns_empty(tmp_path):
    assert _store(tmp_path).load_predictions() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/store.py
from __future__ import annotations

import json
from pathlib import Path

from soccer.models import (
    Evaluation, MatchResult, Prediction, evaluation_from_dict, evaluation_to_dict,
    prediction_from_dict, prediction_to_dict, result_from_dict, result_to_dict,
)


class PredictionStore:
    def __init__(self, predictions_path: Path, results_path: Path,
                 evaluations_path: Path) -> None:
        self._predictions = Path(predictions_path)
        self._results = Path(results_path)
        self._evaluations = Path(evaluations_path)

    @staticmethod
    def _append(path: Path, record: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    @staticmethod
    def _read(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def append_prediction(self, prediction: Prediction) -> None:
        self._append(self._predictions, prediction_to_dict(prediction))

    def append_result(self, result: MatchResult) -> None:
        self._append(self._results, result_to_dict(result))

    def append_evaluation(self, evaluation: Evaluation) -> None:
        self._append(self._evaluations, evaluation_to_dict(evaluation))

    def load_predictions(self) -> list[Prediction]:
        return [prediction_from_dict(r) for r in self._read(self._predictions)]

    def load_results(self) -> list[MatchResult]:
        return [result_from_dict(r) for r in self._read(self._results)]

    def load_evaluations(self) -> list[Evaluation]:
        return [evaluation_from_dict(r) for r in self._read(self._evaluations)]

    def pending(self) -> list[Prediction]:
        evaluated = {e.prediction_id for e in self.load_evaluations()}
        return [p for p in self.load_predictions() if p.id not in evaluated]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_store.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/store.py tests/test_store.py
git commit -m "feat: add JSONL prediction store"
```

---

### Task 14: Evaluation metrics (`evaluation.py`)

**Files:**
- Create: `src/soccer/evaluation.py`
- Test: `tests/test_evaluation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluation.py
import math
from datetime import datetime, timezone
import pytest
from soccer.models import MatchRef, Outcome, Prediction, MatchResult
from soccer.evaluation import (
    brier_score, log_loss, beat_market, score, calibration_bins, CalibrationBin,
)

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)
REF = MatchRef(id="m1", competition="UCL", home="A", away="B",
               kickoff=KICK, venue_id="v1", season="2025-26")

def _pred(probs, market=None) -> Prediction:
    pick = max(probs, key=lambda k: probs[k])
    return Prediction(id="p1", match_ref=REF, created_at=KICK, probs=probs, pick=pick,
                      confidence=probs[pick], rationale="r", market_probs=market,
                      dossier_digest="d", reasoner_name="fake")

HOME_RESULT = MatchResult(match_id="m1", home_goals=2, away_goals=0,
                          status="finished", source="fixture")

def test_brier_perfect_prediction_is_zero():
    probs = {Outcome.HOME: 1.0, Outcome.DRAW: 0.0, Outcome.AWAY: 0.0}
    assert brier_score(probs, Outcome.HOME) == pytest.approx(0.0)

def test_brier_known_value():
    probs = {Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2}
    # (0.5-1)^2 + (0.3-0)^2 + (0.2-0)^2 = 0.25 + 0.09 + 0.04 = 0.38
    assert brier_score(probs, Outcome.HOME) == pytest.approx(0.38)

def test_log_loss_known_value():
    probs = {Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2}
    assert log_loss(probs, Outcome.HOME) == pytest.approx(-math.log(0.5))

def test_beat_market_true_when_model_more_confident_in_actual():
    model = {Outcome.HOME: 0.6, Outcome.DRAW: 0.25, Outcome.AWAY: 0.15}
    market = {Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2}
    assert beat_market(model, market, Outcome.HOME) is True

def test_beat_market_false_without_market():
    model = {Outcome.HOME: 0.6, Outcome.DRAW: 0.25, Outcome.AWAY: 0.15}
    assert beat_market(model, None, Outcome.HOME) is False

def test_score_builds_full_evaluation():
    pred = _pred({Outcome.HOME: 0.6, Outcome.DRAW: 0.25, Outcome.AWAY: 0.15},
                 market={Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2})
    ev = score(pred, HOME_RESULT, "good call", evaluated_at=KICK)
    assert ev.correct is True
    assert ev.beat_market is True
    assert ev.self_critique == "good call"
    assert ev.brier == pytest.approx(0.6**2 - 2*0.6 + 1 + 0.25**2 + 0.15**2)

def test_calibration_bins_group_by_confidence():
    preds = [
        _pred({Outcome.HOME: 0.9, Outcome.DRAW: 0.05, Outcome.AWAY: 0.05}),
        _pred({Outcome.HOME: 0.85, Outcome.DRAW: 0.1, Outcome.AWAY: 0.05}),
    ]
    outcomes = [Outcome.HOME, Outcome.AWAY]  # one hit, one miss
    bins = calibration_bins(preds, outcomes, n_bins=10)
    high = [b for b in bins if b.count > 0 and b.lower >= 0.8][0]
    assert high.count == 2
    assert high.observed == pytest.approx(0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluation.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/evaluation.py
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from soccer.models import Evaluation, MatchResult, Outcome, Prediction

_EPS = 1e-15


def brier_score(probs: dict[Outcome, float], actual: Outcome) -> float:
    return sum((probs[o] - (1.0 if o is actual else 0.0)) ** 2 for o in Outcome)


def log_loss(probs: dict[Outcome, float], actual: Outcome) -> float:
    return -math.log(min(max(probs[actual], _EPS), 1.0))


def beat_market(probs: dict[Outcome, float],
                market: dict[Outcome, float] | None, actual: Outcome) -> bool:
    if market is None:
        return False
    return probs[actual] > market[actual]


def score(prediction: Prediction, result: MatchResult, self_critique: str,
          evaluated_at: datetime) -> Evaluation:
    actual = result.outcome
    return Evaluation(
        prediction_id=prediction.id, result=result,
        correct=prediction.pick is actual,
        brier=brier_score(prediction.probs, actual),
        log_loss=log_loss(prediction.probs, actual),
        beat_market=beat_market(prediction.probs, prediction.market_probs, actual),
        self_critique=self_critique, evaluated_at=evaluated_at,
    )


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    observed: float  # fraction correct among predictions whose confidence is in-band


def calibration_bins(predictions: list[Prediction], outcomes: list[Outcome],
                     n_bins: int = 10) -> list[CalibrationBin]:
    width = 1.0 / n_bins
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lower, upper = i * width, (i + 1) * width
        members = [
            (p, o) for p, o in zip(predictions, outcomes, strict=True)
            if lower <= p.confidence < upper or (i == n_bins - 1 and p.confidence == 1.0)
        ]
        count = len(members)
        observed = (sum(1 for p, o in members if p.pick is o) / count
                    if count else 0.0)
        bins.append(CalibrationBin(lower=lower, upper=upper, count=count,
                                   observed=observed))
    return bins
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluation.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/evaluation.py tests/test_evaluation.py
git commit -m "feat: add evaluation metrics and calibration"
```

---

### Task 15: Settle flow (`settle.py`)

**Files:**
- Create: `src/soccer/settle.py`
- Test: `tests/test_settle.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_settle.py
import json
from datetime import datetime, timezone
from soccer.models import MatchRef, Outcome, Prediction
from soccer.registry import build_fixture_registry
from soccer.reasoning.fake import DeterministicReasoner
from soccer.store import PredictionStore
from soccer.settle import settle

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 4, 2, 9, 0, tzinfo=timezone.utc)

def _ref(mid):
    return MatchRef(id=mid, competition="UCL", home="A", away="B",
                    kickoff=KICK, venue_id="v1", season="2025-26")

def _pred(pid, mid):
    return Prediction(id=pid, match_ref=_ref(mid), created_at=KICK,
                      probs={Outcome.HOME: 0.6, Outcome.DRAW: 0.25, Outcome.AWAY: 0.15},
                      pick=Outcome.HOME, confidence=0.6, rationale="r",
                      market_probs={Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2},
                      dossier_digest="d", reasoner_name="fake")

def _registry(tmp_path):
    payload = {"form": {}, "injuries": {}, "h2h": {}, "weather": {}, "venue": {},
               "odds": {},
               "results": {"m1": {"home_goals": 2, "away_goals": 0,
                                  "status": "finished"}}}
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    return build_fixture_registry(path)

def _store(tmp_path):
    return PredictionStore(predictions_path=tmp_path / "p.jsonl",
                           results_path=tmp_path / "r.jsonl",
                           evaluations_path=tmp_path / "e.jsonl")

def test_settle_scores_finished_and_skips_unfinished(tmp_path):
    store = _store(tmp_path)
    store.append_prediction(_pred("p1", "m1"))  # finished in fixtures
    store.append_prediction(_pred("p2", "m2"))  # no result → skipped
    evals = settle(store, _registry(tmp_path), DeterministicReasoner(),
                   clock=lambda: NOW)
    assert [e.prediction_id for e in evals] == ["p1"]
    assert evals[0].correct is True
    assert evals[0].beat_market is True
    assert store.pending()[0].id == "p2"

def test_settle_is_idempotent(tmp_path):
    store = _store(tmp_path)
    store.append_prediction(_pred("p1", "m1"))
    settle(store, _registry(tmp_path), DeterministicReasoner(), clock=lambda: NOW)
    second = settle(store, _registry(tmp_path), DeterministicReasoner(),
                    clock=lambda: NOW)
    assert second == []  # already evaluated
    assert len(store.load_evaluations()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_settle.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/settle.py
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from soccer.evaluation import score
from soccer.models import Evaluation
from soccer.reasoning.base import Reasoner
from soccer.registry import ToolRegistry
from soccer.store import PredictionStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def settle(store: PredictionStore, registry: ToolRegistry, reasoner: Reasoner,
           clock: Callable[[], datetime] = _utc_now) -> list[Evaluation]:
    new_evaluations: list[Evaluation] = []
    for prediction in store.pending():
        result = registry.results.get_result(prediction.match_ref)
        if result is None:
            continue
        critique = reasoner.self_evaluate(prediction, result)
        evaluation = score(prediction, result, critique, evaluated_at=clock())
        store.append_result(result)
        store.append_evaluation(evaluation)
        new_evaluations.append(evaluation)
    return new_evaluations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_settle.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/settle.py tests/test_settle.py
git commit -m "feat: add on-demand settle and self-evaluation flow"
```

---

### Task 16: Eval harness (`harness.py`)

**Files:**
- Create: `src/soccer/harness.py`
- Test: `tests/test_harness.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_harness.py
import json
from datetime import datetime, timezone
from soccer.models import MatchRef, MatchResult, Outcome
from soccer.registry import build_fixture_registry
from soccer.reasoning.fake import DeterministicReasoner
from soccer.agent import PredictionAgent
from soccer.harness import Scenario, run_scenario

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc)

def _ref(mid):
    return MatchRef(id=mid, competition="UCL", home="A", away="B",
                    kickoff=KICK, venue_id="v1", season="2025-26")

def _registry(tmp_path):
    payload = {
        "form": {"A": {"team": "A", "last_n": ["W", "W"], "gf": 6, "ga": 1,
                       "points": 18, "streak": "W2"},
                 "B": {"team": "B", "last_n": ["L"], "gf": 1, "ga": 4, "points": 3,
                       "streak": "L1"}},
        "injuries": {"A": {"team": "A", "out": [], "doubtful": []},
                     "B": {"team": "B", "out": [], "doubtful": []}},
        "h2h": {"A|B": {"home": "A", "away": "B", "meetings": [],
                        "home_wins": 1, "draws": 0, "away_wins": 0}},
        "weather": {"v1": {"venue_id": "v1", "temp_c": 9.0, "wind_kph": 4.0,
                           "precip_mm": 0.0, "condition": "clear"}},
        "venue": {"v1": {"venue_id": "v1", "name": "S", "city": "C", "surface": "grass",
                         "capacity": 1000, "altitude_m": 0, "home_advantage_hint": 0.1}},
        "odds": {"m1": {"bookmaker": "b", "home": 1.7, "draw": 3.8, "away": 5.0},
                 "m2": {"bookmaker": "b", "home": 1.7, "draw": 3.8, "away": 5.0}},
        "results": {},
    }
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    return build_fixture_registry(path)

def test_run_scenario_reports_metrics(tmp_path):
    registry = _registry(tmp_path)
    agent = PredictionAgent(registry=registry, reasoner=DeterministicReasoner(),
                            clock=lambda: NOW)
    scenario = Scenario(
        name="t", registry=registry, matches=[_ref("m1"), _ref("m2")],
        results={
            "m1": MatchResult(match_id="m1", home_goals=2, away_goals=0,
                              status="finished", source="fixture"),
            "m2": MatchResult(match_id="m2", home_goals=0, away_goals=1,
                              status="finished", source="fixture"),
        })
    report = run_scenario(scenario, agent)
    assert report.n == 2
    assert 0.0 <= report.accuracy <= 1.0
    assert report.accuracy == 0.5  # m1 home hit, m2 away miss
    assert report.mean_brier > 0
    assert report.market_baseline.mean_log_loss > 0
    assert len(report.per_match) == 2
    # edge = our log loss minus market log loss
    assert report.edge_vs_market == (report.mean_log_loss
                                     - report.market_baseline.mean_log_loss)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_harness.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/harness.py
from __future__ import annotations

from dataclasses import dataclass

from soccer.agent import PredictionAgent
from soccer.evaluation import (
    CalibrationBin, brier_score, calibration_bins, log_loss,
)
from soccer.models import MatchRef, MatchResult, Outcome
from soccer.registry import ToolRegistry


@dataclass(frozen=True)
class Scenario:
    name: str
    registry: ToolRegistry
    matches: list[MatchRef]
    results: dict[str, MatchResult]


@dataclass(frozen=True)
class MatchScore:
    match_id: str
    pick: Outcome
    actual: Outcome
    correct: bool
    brier: float
    log_loss: float


@dataclass(frozen=True)
class MarketBaseline:
    mean_brier: float
    mean_log_loss: float


@dataclass(frozen=True)
class EvalReport:
    scenario: str
    n: int
    accuracy: float
    mean_brier: float
    mean_log_loss: float
    calibration: list[CalibrationBin]
    market_baseline: MarketBaseline
    edge_vs_market: float
    per_match: list[MatchScore]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_scenario(scenario: Scenario, agent: PredictionAgent) -> EvalReport:
    per_match: list[MatchScore] = []
    outcomes: list[Outcome] = []
    predictions = []
    market_brier: list[float] = []
    market_log_loss: list[float] = []

    for match in scenario.matches:
        actual = scenario.results[match.id].outcome
        prediction = agent.predict(match)
        predictions.append(prediction)
        outcomes.append(actual)
        per_match.append(MatchScore(
            match_id=match.id, pick=prediction.pick, actual=actual,
            correct=prediction.pick is actual,
            brier=brier_score(prediction.probs, actual),
            log_loss=log_loss(prediction.probs, actual),
        ))
        if prediction.market_probs is not None:
            market_brier.append(brier_score(prediction.market_probs, actual))
            market_log_loss.append(log_loss(prediction.market_probs, actual))

    n = len(scenario.matches)
    mean_log = _mean([s.log_loss for s in per_match])
    market_mean_log = _mean(market_log_loss)
    return EvalReport(
        scenario=scenario.name, n=n,
        accuracy=_mean([1.0 if s.correct else 0.0 for s in per_match]),
        mean_brier=_mean([s.brier for s in per_match]),
        mean_log_loss=mean_log,
        calibration=calibration_bins(predictions, outcomes),
        market_baseline=MarketBaseline(mean_brier=_mean(market_brier),
                                       mean_log_loss=market_mean_log),
        edge_vs_market=mean_log - market_mean_log,
        per_match=per_match,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_harness.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/soccer/harness.py tests/test_harness.py
git commit -m "feat: add offline eval harness with market baseline"
```

---

### Task 17: Scenario fixtures (UCL + WC final) (`fixtures/`, `scenarios.py`)

**Files:**
- Create: `fixtures/ucl-2025-26.json`, `fixtures/wc-2026-final.json`
- Create: `src/soccer/scenarios.py`
- Test: `tests/test_scenarios.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenarios.py
import pytest
from soccer.reasoning.fake import DeterministicReasoner
from soccer.agent import PredictionAgent
from soccer.harness import run_scenario
from soccer.scenarios import load_scenario, SCENARIO_NAMES

@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_each_scenario_runs_end_to_end(name):
    scenario = load_scenario(name)
    agent = PredictionAgent(registry=scenario.registry,
                            reasoner=DeterministicReasoner())
    report = run_scenario(scenario, agent)
    assert report.n == len(scenario.matches) >= 1
    assert 0.0 <= report.accuracy <= 1.0

def test_unknown_scenario_raises():
    with pytest.raises(KeyError):
        load_scenario("does-not-exist")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scenarios.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write fixtures and loader**

Create `fixtures/wc-2026-final.json` (single match; teams France vs Brazil at a neutral venue). Use realistic-but-illustrative values:

```json
{
  "form": {
    "France": {"team": "France", "last_n": ["W", "W", "D", "W", "W"], "gf": 12,
               "ga": 3, "points": 13, "streak": "W2"},
    "Brazil": {"team": "Brazil", "last_n": ["W", "D", "W", "W", "L"], "gf": 10,
               "ga": 5, "points": 10, "streak": "L1"}
  },
  "injuries": {
    "France": {"team": "France", "out": [], "doubtful": [
      {"name": "Key Winger", "status": "doubtful", "reason": "hamstring"}]},
    "Brazil": {"team": "Brazil", "out": [
      {"name": "First Choice CB", "status": "out", "reason": "suspension"}],
      "doubtful": []}
  },
  "h2h": {"France|Brazil": {"home": "France", "away": "Brazil", "meetings": [
      {"date": "2006-07-01T00:00:00+00:00", "home": "France", "away": "Brazil",
       "home_goals": 1, "away_goals": 0}],
    "home_wins": 1, "draws": 0, "away_wins": 0}},
  "weather": {"metlife": {"venue_id": "metlife", "temp_c": 28.0, "wind_kph": 12.0,
                          "precip_mm": 0.0, "condition": "clear"}},
  "venue": {"metlife": {"venue_id": "metlife", "name": "MetLife Stadium",
                        "city": "East Rutherford", "surface": "grass",
                        "capacity": 82500, "altitude_m": 7,
                        "home_advantage_hint": 0.0}},
  "odds": {"wc-final": {"bookmaker": "consensus", "home": 2.4, "draw": 3.2,
                        "away": 3.0}},
  "results": {"wc-final": {"home_goals": 2, "away_goals": 1, "status": "finished"}}
}
```

Create `fixtures/ucl-2025-26.json` with at least three matches (`ucl-1`, `ucl-2`, `ucl-3`). Follow the exact same section/key structure as the WC file: top-level keys `form`, `injuries`, `h2h`, `weather`, `venue`, `odds`, `results`. For each match referenced by `scenarios.py` below, include the two teams under `form`/`injuries`, an `h2h` entry keyed `"<home>|<away>"`, the venue under `venue`, the venue's weather under `weather`, odds keyed by the match id, and a result keyed by the match id. Use these matches:

- `ucl-1`: Real Madrid (home) vs Manchester City (away), venue `bernabeu`, result 3-1
- `ucl-2`: Bayern Munich (home) vs Arsenal (away), venue `allianz`, result 1-1
- `ucl-3`: Inter (home) vs PSG (away), venue `giuseppe-meazza`, result 0-2

Each `form` entry needs keys `team, last_n (list of "W"/"D"/"L"), gf, ga, points, streak`. Each `odds` entry needs `bookmaker, home, draw, away` (decimal). Each `venue` entry needs `venue_id, name, city, surface, capacity, altitude_m, home_advantage_hint`. Each `weather` entry needs `venue_id, temp_c, wind_kph, precip_mm, condition`. Each `injuries` entry needs `team, out (list), doubtful (list)`. Each `h2h` entry needs `home, away, meetings (list), home_wins, draws, away_wins`. Each `results` entry needs `home_goals, away_goals, status`.

```python
# src/soccer/scenarios.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from soccer.models import MatchRef, MatchResult
from soccer.registry import build_fixture_registry
from soccer.harness import Scenario

_FIXTURE_DIR = Path(__file__).resolve().parent.parent.parent / "fixtures"

_KICK = datetime(2026, 7, 19, 19, 0, tzinfo=timezone.utc)


def _ref(match_id: str, competition: str, home: str, away: str, venue_id: str,
         season: str, kickoff: datetime) -> MatchRef:
    return MatchRef(id=match_id, competition=competition, home=home, away=away,
                    kickoff=kickoff, venue_id=venue_id, season=season)


_SCENARIO_MATCHES: dict[str, list[MatchRef]] = {
    "wc-2026-final": [
        _ref("wc-final", "FIFA World Cup", "France", "Brazil", "metlife", "2026",
             _KICK),
    ],
    "ucl-2025-26": [
        _ref("ucl-1", "UEFA Champions League", "Real Madrid", "Manchester City",
             "bernabeu", "2025-26", datetime(2026, 2, 18, 20, 0, tzinfo=timezone.utc)),
        _ref("ucl-2", "UEFA Champions League", "Bayern Munich", "Arsenal", "allianz",
             "2025-26", datetime(2026, 2, 25, 20, 0, tzinfo=timezone.utc)),
        _ref("ucl-3", "UEFA Champions League", "Inter", "PSG", "giuseppe-meazza",
             "2025-26", datetime(2026, 3, 4, 20, 0, tzinfo=timezone.utc)),
    ],
}

_SCENARIO_RESULTS: dict[str, dict[str, MatchResult]] = {
    "wc-2026-final": {
        "wc-final": MatchResult(match_id="wc-final", home_goals=2, away_goals=1,
                                status="finished", source="fixture"),
    },
    "ucl-2025-26": {
        "ucl-1": MatchResult(match_id="ucl-1", home_goals=3, away_goals=1,
                             status="finished", source="fixture"),
        "ucl-2": MatchResult(match_id="ucl-2", home_goals=1, away_goals=1,
                             status="finished", source="fixture"),
        "ucl-3": MatchResult(match_id="ucl-3", home_goals=0, away_goals=2,
                             status="finished", source="fixture"),
    },
}

SCENARIO_NAMES: tuple[str, ...] = ("ucl-2025-26", "wc-2026-final")


def load_scenario(name: str) -> Scenario:
    if name not in _SCENARIO_MATCHES:
        raise KeyError(f"unknown scenario: {name}")
    fixture_path = _FIXTURE_DIR / f"{name}.json"
    registry = build_fixture_registry(fixture_path)
    return Scenario(name=name, registry=registry,
                    matches=_SCENARIO_MATCHES[name],
                    results=_SCENARIO_RESULTS[name])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scenarios.py -v`
Expected: PASS. If a KeyError/ToolError fires, a fixture key does not match a `MatchRef` field (team name, `venue_id`, or match id) — align the JSON keys to the `MatchRef` values above.

- [ ] **Step 5: Commit**

```bash
git add fixtures/ src/soccer/scenarios.py tests/test_scenarios.py
git commit -m "feat: add UCL 2025/26 and WC 2026 final scenario fixtures"
```

---

### Task 18: Config (`config.py`)

**Files:**
- Create: `src/soccer/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from soccer.config import AppConfig

def test_from_env_defaults(monkeypatch):
    for var in ["SOCCER_DATA_DIR", "SOCCER_OLLAMA_HOST", "SOCCER_OLLAMA_MODEL",
                "SOCCER_OLLAMA_TIMEOUT", "SOCCER_PROVIDER_MODE", "SOCCER_REASONER"]:
        monkeypatch.delenv(var, raising=False)
    cfg = AppConfig.from_env()
    assert cfg.data_dir == Path("./data")
    assert cfg.ollama_model == "gemma4:12b-mlx"
    assert cfg.provider_mode == "fixture"
    assert cfg.reasoner == "fake"

def test_from_env_overrides(monkeypatch):
    monkeypatch.setenv("SOCCER_OLLAMA_MODEL", "other:7b")
    monkeypatch.setenv("SOCCER_REASONER", "ollama")
    monkeypatch.setenv("SOCCER_OLLAMA_TIMEOUT", "30")
    cfg = AppConfig.from_env()
    assert cfg.ollama_model == "other:7b"
    assert cfg.reasoner == "ollama"
    assert cfg.ollama_timeout == 30.0

def test_invalid_reasoner_rejected(monkeypatch):
    monkeypatch.setenv("SOCCER_REASONER", "bogus")
    import pytest
    with pytest.raises(ValueError):
        AppConfig.from_env()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_PROVIDER_MODES = {"fixture", "http"}
_REASONERS = {"fake", "ollama"}


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    ollama_host: str
    ollama_model: str
    ollama_timeout: float
    provider_mode: str
    reasoner: str

    @classmethod
    def from_env(cls) -> AppConfig:
        provider_mode = os.environ.get("SOCCER_PROVIDER_MODE", "fixture")
        reasoner = os.environ.get("SOCCER_REASONER", "fake")
        if provider_mode not in _PROVIDER_MODES:
            raise ValueError(f"SOCCER_PROVIDER_MODE must be one of {_PROVIDER_MODES}")
        if reasoner not in _REASONERS:
            raise ValueError(f"SOCCER_REASONER must be one of {_REASONERS}")
        return cls(
            data_dir=Path(os.environ.get("SOCCER_DATA_DIR", "./data")),
            ollama_host=os.environ.get("SOCCER_OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=os.environ.get("SOCCER_OLLAMA_MODEL", "gemma4:12b-mlx"),
            ollama_timeout=float(os.environ.get("SOCCER_OLLAMA_TIMEOUT", "60")),
            provider_mode=provider_mode,
            reasoner=reasoner,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/config.py tests/test_config.py
git commit -m "feat: add AppConfig env loader"
```

---

### Task 19: CLI (`cli.py`, `__main__.py`)

**Files:**
- Create: `src/soccer/cli.py`, `src/soccer/__main__.py`
- Test: `tests/test_cli.py`

The CLI wires everything: `eval` runs a scenario and prints metrics; `report` summarizes the JSONL log. `predict`/`settle` against a live match need a real registry, which in fixture mode means a scenario; to keep Phase 1 fully testable, `predict --match <id>` resolves the match from the loaded scenarios and uses that scenario's registry.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import pytest
from soccer.cli import main

def test_eval_prints_report(capsys):
    code = main(["eval", "--scenario", "wc-2026-final", "--reasoner", "fake"])
    out = capsys.readouterr().out
    assert code == 0
    assert "wc-2026-final" in out
    assert "accuracy" in out.lower()
    assert "edge_vs_market" in out or "edge vs market" in out.lower()

def test_eval_all_scenarios(capsys):
    code = main(["eval", "--scenario", "all", "--reasoner", "fake"])
    out = capsys.readouterr().out
    assert code == 0
    assert "ucl-2025-26" in out and "wc-2026-final" in out

def test_predict_then_report_roundtrip(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
    code = main(["predict", "--match", "wc-final", "--reasoner", "fake"])
    assert code == 0
    out = capsys.readouterr().out
    assert "France" in out and "Brazil" in out
    code = main(["report"])
    report_out = capsys.readouterr().out
    assert code == 0
    assert "wc-final" in report_out

def test_predict_unknown_match_errors(capsys):
    code = main(["predict", "--match", "nope", "--reasoner", "fake"])
    assert code == 1
    err = capsys.readouterr().err
    assert "nope" in err

def test_settle_after_predict(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
    main(["predict", "--match", "wc-final", "--reasoner", "fake"])
    capsys.readouterr()
    code = main(["settle", "--reasoner", "fake"])
    out = capsys.readouterr().out
    assert code == 0
    assert "1" in out  # one prediction settled
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Write minimal implementation**

```python
# src/soccer/cli.py
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from soccer.agent import PredictionAgent
from soccer.config import AppConfig
from soccer.harness import EvalReport, run_scenario
from soccer.models import MatchRef
from soccer.reasoning.base import Reasoner
from soccer.reasoning.fake import DeterministicReasoner
from soccer.reasoning.ollama import OllamaReasoner
from soccer.registry import ToolRegistry
from soccer.scenarios import SCENARIO_NAMES, load_scenario
from soccer.settle import settle
from soccer.store import PredictionStore


def _make_reasoner(name: str, config: AppConfig) -> Reasoner:
    if name == "ollama":
        return OllamaReasoner(host=config.ollama_host, model=config.ollama_model,
                              timeout=config.ollama_timeout)
    return DeterministicReasoner()


def _make_store(config: AppConfig) -> PredictionStore:
    base = config.data_dir
    return PredictionStore(predictions_path=base / "predictions.jsonl",
                           results_path=base / "results.jsonl",
                           evaluations_path=base / "evaluations.jsonl")


def _find_match(match_id: str) -> tuple[MatchRef, ToolRegistry]:
    for name in SCENARIO_NAMES:
        scenario = load_scenario(name)
        for ref in scenario.matches:
            if ref.id == match_id:
                return ref, scenario.registry
    raise KeyError(match_id)


def _print_report(report: EvalReport) -> None:
    print(f"== {report.scenario} ==")
    print(f"  n={report.n}  accuracy={report.accuracy:.3f}")
    print(f"  mean_brier={report.mean_brier:.4f}  "
          f"mean_log_loss={report.mean_log_loss:.4f}")
    print(f"  market mean_log_loss={report.market_baseline.mean_log_loss:.4f}")
    print(f"  edge_vs_market={report.edge_vs_market:+.4f} "
          f"({'better' if report.edge_vs_market < 0 else 'worse'} than market)")
    for s in report.per_match:
        flag = "HIT " if s.correct else "MISS"
        print(f"    [{flag}] {s.match_id}: pick={s.pick.value} "
              f"actual={s.actual.value} brier={s.brier:.3f}")


def _cmd_eval(args: argparse.Namespace, config: AppConfig) -> int:
    names = SCENARIO_NAMES if args.scenario == "all" else (args.scenario,)
    reasoner = _make_reasoner(args.reasoner, config)
    for name in names:
        scenario = load_scenario(name)
        agent = PredictionAgent(registry=scenario.registry, reasoner=reasoner)
        _print_report(run_scenario(scenario, agent))
    return 0


def _cmd_predict(args: argparse.Namespace, config: AppConfig) -> int:
    try:
        ref, registry = _find_match(args.match)
    except KeyError:
        print(f"unknown match: {args.match}", file=sys.stderr)
        return 1
    agent = PredictionAgent(registry=registry,
                            reasoner=_make_reasoner(args.reasoner, config))
    prediction = agent.predict(ref)
    _make_store(config).append_prediction(prediction)
    print(f"{ref.home} vs {ref.away}: pick={prediction.pick.value} "
          f"confidence={prediction.confidence:.2f}")
    print(f"  probs={ {k.value: round(v, 3) for k, v in prediction.probs.items()} }")
    print(f"  rationale: {prediction.rationale}")
    return 0


def _cmd_settle(args: argparse.Namespace, config: AppConfig) -> int:
    store = _make_store(config)
    reasoner = _make_reasoner(args.reasoner, config)
    settled = 0
    for prediction in store.pending():
        try:
            _, registry = _find_match(prediction.match_ref.id)
        except KeyError:
            continue
        settled += len(settle(store, registry, reasoner))
    print(f"settled {settled} prediction(s)")
    return 0


def _cmd_report(args: argparse.Namespace, config: AppConfig) -> int:
    store = _make_store(config)
    predictions = store.load_predictions()
    evaluations = {e.prediction_id: e for e in store.load_evaluations()}
    if not predictions:
        print("no predictions logged")
        return 0
    correct = sum(1 for e in evaluations.values() if e.correct)
    print(f"predictions={len(predictions)} evaluated={len(evaluations)} "
          f"correct={correct}")
    for p in predictions:
        ev = evaluations.get(p.id)
        status = "pending" if ev is None else ("HIT" if ev.correct else "MISS")
        print(f"  {p.match_ref.id}: pick={p.pick.value} "
              f"conf={p.confidence:.2f} [{status}]")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="soccer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_predict = sub.add_parser("predict", help="predict a known match")
    p_predict.add_argument("--match", required=True)
    p_predict.add_argument("--reasoner", choices=["fake", "ollama"], default=None)

    p_settle = sub.add_parser("settle", help="settle finished predictions")
    p_settle.add_argument("--reasoner", choices=["fake", "ollama"], default=None)

    p_eval = sub.add_parser("eval", help="run an eval scenario")
    p_eval.add_argument("--scenario", required=True,
                        choices=[*SCENARIO_NAMES, "all"])
    p_eval.add_argument("--reasoner", choices=["fake", "ollama"], default=None)

    sub.add_parser("report", help="summarize logged predictions")

    args = parser.parse_args(argv)
    config = AppConfig.from_env()
    # CLI flag overrides env for reasoner selection where present.
    if getattr(args, "reasoner", None) is None:
        args.reasoner = config.reasoner

    handlers = {"predict": _cmd_predict, "settle": _cmd_settle,
                "eval": _cmd_eval, "report": _cmd_report}
    return handlers[args.command](args, config)
```

```python
# src/soccer/__main__.py
import sys

from soccer.cli import main

if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/soccer/cli.py src/soccer/__main__.py tests/test_cli.py
git commit -m "feat: add CLI with predict/settle/eval/report commands"
```

---

### Task 20: README + full quality gate

**Files:**
- Create: `README.md`, `docs/architecture.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# Soccer Prediction Agent

A multi-tool agent that builds a typed dossier for an upcoming match (form, injuries,
head-to-head, weather, venue, bookmaker odds), reasons with a local Ollama model behind
a swappable interface, logs a 1X2 prediction with rationale and confidence, settles
results on demand, and scores itself against the bookmaker via an offline eval harness.

## Setup

    python -m venv .venv && source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

## Commands

    soccer eval --scenario ucl-2025-26 --reasoner fake
    soccer eval --scenario wc-2026-final --reasoner ollama
    soccer predict --match wc-final
    soccer settle
    soccer report

Configuration is read from environment variables (see `.env.example`): data directory,
Ollama host/model/timeout, provider mode (`fixture|http`), and reasoner (`fake|ollama`).
The `fake` reasoner and `fixture` providers are fully offline and require no network or
Ollama; they are the default and what the test suite and CI use.

## Quality gate

    make check   # ruff lint + mypy + pytest with coverage

## Architecture

See `docs/architecture.md` and the Phase 1 design at
`docs/superpowers/specs/2026-06-08-soccer-prediction-agent-phase1-design.md`.
```

- [ ] **Step 2: Write `docs/architecture.md`**

```markdown
# Architecture

`PredictionAgent` runs a deterministic pipeline over a `ToolRegistry` of single-method
provider Protocols (fixture + HTTP implementations), assembling a `MatchDossier`
(`dossier.build_dossier`, which degrades gracefully on tool failure), then calls a
`Reasoner` once (`reasoning/`: `DeterministicReasoner` or `OllamaReasoner`). Predictions
persist as JSONL via `PredictionStore`. `settle` matches finished results to pending
predictions, scores them (`evaluation.score`), and stores a self-critique. The offline
`harness` runs `Scenario` fixtures with known results and reports accuracy, Brier,
log-loss, calibration, and edge vs the bookmaker baseline. The CLI (`soccer`) exposes
`predict`, `settle`, `eval`, and `report`.

Swap points (all at the registry/config boundary): provider mode (`fixture|http`) and
reasoner (`fake|ollama`). The agent depends only on `ToolRegistry` and the `Reasoner`
protocol. The registry's `as_tools()` view is the seam for a future model-driven
tool-selection loop.
```

- [ ] **Step 3: Run the full quality gate**

Run:
```bash
ruff format .
ruff check .
mypy src tests
pytest --cov=soccer --cov-report=term-missing
```
Expected: format clean; lint clean; mypy clean (strict); all tests pass; coverage reported. Fix any issues before committing.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/architecture.md
git commit -m "docs: add README and architecture overview"
```

---

## Self-Review

**Spec coverage:**
- Pull recent form, injuries, H2H, weather, venue, odds → Tasks 5 (providers), 8 (dossier). ✓
- Reason about matchup → Tasks 9–11 (fake + ollama reasoners). ✓
- Output prediction with rationale + confidence (1X2) → Tasks 3 (model), 12 (agent). ✓
- Log prediction → Task 13 (store). ✓
- Wait for result + self-evaluate (on-demand `settle`) → Tasks 14–15. ✓
- Eval harness with bookmaker baseline → Task 16. ✓
- Pluggable adapters (fixture + HTTP) → Tasks 5, 6, 7. ✓
- Local Ollama behind swappable interface → Task 11. ✓
- Two use cases (UCL 2025/26, WC 2026 final) → Task 17. ✓
- CLI surface (predict/settle/eval/report) → Task 19. ✓
- Config at boundary, no import-time env reads → Task 18. ✓
- Approach C tool registry + uniform Tool view for future loop → Tasks 4, 7. ✓

**Placeholder scan:** No "TBD"/"implement later"; HTTP stubs raise an explicit
`NotImplementedError` by design (Task 6). The UCL fixture JSON is specified by structure +
exact match list rather than full literal — acceptable because the structure is fully shown
in the WC fixture and the required keys are enumerated.

**Type consistency:** `ReasonResult` (probs/confidence/rationale), `Reasoner.name`,
`make_prediction_id`, `score(prediction, result, self_critique, evaluated_at)`,
`build_dossier(match, registry)`, `build_fixture_registry(path)`, `run_scenario(scenario,
agent)`, and serialization helper names are used identically across tasks. `Reasoner`
protocol carries a `name` attribute, satisfied by both reasoners.

**Dependencies:** runtime stdlib-only (HTTP via `urllib` behind injected `post_json`),
matching AGENTS.md's minimal-dependency rule.
```
