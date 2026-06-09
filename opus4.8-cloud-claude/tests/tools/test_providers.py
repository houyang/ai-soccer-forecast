# tests/tools/test_providers.py
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from soccer.models import MatchRef, Outcome
from soccer.tools.base import ToolError
from soccer.tools.fixtures import FixtureStore
from soccer.tools.form import FixtureFormProvider
from soccer.tools.head_to_head import FixtureH2HProvider
from soccer.tools.injuries import FixtureInjuryProvider
from soccer.tools.odds import FixtureOddsProvider
from soccer.tools.results import FixtureResultProvider
from soccer.tools.venue import FixtureVenueProvider
from soccer.tools.weather import FixtureWeatherProvider

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=UTC)
REF = MatchRef(
    id="m1",
    competition="UCL",
    home="A",
    away="B",
    kickoff=KICK,
    venue_id="v1",
    season="2025-26",
)


@pytest.fixture
def store(tmp_path: Path) -> FixtureStore:
    payload = {
        "form": {
            "A": {
                "team": "A",
                "last_n": ["W", "W", "D"],
                "gf": 7,
                "ga": 2,
                "points": 21,
                "streak": "W2",
            }
        },
        "injuries": {
            "A": {
                "team": "A",
                "out": [{"name": "P1", "status": "out", "reason": "knee"}],
                "doubtful": [],
            }
        },
        "h2h": {
            "A|B": {
                "home": "A",
                "away": "B",
                "meetings": [
                    {
                        "date": "2025-01-01T00:00:00+00:00",
                        "home": "A",
                        "away": "B",
                        "home_goals": 2,
                        "away_goals": 1,
                    }
                ],
                "home_wins": 1,
                "draws": 0,
                "away_wins": 0,
            }
        },
        "weather": {
            "v1": {
                "venue_id": "v1",
                "temp_c": 12.0,
                "wind_kph": 9.0,
                "precip_mm": 0.0,
                "condition": "clear",
            }
        },
        "venue": {
            "v1": {
                "venue_id": "v1",
                "name": "Stad",
                "city": "X",
                "surface": "grass",
                "capacity": 60000,
                "altitude_m": 50,
                "home_advantage_hint": 0.1,
            }
        },
        "odds": {"m1": {"bookmaker": "b", "home": 2.0, "draw": 3.5, "away": 3.8}},
        "results": {"m1": {"home_goals": 2, "away_goals": 1, "status": "finished"}},
    }
    path = tmp_path / "ucl.json"
    path.write_text(json.dumps(payload))
    return FixtureStore(path)


def test_form_provider(store: FixtureStore) -> None:
    form = FixtureFormProvider(store).get_form("A", KICK)
    assert form.points == 21 and form.last_n[0].value == "W"


def test_injury_provider(store: FixtureStore) -> None:
    rep = FixtureInjuryProvider(store).get_injuries("A", KICK)
    assert rep.out[0].name == "P1"


def test_h2h_provider(store: FixtureStore) -> None:
    rec = FixtureH2HProvider(store).get_h2h("A", "B")
    assert rec.home_wins == 1 and rec.meetings[0].home_goals == 2


def test_weather_provider(store: FixtureStore) -> None:
    w = FixtureWeatherProvider(store).get_weather("v1", KICK)
    assert w.condition == "clear"


def test_venue_provider(store: FixtureStore) -> None:
    v = FixtureVenueProvider(store).get_venue("v1")
    assert v.capacity == 60000


def test_odds_provider(store: FixtureStore) -> None:
    o = FixtureOddsProvider(store).get_odds(REF)
    assert o.implied_probs[Outcome.HOME] > o.implied_probs[Outcome.AWAY]


def test_result_provider_returns_none_when_absent(store: FixtureStore) -> None:
    none_ref = MatchRef(
        id="zzz",
        competition="UCL",
        home="A",
        away="B",
        kickoff=KICK,
        venue_id="v1",
        season="2025-26",
    )
    assert FixtureResultProvider(store).get_result(none_ref) is None


def test_result_provider_returns_result(store: FixtureStore) -> None:
    r = FixtureResultProvider(store).get_result(REF)
    assert r is not None and r.outcome is Outcome.HOME


def test_result_provider_raises_when_results_section_absent(tmp_path: Path) -> None:
    path = tmp_path / "no_results.json"
    path.write_text(json.dumps({"form": {"A": {"team": "A"}}}))
    store = FixtureStore(path)
    with pytest.raises(ToolError):
        FixtureResultProvider(store).get_result(REF)
