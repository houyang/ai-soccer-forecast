import json
from datetime import UTC, datetime
from pathlib import Path

from soccer.dossier import build_dossier, dossier_digest
from soccer.models import MatchRef
from soccer.registry import build_fixture_registry

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


def _full_payload() -> dict[str, dict[str, object]]:
    return {
        "form": {
            "A": {
                "team": "A",
                "last_n": ["W"],
                "gf": 5,
                "ga": 1,
                "points": 12,
                "streak": "W1",
            },
            "B": {
                "team": "B",
                "last_n": ["L"],
                "gf": 2,
                "ga": 4,
                "points": 5,
                "streak": "L1",
            },
        },
        "injuries": {
            "A": {"team": "A", "out": [], "doubtful": []},
            "B": {"team": "B", "out": [], "doubtful": []},
        },
        "h2h": {
            "A|B": {
                "home": "A",
                "away": "B",
                "meetings": [],
                "home_wins": 0,
                "draws": 0,
                "away_wins": 0,
            }
        },
        "weather": {
            "v1": {
                "venue_id": "v1",
                "temp_c": 10.0,
                "wind_kph": 5.0,
                "precip_mm": 0.0,
                "condition": "clear",
            }
        },
        "venue": {
            "v1": {
                "venue_id": "v1",
                "name": "S",
                "city": "C",
                "surface": "grass",
                "capacity": 50000,
                "altitude_m": 10,
                "home_advantage_hint": 0.1,
            }
        },
        "odds": {"m1": {"bookmaker": "b", "home": 1.8, "draw": 3.6, "away": 4.5}},
        "results": {},
    }


def test_full_dossier_has_no_missing(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps(_full_payload()))
    d = build_dossier(REF, build_fixture_registry(path))
    assert d.missing == ()
    home, away = d.form["home"], d.form["away"]
    assert home is not None and away is not None
    assert home.team == "A" and away.team == "B"
    assert d.odds is not None


def test_missing_provider_recorded_not_fatal(tmp_path: Path) -> None:
    payload = _full_payload()
    del payload["odds"]["m1"]  # odds lookup will raise ToolError
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    d = build_dossier(REF, build_fixture_registry(path))
    assert d.odds is None
    assert "odds" in d.missing


def test_dossier_digest_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps(_full_payload()))
    reg = build_fixture_registry(path)
    d1 = build_dossier(REF, reg)
    d2 = build_dossier(REF, reg)
    assert dossier_digest(d1) == dossier_digest(d2)
