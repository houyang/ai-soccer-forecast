import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from soccer.dossier import build_dossier
from soccer.models import MatchDossier, MatchRef, Outcome
from soccer.reasoning.base import ReasonerError
from soccer.reasoning.prompt import parse_reason_json, render_prompt
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


def _dossier(tmp_path: Path) -> MatchDossier:
    payload = {
        "form": {
            "A": {"team": "A", "last_n": ["W"], "gf": 3, "ga": 0, "points": 9, "streak": "W1"},
            "B": {"team": "B", "last_n": ["L"], "gf": 0, "ga": 3, "points": 1, "streak": "L1"},
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
                "home_wins": 1,
                "draws": 0,
                "away_wins": 0,
            }
        },
        "weather": {
            "v1": {
                "venue_id": "v1",
                "temp_c": 9.0,
                "wind_kph": 4.0,
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
                "capacity": 1000,
                "altitude_m": 0,
                "home_advantage_hint": 0.1,
            }
        },
        "odds": {"m1": {"bookmaker": "b", "home": 1.8, "draw": 3.5, "away": 4.0}},
        "results": {},
    }
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    return build_dossier(REF, build_fixture_registry(path))


def test_render_prompt_mentions_teams_and_json(tmp_path: Path) -> None:
    text = render_prompt(_dossier(tmp_path))
    assert "A" in text and "B" in text
    assert "JSON" in text or "json" in text


def test_parse_valid_json() -> None:
    raw = '{"home": 0.5, "draw": 0.3, "away": 0.2, "confidence": 0.55, "rationale": "x"}'
    res = parse_reason_json(raw)
    assert res.probs[Outcome.HOME] == 0.5 and res.confidence == 0.55


def test_parse_renormalises_probs() -> None:
    raw = '{"home": 2, "draw": 1, "away": 1, "confidence": 0.5, "rationale": "x"}'
    res = parse_reason_json(raw)
    assert abs(sum(res.probs.values()) - 1.0) < 1e-6


def test_parse_rejects_garbage() -> None:
    with pytest.raises(ReasonerError):
        parse_reason_json("not json at all")


def test_parse_rejects_missing_keys() -> None:
    with pytest.raises(ReasonerError):
        parse_reason_json('{"home": 0.5}')
