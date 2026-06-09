import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from soccer.dossier import build_dossier
from soccer.models import MatchDossier, MatchRef, MatchResult, Outcome, Prediction
from soccer.reasoning.base import ReasonerError
from soccer.reasoning.ollama import OllamaReasoner
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


def test_ollama_predict_parses_transport_response(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        captured["url"] = url
        captured["model"] = payload["model"]
        content = (
            '{"home": 0.6, "draw": 0.25, "away": 0.15, '
            '"confidence": 0.6, "rationale": "home strong"}'
        )
        return {"message": {"content": content}}

    r = OllamaReasoner(
        host="http://localhost:11434",
        model="gemma4:12b-mlx",
        timeout=5,
        post_json=fake_post,
    )
    res = r.predict(_dossier(tmp_path))
    assert res.probs[Outcome.HOME] == 0.6
    assert captured["model"] == "gemma4:12b-mlx"
    assert captured["url"].endswith("/api/chat")


def test_ollama_predict_raises_on_bad_json(tmp_path: Path) -> None:
    def fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {"message": {"content": "definitely not json"}}

    r = OllamaReasoner(host="http://localhost:11434", model="m", timeout=5, post_json=fake_post)
    with pytest.raises(ReasonerError):
        r.predict(_dossier(tmp_path))


def test_ollama_predict_raises_on_malformed_envelope(tmp_path: Path) -> None:
    def fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {"unexpected": True}

    r = OllamaReasoner(host="http://localhost:11434", model="m", timeout=5, post_json=fake_post)
    with pytest.raises(ReasonerError):
        r.predict(_dossier(tmp_path))


def test_ollama_self_evaluate(tmp_path: Path) -> None:
    def fake_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return {"message": {"content": "I overweighted home form."}}

    r = OllamaReasoner(host="http://localhost:11434", model="m", timeout=5, post_json=fake_post)
    pred = Prediction(
        id="x",
        match_ref=REF,
        created_at=KICK,
        probs={Outcome.HOME: 0.6, Outcome.DRAW: 0.25, Outcome.AWAY: 0.15},
        pick=Outcome.HOME,
        confidence=0.6,
        rationale="r",
        market_probs=None,
        dossier_digest="d",
        reasoner_name="ollama",
    )
    result = MatchResult(
        match_id="m1", home_goals=0, away_goals=2, status="finished", source="fixture"
    )
    assert "overweighted" in r.self_evaluate(pred, result)
