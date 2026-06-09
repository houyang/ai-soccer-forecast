import json
from datetime import UTC, datetime
from pathlib import Path

from soccer.agent import PredictionAgent
from soccer.models import MatchRef, Outcome
from soccer.reasoning.fake import DeterministicReasoner
from soccer.registry import ToolRegistry, build_fixture_registry

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=UTC)
NOW = datetime(2026, 3, 30, 12, 0, tzinfo=UTC)
REF = MatchRef(
    id="m1",
    competition="UCL",
    home="A",
    away="B",
    kickoff=KICK,
    venue_id="v1",
    season="2025-26",
)


def _registry(tmp_path: Path) -> ToolRegistry:
    payload = {
        "form": {
            "A": {
                "team": "A",
                "last_n": ["W", "W"],
                "gf": 6,
                "ga": 1,
                "points": 18,
                "streak": "W2",
            },
            "B": {"team": "B", "last_n": ["L"], "gf": 1, "ga": 4, "points": 3, "streak": "L1"},
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
        "odds": {"m1": {"bookmaker": "b", "home": 1.7, "draw": 3.8, "away": 5.0}},
        "results": {},
    }
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    return build_fixture_registry(path)


def test_agent_produces_valid_prediction(tmp_path: Path) -> None:
    agent = PredictionAgent(
        registry=_registry(tmp_path),
        reasoner=DeterministicReasoner(),
        clock=lambda: NOW,
    )
    pred = agent.predict(REF)
    assert pred.pick in Outcome
    assert abs(sum(pred.probs.values()) - 1.0) < 1e-6
    assert pred.reasoner_name == "fake"
    assert pred.market_probs is not None  # odds present
    assert pred.created_at == NOW
    assert pred.id  # stable id assigned


def test_agent_id_is_deterministic_for_same_clock(tmp_path: Path) -> None:
    agent = PredictionAgent(
        registry=_registry(tmp_path),
        reasoner=DeterministicReasoner(),
        clock=lambda: NOW,
    )
    assert agent.predict(REF).id == agent.predict(REF).id
