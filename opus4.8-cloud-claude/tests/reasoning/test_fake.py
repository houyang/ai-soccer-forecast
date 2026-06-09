import json
from datetime import UTC, datetime
from pathlib import Path

from soccer.dossier import build_dossier
from soccer.models import MatchDossier, MatchRef, MatchResult, Outcome, Prediction
from soccer.reasoning.fake import DeterministicReasoner
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


def _payload(with_odds: bool = True) -> dict[str, object]:
    return {
        "form": {
            "A": {
                "team": "A",
                "last_n": ["W", "W"],
                "gf": 6,
                "ga": 1,
                "points": 18,
                "streak": "W2",
            },
            "B": {
                "team": "B",
                "last_n": ["L", "D"],
                "gf": 2,
                "ga": 5,
                "points": 4,
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
                "home_wins": 2,
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
        "odds": (
            {"m1": {"bookmaker": "b", "home": 1.7, "draw": 3.8, "away": 5.0}} if with_odds else {}
        ),
        "results": {},
    }


def _dossier(tmp_path: Path, with_odds: bool = True) -> MatchDossier:
    path = tmp_path / "f.json"
    path.write_text(json.dumps(_payload(with_odds)))
    return build_dossier(REF, build_fixture_registry(path))


def test_fake_reasoner_is_deterministic(tmp_path: Path) -> None:
    r = DeterministicReasoner()
    d = _dossier(tmp_path)
    a = r.predict(d)
    b = r.predict(d)
    assert a.probs == b.probs and a.confidence == b.confidence


def test_fake_reasoner_probs_valid_and_favours_strong_home(tmp_path: Path) -> None:
    res = DeterministicReasoner().predict(_dossier(tmp_path))
    assert abs(sum(res.probs.values()) - 1.0) < 1e-6
    assert res.probs[Outcome.HOME] > res.probs[Outcome.AWAY]
    assert res.rationale


def test_fake_reasoner_without_odds_still_valid(tmp_path: Path) -> None:
    res = DeterministicReasoner().predict(_dossier(tmp_path, with_odds=False))
    assert abs(sum(res.probs.values()) - 1.0) < 1e-6


def test_fake_self_evaluate_returns_text(tmp_path: Path) -> None:
    r = DeterministicReasoner()
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
        reasoner_name="fake",
    )
    result = MatchResult(
        match_id="m1", home_goals=0, away_goals=1, status="finished", source="fixture"
    )
    assert "AWAY" in r.self_evaluate(pred, result)
