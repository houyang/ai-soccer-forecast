import json
from datetime import UTC, datetime
from pathlib import Path

from soccer.models import MatchRef, Outcome, Prediction
from soccer.reasoning.fake import DeterministicReasoner
from soccer.registry import ToolRegistry, build_fixture_registry
from soccer.settle import settle
from soccer.store import PredictionStore

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=UTC)
NOW = datetime(2026, 4, 2, 9, 0, tzinfo=UTC)


def _ref(mid: str) -> MatchRef:
    return MatchRef(
        id=mid,
        competition="UCL",
        home="A",
        away="B",
        kickoff=KICK,
        venue_id="v1",
        season="2025-26",
    )


def _pred(pid: str, mid: str) -> Prediction:
    return Prediction(
        id=pid,
        match_ref=_ref(mid),
        created_at=KICK,
        probs={Outcome.HOME: 0.6, Outcome.DRAW: 0.25, Outcome.AWAY: 0.15},
        pick=Outcome.HOME,
        confidence=0.6,
        rationale="r",
        market_probs={Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2},
        dossier_digest="d",
        reasoner_name="fake",
    )


def _registry(tmp_path: Path) -> ToolRegistry:
    payload = {
        "form": {},
        "injuries": {},
        "h2h": {},
        "weather": {},
        "venue": {},
        "odds": {},
        "results": {"m1": {"home_goals": 2, "away_goals": 0, "status": "finished"}},
    }
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    return build_fixture_registry(path)


def _store(tmp_path: Path) -> PredictionStore:
    return PredictionStore(
        predictions_path=tmp_path / "p.jsonl",
        results_path=tmp_path / "r.jsonl",
        evaluations_path=tmp_path / "e.jsonl",
    )


def test_settle_scores_finished_and_skips_unfinished(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_prediction(_pred("p1", "m1"))  # finished in fixtures
    store.append_prediction(_pred("p2", "m2"))  # no result → skipped
    evals = settle(store, _registry(tmp_path), DeterministicReasoner(), clock=lambda: NOW)
    assert [e.prediction_id for e in evals] == ["p1"]
    assert evals[0].correct is True
    assert evals[0].beat_market is True
    assert store.pending()[0].id == "p2"


def test_settle_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.append_prediction(_pred("p1", "m1"))
    settle(store, _registry(tmp_path), DeterministicReasoner(), clock=lambda: NOW)
    second = settle(store, _registry(tmp_path), DeterministicReasoner(), clock=lambda: NOW)
    assert second == []  # already evaluated
    assert len(store.load_evaluations()) == 1
