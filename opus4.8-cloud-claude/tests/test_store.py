from datetime import UTC, datetime
from pathlib import Path

from soccer.models import Evaluation, MatchRef, MatchResult, Outcome, Prediction
from soccer.store import PredictionStore

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


def _pred(pid: str = "p1", mid: str = "m1") -> Prediction:
    ref = MatchRef(
        id=mid,
        competition="UCL",
        home="A",
        away="B",
        kickoff=KICK,
        venue_id="v1",
        season="2025-26",
    )
    return Prediction(
        id=pid,
        match_ref=ref,
        created_at=KICK,
        probs={Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2},
        pick=Outcome.HOME,
        confidence=0.5,
        rationale="r",
        market_probs=None,
        dossier_digest="d",
        reasoner_name="fake",
    )


def _store(tmp_path: Path) -> PredictionStore:
    return PredictionStore(
        predictions_path=tmp_path / "p.jsonl",
        results_path=tmp_path / "r.jsonl",
        evaluations_path=tmp_path / "e.jsonl",
    )


def test_prediction_round_trip(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.append_prediction(_pred())
    loaded = s.load_predictions()
    assert len(loaded) == 1 and loaded[0] == _pred()


def test_pending_excludes_evaluated(tmp_path: Path) -> None:
    s = _store(tmp_path)
    s.append_prediction(_pred("p1", "m1"))
    s.append_prediction(_pred("p2", "m2"))
    result = MatchResult(
        match_id="m1", home_goals=1, away_goals=0, status="finished", source="fixture"
    )
    s.append_result(result)
    s.append_evaluation(
        Evaluation(
            prediction_id="p1",
            result=result,
            correct=True,
            brier=0.1,
            log_loss=0.2,
            beat_market=True,
            self_critique="ok",
            evaluated_at=KICK,
        )
    )
    pending = s.pending()
    assert [p.id for p in pending] == ["p2"]


def test_load_empty_returns_empty(tmp_path: Path) -> None:
    assert _store(tmp_path).load_predictions() == []
