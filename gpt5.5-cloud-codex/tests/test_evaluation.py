from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from soccer.evaluation import EvaluationHarness
from soccer.fixture_tools import build_fixture_agent, default_catalog
from soccer.models import MatchResult
from soccer.storage import JsonlPredictionLog


def test_evaluation_scores_settled_predictions(tmp_path: Path) -> None:
    log = JsonlPredictionLog(tmp_path / "predictions.jsonl")
    catalog = default_catalog()
    agent = build_fixture_agent(log)
    record = agent.predict(catalog.requests["world-cup-final-2026"])
    log.attach_result(
        MatchResult(
            match_id=record.request.match_id,
            home_score=1,
            away_score=0,
            completed_at=datetime(2026, 7, 20, tzinfo=UTC),
        )
    )

    summary = EvaluationHarness(log).evaluate()

    assert summary.settled_count == 1
    assert 0.0 <= summary.accuracy <= 1.0
    assert summary.average_confidence == record.prediction.confidence
    assert summary.brier_score >= 0.0


def test_evaluation_returns_zero_summary_without_results(tmp_path: Path) -> None:
    log = JsonlPredictionLog(tmp_path / "predictions.jsonl")
    catalog = default_catalog()
    record = build_fixture_agent(log).predict(catalog.requests["ucl-final-2026"])
    assert replace(record, result=None).result is None

    summary = EvaluationHarness(log).evaluate()

    assert summary.settled_count == 0
    assert summary.accuracy == 0.0
