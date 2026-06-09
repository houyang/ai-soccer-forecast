from datetime import UTC, datetime
from pathlib import Path

from soccer.fixture_tools import build_fixture_agent, default_catalog
from soccer.models import MatchResult, Outcome
from soccer.storage import JsonlPredictionLog


def test_agent_gathers_evidence_logs_prediction_and_records_result(tmp_path: Path) -> None:
    log = JsonlPredictionLog(tmp_path / "predictions.jsonl")
    catalog = default_catalog()
    agent = build_fixture_agent(log)

    record = agent.predict(catalog.requests["ucl-final-2026"])

    assert record.prediction.match_id == "ucl-final-2026"
    assert record.prediction.outcome in set(Outcome)
    assert 0.0 < record.prediction.confidence < 1.0
    assert "European Club A form" in record.prediction.rationale
    assert len(log.list_records()) == 1

    updated = agent.record_result(
        MatchResult(
            match_id="ucl-final-2026",
            home_score=2,
            away_score=1,
            completed_at=datetime(2026, 5, 31, tzinfo=UTC),
        )
    )

    assert updated.result is not None
    assert updated.result.outcome == Outcome.HOME_WIN
    assert log.list_records()[0].result is not None
