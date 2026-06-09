from datetime import UTC, datetime
from pathlib import Path

import pytest

from soccer.fixture_tools import build_fixture_agent, default_catalog
from soccer.models import MatchResult
from soccer.storage import JsonlPredictionLog


def test_jsonl_log_round_trips_prediction_record(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    log = JsonlPredictionLog(path)
    catalog = default_catalog()
    expected = build_fixture_agent(log).predict(catalog.requests["ucl-final-2026"])

    actual = JsonlPredictionLog(path).list_records()[0]

    assert actual.request == expected.request
    assert actual.evidence.home_form == expected.evidence.home_form
    assert actual.prediction.outcome == expected.prediction.outcome


def test_attaching_unknown_result_raises(tmp_path: Path) -> None:
    log = JsonlPredictionLog(tmp_path / "predictions.jsonl")

    with pytest.raises(LookupError, match="No prediction found"):
        log.attach_result(
            MatchResult(
                match_id="missing",
                home_score=0,
                away_score=0,
                completed_at=datetime.now(UTC),
            )
        )
