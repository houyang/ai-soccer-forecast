from datetime import UTC, datetime
from pathlib import Path

from soccer.fixture_tools import build_fixture_agent, default_catalog
from soccer.models import MatchResult
from soccer.storage import JsonlPredictionLog


def test_world_cup_competition_batch_predicts_and_logs(tmp_path: Path) -> None:
    log = JsonlPredictionLog(tmp_path / "predictions.jsonl")
    catalog = default_catalog()
    requests = catalog.requests_for_competition("world-cup-2026")

    records = build_fixture_agent(log).predict_many(requests)

    assert len(records) == 3
    assert {record.request.match_id for record in records} == {
        "wc-2026-match-001",
        "wc-2026-match-010",
        "wc-2026-final",
    }
    assert len(log.list_records()) == 3


def test_agent_records_results_in_batches(tmp_path: Path) -> None:
    log = JsonlPredictionLog(tmp_path / "predictions.jsonl")
    catalog = default_catalog()
    agent = build_fixture_agent(log)
    agent.predict_many(catalog.requests_for_competition("world-cup-2026"))

    records = agent.record_results(
        (
            MatchResult(
                match_id="wc-2026-match-001",
                home_score=2,
                away_score=0,
                completed_at=datetime(2026, 6, 12, tzinfo=UTC),
            ),
            MatchResult(
                match_id="wc-2026-match-010",
                home_score=0,
                away_score=3,
                completed_at=datetime(2026, 6, 15, tzinfo=UTC),
            ),
        )
    )

    assert len(records) == 2
    settled = [record for record in log.list_records() if record.result is not None]
    assert len(settled) == 2
