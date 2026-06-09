"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from soccer.evaluation import EvaluationHarness
from soccer.fixture_tools import FixtureCatalog, build_fixture_agent, default_catalog
from soccer.live_world_cup import (
    DEFAULT_WORLD_CUP_SCHEDULE_URL,
    WORLD_CUP_COMPETITION_ID,
    load_world_cup_catalog,
)
from soccer.models import MatchRequest, MatchResult, PredictionRecord
from soccer.storage import JsonlPredictionLog


def main() -> None:
    catalog = default_catalog()
    parser = argparse.ArgumentParser(prog="soccer-forecast")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Run a fixture-backed prediction")
    predict.add_argument("match_id", choices=sorted(catalog.requests))
    predict.add_argument("--log", type=Path, default=Path("predictions.jsonl"))

    list_matches = subparsers.add_parser("list-matches", help="List fixture-backed matches")
    list_matches.add_argument("competition_id", choices=sorted(catalog.competitions))
    list_matches.add_argument(
        "--live-world-cup",
        action="store_true",
        help="Fetch live World Cup catalog data instead of local fixtures",
    )
    list_matches.add_argument("--source-url", default=DEFAULT_WORLD_CUP_SCHEDULE_URL)

    predict_competition = subparsers.add_parser(
        "predict-competition",
        help="Run fixture-backed predictions for a competition",
    )
    predict_competition.add_argument("competition_id", choices=sorted(catalog.competitions))
    predict_competition.add_argument("--log", type=Path, default=Path("predictions.jsonl"))
    predict_competition.add_argument(
        "--live-world-cup",
        action="store_true",
        help="Fetch live World Cup catalog data instead of local fixtures",
    )
    predict_competition.add_argument("--source-url", default=DEFAULT_WORLD_CUP_SCHEDULE_URL)

    result = subparsers.add_parser("record-result", help="Attach a final score")
    result.add_argument("match_id")
    result.add_argument("home_score", type=int)
    result.add_argument("away_score", type=int)
    result.add_argument("--log", type=Path, default=Path("predictions.jsonl"))

    results = subparsers.add_parser("record-results", help="Attach final scores from JSON")
    results.add_argument("results_file", type=Path)
    results.add_argument("--log", type=Path, default=Path("predictions.jsonl"))

    evaluate = subparsers.add_parser("evaluate", help="Evaluate settled predictions")
    evaluate.add_argument("log", type=Path)

    args = parser.parse_args()
    if args.command == "predict":
        log = JsonlPredictionLog(args.log)
        record = build_fixture_agent(log).predict(catalog.requests[args.match_id])
        _print_prediction(record)
    elif args.command == "list-matches":
        active_catalog = _catalog_for_args(
            catalog,
            args.competition_id,
            args.live_world_cup,
            args.source_url,
        )
        for request in active_catalog.requests_for_competition(args.competition_id):
            print(_format_match(request))
    elif args.command == "predict-competition":
        active_catalog = _catalog_for_args(
            catalog,
            args.competition_id,
            args.live_world_cup,
            args.source_url,
        )
        log = JsonlPredictionLog(args.log)
        records = build_fixture_agent(log, active_catalog).predict_many(
            active_catalog.requests_for_competition(args.competition_id)
        )
        for record in records:
            print(_format_prediction_summary(record))
    elif args.command == "record-result":
        log = JsonlPredictionLog(args.log)
        record = build_fixture_agent(log).record_result(
            MatchResult(
                match_id=args.match_id,
                home_score=args.home_score,
                away_score=args.away_score,
                completed_at=datetime.now(UTC),
            )
        )
        if record.result is None:
            raise RuntimeError("Result was not attached to the prediction record")
        print(
            f"Recorded {record.result.home_score}-{record.result.away_score} "
            f"for {record.request.match_id}"
        )
    elif args.command == "record-results":
        log = JsonlPredictionLog(args.log)
        records = build_fixture_agent(log).record_results(_load_results(args.results_file))
        print(f"Recorded {len(records)} results")
    elif args.command == "evaluate":
        summary = EvaluationHarness(JsonlPredictionLog(args.log)).evaluate()
        print(f"Settled: {summary.settled_count}")
        print(f"Accuracy: {summary.accuracy:.3f}")
        print(f"Average confidence: {summary.average_confidence:.3f}")
        print(f"Brier score: {summary.brier_score:.3f}")


def _format_match(request: MatchRequest) -> str:
    kickoff = request.kickoff.isoformat()
    return f"{request.match_id} | {kickoff} | {request.home_team} vs {request.away_team}"


def _format_prediction_summary(record: PredictionRecord) -> str:
    return (
        f"{record.request.match_id} | {record.request.home_team} vs "
        f"{record.request.away_team} | {record.prediction.outcome.value} | "
        f"{record.prediction.confidence:.3f}"
    )


def _print_prediction(record: PredictionRecord) -> None:
    print(f"Prediction: {record.prediction.outcome.value}")
    print(f"Confidence: {record.prediction.confidence:.3f}")
    print(record.prediction.rationale)


def _catalog_for_args(
    fixture_catalog: FixtureCatalog,
    competition_id: str,
    live_world_cup: bool,
    source_url: str,
) -> FixtureCatalog:
    if not live_world_cup:
        return fixture_catalog
    if competition_id != WORLD_CUP_COMPETITION_ID:
        raise ValueError("--live-world-cup can only be used with world-cup-2026")
    return load_world_cup_catalog(source_url)


def _load_results(path: Path) -> tuple[MatchResult, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Result import must be a JSON list")
    return tuple(_result_from_mapping(cast(dict[str, object], item)) for item in data)


def _result_from_mapping(item: dict[str, object]) -> MatchResult:
    completed_at = item.get("completed_at")
    if not isinstance(completed_at, str):
        completed_at = datetime.now(UTC).isoformat()

    match_id = item.get("match_id")
    home_score = item.get("home_score")
    away_score = item.get("away_score")
    if not isinstance(match_id, str):
        raise ValueError("Result item is missing a string match_id")
    if not isinstance(home_score, int) or not isinstance(away_score, int):
        raise ValueError(f"Result item for {match_id!r} must include integer scores")

    return MatchResult(
        match_id=match_id,
        home_score=home_score,
        away_score=away_score,
        completed_at=datetime.fromisoformat(completed_at),
    )


if __name__ == "__main__":
    main()
