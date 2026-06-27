"""Command-line entry points."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from soccer.api_football import (
    ApiFootballClient,
    fetch_world_cup_2026_match_preview_updates,
    fetch_world_cup_2026_match_updates,
    fetch_world_cup_2026_snapshot,
)
from soccer.evaluation import EvaluationHarness
from soccer.fixture_tools import FixtureCatalog, build_fixture_agent, default_catalog
from soccer.live_world_cup import (
    DEFAULT_WORLD_CUP_SCHEDULE_URL,
    WORLD_CUP_COMPETITION_ID,
    load_world_cup_catalog,
)
from soccer.models import MatchRequest, MatchResult, PredictionRecord
from soccer.storage import JsonlPredictionLog
from soccer.world_cup_2026 import (
    DEFAULT_WORLD_CUP_DATA_DIR,
    GroupStageMatch,
    WorldCupDataSet,
    load_world_cup_dataset,
    predict_group_stage_scores,
    prediction_to_json,
    rank_world_cup_entities,
    render_group_stage_markdown,
)
from soccer.world_cup_preview import (
    build_world_cup_match_preview,
    match_preview_to_json,
    render_world_cup_match_preview_pdf,
)


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

    fetch_world_cup_data = subparsers.add_parser(
        "fetch-world-cup-data",
        help="Fetch API-Football snapshots for FIFA World Cup 2026 modeling",
    )
    fetch_world_cup_data.add_argument("--api-key")
    fetch_world_cup_data.add_argument("--data-dir", type=Path, default=DEFAULT_WORLD_CUP_DATA_DIR)
    fetch_world_cup_data.add_argument("--world-cup-league-id", type=int, default=1)
    fetch_world_cup_data.add_argument("--world-cup-season", type=int, default=2026)
    fetch_world_cup_data.add_argument("--club-season", type=int, default=2025)
    fetch_world_cup_data.add_argument("--recent-fixture-count", type=int, default=20)
    fetch_world_cup_data.add_argument("--request-delay-seconds", type=float, default=0.0)

    fetch_world_cup_updates = subparsers.add_parser(
        "fetch-world-cup-match-updates",
        help="Refresh FIFA World Cup 2026 fixture results, standings, lineups, and events",
    )
    fetch_world_cup_updates.add_argument("--api-key")
    fetch_world_cup_updates.add_argument(
        "--data-dir", type=Path, default=DEFAULT_WORLD_CUP_DATA_DIR
    )
    fetch_world_cup_updates.add_argument("--world-cup-league-id", type=int, default=1)
    fetch_world_cup_updates.add_argument("--world-cup-season", type=int, default=2026)
    fetch_world_cup_updates.add_argument(
        "--completed-round-limit",
        type=int,
        help="Only fetch tactical snapshots for completed group-stage matches through this round",
    )
    fetch_world_cup_updates.add_argument("--request-delay-seconds", type=float, default=0.0)

    world_cup_scores = subparsers.add_parser(
        "predict-world-cup-group-stage",
        help="Predict final scores for FIFA World Cup 2026 group stage matches",
    )
    world_cup_scores.add_argument("--data-dir", type=Path, default=DEFAULT_WORLD_CUP_DATA_DIR)
    world_cup_scores.add_argument("--output", choices=("text", "json", "markdown"), default="text")
    world_cup_scores.add_argument("--expected-teams", type=int, default=48)
    world_cup_scores.add_argument(
        "--completed-round-limit",
        type=int,
        help="Use completed-match updates only through this group-stage round",
    )
    world_cup_scores.add_argument(
        "--remaining-only",
        action="store_true",
        help="Only output matches not completed within the selected update window",
    )

    world_cup_preview = subparsers.add_parser(
        "predict-world-cup-match-preview",
        help="Predict one not-started World Cup match and write PDF/JSON previews",
    )
    world_cup_preview.add_argument("match_id", help="Fixture id, with or without wc-2026- prefix")
    world_cup_preview.add_argument("--api-key")
    world_cup_preview.add_argument("--data-dir", type=Path, default=DEFAULT_WORLD_CUP_DATA_DIR)
    world_cup_preview.add_argument("--world-cup-league-id", type=int, default=1)
    world_cup_preview.add_argument("--world-cup-season", type=int, default=2026)
    world_cup_preview.add_argument("--expected-teams", type=int)
    world_cup_preview.add_argument("--output", type=Path)
    world_cup_preview.add_argument("--json-output", type=Path)
    world_cup_preview.add_argument("--request-delay-seconds", type=float, default=0.0)
    world_cup_preview.add_argument(
        "--no-refresh",
        action="store_true",
        help="Use existing local snapshots instead of refreshing provider match updates",
    )

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
        evaluation_summary = EvaluationHarness(JsonlPredictionLog(args.log)).evaluate()
        print(f"Settled: {evaluation_summary.settled_count}")
        print(f"Accuracy: {evaluation_summary.accuracy:.3f}")
        print(f"Average confidence: {evaluation_summary.average_confidence:.3f}")
        print(f"Brier score: {evaluation_summary.brier_score:.3f}")
    elif args.command == "fetch-world-cup-data":
        api_key = args.api_key or os.environ.get("API_FOOTBALL_KEY")
        if api_key is None:
            raise ValueError("Provide --api-key or set API_FOOTBALL_KEY")
        fetch_summary = fetch_world_cup_2026_snapshot(
            ApiFootballClient(api_key),
            args.data_dir,
            world_cup_league_id=args.world_cup_league_id,
            world_cup_season=args.world_cup_season,
            club_season=args.club_season,
            recent_fixture_count=args.recent_fixture_count,
            request_delay_seconds=args.request_delay_seconds,
        )
        print(f"Snapshot directory: {fetch_summary.output_dir}")
        print(f"National teams: {fetch_summary.national_teams}")
        print(f"Players: {fetch_summary.players}")
        print(f"Coaches: {fetch_summary.coaches}")
        print(f"Clubs: {fetch_summary.clubs}")
        print(f"Leagues: {fetch_summary.leagues}")
        print(f"Files written: {fetch_summary.files_written}")
    elif args.command == "fetch-world-cup-match-updates":
        api_key = args.api_key or os.environ.get("API_FOOTBALL_KEY")
        if api_key is None:
            raise ValueError("Provide --api-key or set API_FOOTBALL_KEY")
        update_summary = fetch_world_cup_2026_match_updates(
            ApiFootballClient(api_key),
            args.data_dir,
            world_cup_league_id=args.world_cup_league_id,
            world_cup_season=args.world_cup_season,
            completed_round_limit=args.completed_round_limit,
            request_delay_seconds=args.request_delay_seconds,
        )
        print(f"Snapshot directory: {update_summary.output_dir}")
        print(f"Fixtures: {update_summary.fixtures}")
        print(f"Standings refreshed: {update_summary.standings_refreshed}")
        print(f"Tactical fixtures: {update_summary.tactical_fixtures}")
        print(f"Files written: {update_summary.files_written}")
    elif args.command == "predict-world-cup-group-stage":
        dataset = load_world_cup_dataset(
            args.data_dir,
            expected_team_count=args.expected_teams,
            completed_round_limit=args.completed_round_limit,
        )
        rankings = rank_world_cup_entities(dataset)
        predictions = predict_group_stage_scores(
            dataset,
            rankings,
            remaining_only=args.remaining_only,
        )
        if args.output == "json":
            print(json.dumps([prediction_to_json(prediction) for prediction in predictions]))
        elif args.output == "markdown":
            print(render_group_stage_markdown(predictions), end="")
        else:
            for prediction in predictions:
                group = f"{prediction.group} | " if prediction.group else ""
                print(
                    f"{prediction.match_id} | {group}{prediction.home_team} "
                    f"{prediction.home_score}-{prediction.away_score} "
                    f"{prediction.away_team} | {prediction.outcome.value} | "
                    f"{prediction.confidence:.3f}"
                )
    elif args.command == "predict-world-cup-match-preview":
        match_id = _normalize_world_cup_match_id(args.match_id)
        if not args.no_refresh:
            api_key = args.api_key or os.environ.get("API_FOOTBALL_KEY")
            if api_key is None:
                raise ValueError("Provide --api-key, set API_FOOTBALL_KEY, or use --no-refresh")
            preview_update_summary = fetch_world_cup_2026_match_preview_updates(
                ApiFootballClient(api_key),
                args.data_dir,
                match_id,
                world_cup_league_id=args.world_cup_league_id,
                world_cup_season=args.world_cup_season,
                request_delay_seconds=args.request_delay_seconds,
            )
            print(f"Refreshed fixture: {preview_update_summary.target_fixture_id}")
            print(f"Prior completed fixtures: {preview_update_summary.prior_completed_fixtures}")
            print(f"Files written: {preview_update_summary.files_written}")

        base_dataset = load_world_cup_dataset(
            args.data_dir,
            expected_team_count=args.expected_teams,
        )
        if match_id in base_dataset.completed_matches:
            raise ValueError(f"{match_id} is already completed in the loaded dataset")
        target_match = _world_cup_match_for_id(base_dataset, match_id)
        dataset = load_world_cup_dataset(
            args.data_dir,
            expected_team_count=args.expected_teams,
            completed_before=target_match.kickoff,
        )
        rankings = rank_world_cup_entities(dataset)
        preview = build_world_cup_match_preview(
            dataset,
            rankings,
            match_id,
            snapshot_dir=args.data_dir,
        )
        output_path = args.output or Path("predictions") / f"{match_id}-preview.pdf"
        render_world_cup_match_preview_pdf(preview, output_path)
        print(f"PDF written: {output_path}")
        if args.json_output is not None:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(
                json.dumps(match_preview_to_json(preview), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            print(f"JSON written: {args.json_output}")


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


def _normalize_world_cup_match_id(match_id: str) -> str:
    value = match_id.strip()
    if value.startswith("wc-2026-"):
        return value
    return f"wc-2026-{value}"


def _world_cup_match_for_id(dataset: WorldCupDataSet, match_id: str) -> GroupStageMatch:
    for match in dataset.matches:
        if match.match_id == match_id:
            return match
    raise ValueError(f"World Cup match {match_id!r} was not found")


if __name__ == "__main__":
    main()
