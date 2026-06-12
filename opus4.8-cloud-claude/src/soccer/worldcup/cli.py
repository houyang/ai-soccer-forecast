"""`soccer wc {fetch,rank,predict}` subcommands.

`fetch` is the only networked path (needs SOCCER_API_FOOTBALL_KEY); `rank` and `predict`
read the locally cached dataset and run fully offline.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC
from pathlib import Path

from soccer.config import AppConfig
from soccer.worldcup.apifootball import ApiFootballClient, ApiFootballError, urllib_transport
from soccer.worldcup.cache import JsonCache
from soccer.worldcup.entities import WorldCup
from soccer.worldcup.ingest import ingest_world_cup
from soccer.worldcup.predict import MatchPrediction, predict_group_stage
from soccer.worldcup.ranking import Rankings, rank_all, top_n


def _dataset_path(config: AppConfig) -> Path:
    return config.data_dir / "worldcup-2026.json"


def _prediction_dir(config: AppConfig) -> Path:
    return config.prediction_dir


def _predictions_path(config: AppConfig) -> Path:
    return _prediction_dir(config) / "worldcup-2026-predictions.json"


def _report_path(config: AppConfig) -> Path:
    return _prediction_dir(config) / "worldcup-2026-predictions.md"


def load_dataset(path: Path) -> WorldCup:
    if not path.exists():
        raise FileNotFoundError(
            f"dataset {path} not found; run `soccer wc fetch` first (needs an API key)"
        )
    return WorldCup.from_dict(json.loads(path.read_text(encoding="utf-8")))


def cmd_fetch(args: argparse.Namespace, config: AppConfig) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not config.api_football_key:
        print("SOCCER_API_FOOTBALL_KEY is not set; cannot fetch live data", flush=True)
        return 1
    client = ApiFootballClient(
        config.api_football_key,
        base_url=config.api_football_base_url,
        transport=urllib_transport(timeout=30.0),
        cache=JsonCache(config.data_dir / "api"),
        throttle_seconds=args.throttle,
    )
    try:
        world_cup = ingest_world_cup(client)
    except ApiFootballError as exc:
        print(f"fetch failed: {exc}")
        return 1
    path = _dataset_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(world_cup.to_dict()), encoding="utf-8")
    print(
        f"fetched {len(world_cup.teams)} teams, {len(world_cup.players)} players, "
        f"{len(world_cup.clubs)} clubs, {len(world_cup.leagues)} leagues, "
        f"{len(world_cup.matches)} matches -> {path}"
    )
    return 0


def _print_ranking_tables(wc: WorldCup, ranks: Rankings, limit: int) -> None:
    print(f"== Top {limit} leagues ==")
    for lid, score in top_n(ranks.leagues, limit):
        lg = wc.leagues[lid]
        print(f"  {score:5.1f}  {lg.name} ({lg.country})")
    print(f"== Top {limit} clubs ==")
    for cid, score in top_n(ranks.clubs, limit):
        print(f"  {score:5.1f}  {wc.clubs[cid].name}")
    print(f"== Top {limit} players ==")
    for pid, score in top_n(ranks.players, limit):
        p = wc.players[pid]
        print(f"  {score:5.1f}  {p.name} ({p.position}, {p.goals}g, r{p.rating})")
    print(f"== Top {limit} coaches ==")
    for cid, score in top_n(ranks.coaches, limit):
        print(f"  {score:5.1f}  {wc.coaches[cid].name} ({wc.teams[wc.coaches[cid].team_id].name})")
    print("== National teams ==")
    for tid, score in top_n(ranks.teams, len(ranks.teams)):
        t = wc.teams[tid]
        print(f"  {score:5.1f}  {t.name} ({t.group})")


def cmd_rank(args: argparse.Namespace, config: AppConfig) -> int:
    wc = load_dataset(_dataset_path(config))
    _print_ranking_tables(wc, rank_all(wc), args.top)
    return 0


def _print_predictions(predictions: list[MatchPrediction]) -> None:
    by_matchday: dict[int, list[MatchPrediction]] = {}
    for pred in predictions:
        by_matchday.setdefault(pred.matchday, []).append(pred)
    for matchday in sorted(by_matchday):
        print(f"\n===== Matchday {matchday} =====")
        for pred in sorted(by_matchday[matchday], key=lambda p: (p.group, p.home_name)):
            print(
                f"  [{pred.group}] {pred.home_name} {pred.score_home}-{pred.score_away} "
                f"{pred.away_name}   "
                f"(W {pred.p_home:.0%} / D {pred.p_draw:.0%} / L {pred.p_away:.0%})"
            )


def _render_report(predictions: list[MatchPrediction]) -> str:
    """Human-friendly Markdown: group by group, round by round, ordered by kickoff time."""
    by_group: dict[str, dict[int, list[MatchPrediction]]] = {}
    for pred in predictions:
        by_group.setdefault(pred.group, {}).setdefault(pred.matchday, []).append(pred)
    lines = [
        "# FIFA 2026 World Cup — Group-Stage Predictions",
        "",
        "Predicted result and final score for every group-stage match, organized by group,",
        "then matchday, ordered by kickoff time. Percentages are home win / draw / away win.",
        "",
    ]
    for group in sorted(by_group):
        lines += [f"## {group}", ""]
        for matchday in sorted(by_group[group]):
            lines += [f"### Matchday {matchday}", ""]
            for pred in sorted(by_group[group][matchday], key=lambda p: p.kickoff):
                kickoff = pred.kickoff.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
                lines.append(
                    f"- `{kickoff}`  **{pred.home_name} {pred.score_home}-{pred.score_away} "
                    f"{pred.away_name}**  "
                    f"(W {pred.p_home:.0%} / D {pred.p_draw:.0%} / L {pred.p_away:.0%})"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def cmd_predict(args: argparse.Namespace, config: AppConfig) -> int:
    wc = load_dataset(_dataset_path(config))
    predictions = predict_group_stage(wc, rank_all(wc))
    _print_predictions(predictions)
    json_path = _predictions_path(config)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps([p.to_dict() for p in predictions], indent=2), encoding="utf-8")
    report_path = _report_path(config)
    report_path.write_text(_render_report(predictions), encoding="utf-8")
    print(f"\nwrote {len(predictions)} predictions -> {json_path}")
    print(f"wrote readable report -> {report_path}")
    return 0


def add_wc_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("wc", help="FIFA 2026 World Cup pipeline")
    wc_sub = parser.add_subparsers(dest="wc_command", required=True)

    p_fetch = wc_sub.add_parser("fetch", help="pull live data into the local cache")
    p_fetch.add_argument("--throttle", type=float, default=0.02, help="seconds between calls")
    p_fetch.set_defaults(func=cmd_fetch)

    p_rank = wc_sub.add_parser("rank", help="print 0-100 rankings")
    p_rank.add_argument("--top", type=int, default=15)
    p_rank.set_defaults(func=cmd_rank)

    p_predict = wc_sub.add_parser("predict", help="predict all group-stage scorelines")
    p_predict.set_defaults(func=cmd_predict)
