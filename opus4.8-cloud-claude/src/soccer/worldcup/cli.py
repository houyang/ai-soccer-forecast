"""`soccer wc {fetch,rank,predict,knockout}` subcommands.

`fetch` is the only networked path (needs SOCCER_API_FOOTBALL_KEY); `rank`, `predict`, and
`knockout` read the locally cached dataset and run fully offline.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict
from datetime import UTC
from pathlib import Path
from typing import Any

from soccer.config import AppConfig
from soccer.worldcup.adjust import compute_adjustments
from soccer.worldcup.apifootball import ApiFootballClient, ApiFootballError, urllib_transport
from soccer.worldcup.bracket import BracketError, build_bracket
from soccer.worldcup.cache import JsonCache
from soccer.worldcup.card import build_card
from soccer.worldcup.cardpdf import render_card_pdf
from soccer.worldcup.entities import WorldCup
from soccer.worldcup.ingest import ingest_world_cup
from soccer.worldcup.live import refresh_fixture, refresh_live
from soccer.worldcup.predict import (
    KnockoutPrediction,
    MatchPrediction,
    predict_group_stage,
    predict_remaining,
)
from soccer.worldcup.ranking import Rankings, rank_all, top_n
from soccer.worldcup.simulate import Podium, TeamOdds, run_modal_bracket, run_monte_carlo
from soccer.worldcup.standings import team_labels


def _dataset_path(config: AppConfig) -> Path:
    return config.data_dir / "worldcup-2026.json"


def _prediction_dir(config: AppConfig) -> Path:
    return config.prediction_dir


def _resolve_outputs(args: argparse.Namespace, config: AppConfig) -> tuple[Path, Path]:
    out_dir = Path(args.out_dir) if getattr(args, "out_dir", None) else _prediction_dir(config)
    name = getattr(args, "name", None) or "worldcup-2026-predictions"
    return out_dir / f"{name}.json", out_dir / f"{name}.md"


def _render_remaining_report(wc: WorldCup, predictions: list[MatchPrediction]) -> str:
    """Markdown: completed actual results first, then updated predictions per group/matchday."""
    lines = [
        "# FIFA 2026 World Cup — Updated Predictions (after Matchday 1)",
        "",
        "Actual results so far, then predicted result and final score for every remaining",
        "group-stage match. Percentages are home win / draw / away win.",
        "",
        "## Completed results",
        "",
    ]
    played = sorted((m for m in wc.matches if m.played), key=lambda m: (m.matchday, m.kickoff))
    for match in played:
        home = wc.teams[match.home_id].name
        away = wc.teams[match.away_id].name
        lines.append(
            f"- `MD{match.matchday}` **{home} {match.home_goals}-{match.away_goals} {away}**"
        )
    lines.append("")
    by_group: dict[str, dict[int, list[MatchPrediction]]] = {}
    for pred in predictions:
        by_group.setdefault(pred.group, {}).setdefault(pred.matchday, []).append(pred)
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
    rankings = rank_all(wc)
    remaining = getattr(args, "remaining", False)
    json_path, report_path = _resolve_outputs(args, config)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    if remaining:
        adjustments = compute_adjustments(wc, rankings)
        predictions = predict_remaining(wc, rankings, adjustments)
        results = [
            {
                "fixture_id": m.fixture_id,
                "matchday": m.matchday,
                "group": m.group,
                "home_name": wc.teams[m.home_id].name,
                "away_name": wc.teams[m.away_id].name,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
            }
            for m in sorted(
                (m for m in wc.matches if m.played), key=lambda m: (m.matchday, m.kickoff)
            )
        ]
        payload: object = {
            "predictions": [p.to_dict() for p in predictions],
            "results": results,
            "adjustments": {str(tid): asdict(a) for tid, a in adjustments.items()},
        }
        report = _render_remaining_report(wc, predictions)
    else:
        predictions = predict_group_stage(wc, rankings)
        payload = [p.to_dict() for p in predictions]
        report = _render_report(predictions)
    _print_predictions(predictions)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print(f"\nwrote {len(predictions)} predictions -> {json_path}")
    print(f"wrote readable report -> {report_path}")
    return 0


def cmd_refresh(args: argparse.Namespace, config: AppConfig) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not config.api_football_key:
        print("SOCCER_API_FOOTBALL_KEY is not set; cannot refresh live data", flush=True)
        return 1
    wc = load_dataset(_dataset_path(config))
    client = ApiFootballClient(
        config.api_football_key,
        base_url=config.api_football_base_url,
        transport=urllib_transport(timeout=30.0),
        cache=JsonCache(config.data_dir / "api"),
        throttle_seconds=args.throttle,
    )
    try:
        updated = refresh_live(wc, client)
    except ApiFootballError as exc:
        print(f"refresh failed: {exc}")
        return 1
    path = _dataset_path(config)
    path.write_text(json.dumps(updated.to_dict()), encoding="utf-8")
    played = sum(1 for m in updated.matches if m.played)
    print(f"refreshed {played} played matches, {len(updated.lineups)} lineups -> {path}")
    return 0


def cmd_card(args: argparse.Namespace, config: AppConfig) -> int:
    wc = load_dataset(_dataset_path(config))
    if getattr(args, "refresh", False):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        if not config.api_football_key:
            print("SOCCER_API_FOOTBALL_KEY is not set; cannot refresh live data", flush=True)
            return 1
        client = ApiFootballClient(
            config.api_football_key,
            base_url=config.api_football_base_url,
            transport=urllib_transport(timeout=30.0),
            cache=JsonCache(config.data_dir / "api"),
            throttle_seconds=args.throttle,
        )
        try:
            wc = refresh_fixture(wc, client, args.fixture_id)
        except ApiFootballError as exc:
            print(f"refresh failed: {exc}")
            return 1
        _dataset_path(config).write_text(json.dumps(wc.to_dict()), encoding="utf-8")

    rankings = rank_all(wc)
    try:
        card = build_card(wc, rankings, args.fixture_id)
    except ValueError as exc:
        print(f"{exc}; run `soccer wc predict` to list fixture ids")
        return 1

    out_dir = Path(args.out_dir) if args.out_dir else _prediction_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or f"card-{args.fixture_id}"
    written: list[Path] = []
    if args.format in ("json", "both"):
        json_path = out_dir / f"{name}.json"
        json_path.write_text(json.dumps(card.to_dict(), indent=2), encoding="utf-8")
        written.append(json_path)
    if args.format in ("pdf", "both"):
        pdf_path = out_dir / f"{name}.pdf"
        try:
            render_card_pdf(card, pdf_path)
        except RuntimeError as exc:
            print(str(exc))
            return 1
        written.append(pdf_path)

    pred = card.prediction
    print(
        f"{card.home.name} {pred.score_home}-{pred.score_away} {card.away.name} "
        f"(W {pred.p_home:.0%} / D {pred.p_draw:.0%} / L {pred.p_away:.0%}); "
        f"lineups {card.home.source}/{card.away.source}"
    )
    for path in written:
        print(f"wrote {path}")
    return 0


def _render_knockout_report(
    preds: list[KnockoutPrediction],
    podium: Podium,
    odds: dict[int, TeamOdds],
) -> str:
    lines = [
        "# FIFA 2026 World Cup — Knockout-Stage Forecast",
        "",
        "Most-likely bracket from the live Round-of-32 draw forward, with each tie's "
        "predicted score and advancement odds, then Monte-Carlo title odds.",
        "",
        "## Predicted podium",
        "",
        f"- 🥇 **Champion:** {podium.champion_name}",
        f"- 🥈 **Runner-up:** {podium.runner_up_name}",
        f"- 🥉 **Third:** {podium.third_name}",
        f"- 4th: {podium.fourth_name}",
        "",
        "## Bracket",
        "",
    ]
    current = ""
    for p in preds:
        if p.round_name != current:
            current = p.round_name
            lines += [f"### {current}", ""]
        et = "  _(likely AET/pens)_" if p.expected_extra_time else ""
        lines.append(
            f"- `M{p.match_no}` **{p.home_name} {p.score_home}-{p.score_away} {p.away_name}**  "
            f"(adv {p.home_name} {p.p_home_advance:.0%} / {p.away_name} {p.p_away_advance:.0%})"
            f"{et}"
        )
    lines += [
        "",
        "## Title odds (top 12)",
        "",
        "| Team | Win | Final | Semi | Quarter |",
        "|---|---|---|---|---|",
    ]
    for o in sorted(odds.values(), key=lambda o: o.win, reverse=True)[:12]:
        row = (
            f"| {o.name} | {o.win:.1%} | {o.reach_final:.0%}"
            f" | {o.reach_sf:.0%} | {o.reach_qf:.0%} |"
        )
        lines.append(row)
    return "\n".join(lines).rstrip() + "\n"


def cmd_knockout(args: argparse.Namespace, config: AppConfig) -> int:
    wc = load_dataset(_dataset_path(config))
    if not any(m.round_name == "Round of 32" for m in wc.matches):
        print(
            "no Round of 32 fixtures in the dataset; run `soccer wc fetch` "
            "(or `refresh`) to pull the knockout draw first"
        )
        return 1
    rankings = rank_all(wc)
    try:
        ties = build_bracket(wc, team_labels(wc))
    except BracketError as exc:
        print(f"bracket error: {exc}")
        return 1
    preds, podium = run_modal_bracket(wc, rankings, ties)
    odds = run_monte_carlo(wc, rankings, ties, rng=random.Random(args.seed), n_sims=args.sims)
    out_dir = Path(args.out_dir) if args.out_dir else _prediction_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or "worldcup-2026-knockout"
    payload: dict[str, Any] = {
        "bracket": [p.to_dict() for p in preds],
        "podium": podium.to_dict(),
        "title_odds": [
            o.to_dict() for o in sorted(odds.values(), key=lambda o: o.win, reverse=True)
        ],
    }
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / f"{name}.md").write_text(
        _render_knockout_report(preds, podium, odds), encoding="utf-8"
    )
    print(f"predicted champion: {podium.champion_name}")
    print(f"wrote {out_dir / f'{name}.json'} and {out_dir / f'{name}.md'}")
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

    p_predict = wc_sub.add_parser("predict", help="predict group-stage scorelines")
    p_predict.add_argument(
        "--remaining",
        action="store_true",
        help="forecast only unplayed matches, using actual results + lineups",
    )
    p_predict.add_argument("--out-dir", default=None, help="output directory for the files")
    p_predict.add_argument("--name", default=None, help="basename for the .json/.md files")
    p_predict.set_defaults(func=cmd_predict)

    p_refresh = wc_sub.add_parser("refresh", help="merge live results + lineups into the dataset")
    p_refresh.add_argument("--throttle", type=float, default=0.02, help="seconds between calls")
    p_refresh.set_defaults(func=cmd_refresh)

    p_card = wc_sub.add_parser("card", help="single-match pre-match preview (PDF/JSON)")
    p_card.add_argument("fixture_id", type=int, help="fixture id to preview")
    p_card.add_argument(
        "--refresh",
        action="store_true",
        help="pull this fixture's latest lineup/result first (needs an API key)",
    )
    p_card.add_argument("--out-dir", default=None, help="output directory for the files")
    p_card.add_argument("--name", default=None, help="basename for the output files")
    p_card.add_argument(
        "--format", choices=["pdf", "json", "both"], default="both", help="output format"
    )
    p_card.add_argument("--throttle", type=float, default=0.02, help="seconds between calls")
    p_card.set_defaults(func=cmd_card)

    p_ko = wc_sub.add_parser("knockout", help="forecast the knockout bracket to the final")
    p_ko.add_argument("--sims", type=int, default=20000, help="Monte-Carlo iterations")
    p_ko.add_argument("--seed", type=int, default=20260628, help="RNG seed (reproducible)")
    p_ko.add_argument("--out-dir", default=None, help="output directory for the files")
    p_ko.add_argument("--name", default=None, help="basename for the .json/.md files")
    p_ko.set_defaults(func=cmd_knockout)
