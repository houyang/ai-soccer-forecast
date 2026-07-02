"""Command-line interface for the World Cup predictor.

Commands:
  predict            -> predictions/worldcup-2026-predictions-after1st-group.{md,json}
  card "Home" "Away" -> predictions/<Home>-vs-<Away>.{pdf,json}
  bracket            -> predictions/worldcup-2026-knockout-bracket.{md,json} + per-round PDFs
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from soccer_agent.worldcup.card import build_card
from soccer_agent.worldcup.cardpdf import render_card_pdf
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.forecast import forecast_bracket
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.live import LineupFetcher
from soccer_agent.worldcup.ranking import rank_all
from soccer_agent.worldcup.simulate import simulate_bracket
from soccer_agent.worldcup.standings import group_standings

PRED_DIR = Path("predictions")


def _engine():
    wc = load_worldcup()
    rankings = rank_all(wc)
    forms = compute_forms(wc)
    strengths = recalibrated_strength(wc, rankings, forms)
    fetcher = LineupFetcher() if os.getenv("API_FOOTBALL_KEY") else None
    return wc, rankings, strengths, forms, fetcher


def _team_by_name(wc, name: str):
    name_l = name.lower().strip()
    for t in wc.teams.values():
        if t.name.lower() == name_l or t.name.lower().startswith(name_l):
            return t
    raise SystemExit(f"team not found: {name}")


def _write_predictions(wc, rankings, strengths, forms, fetcher) -> int:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    sim = simulate_bracket(wc, rankings, strengths, forms, fetcher=fetcher, n=10000)
    gs = group_standings(wc)

    # Standings dict
    standings_json = {g: [r.__dict__ for r in rows] for g, rows in gs.items()}
    r32_json = [p.to_dict() for p in sim.r32_predictions]
    champ_sorted = sorted(sim.champion.items(), key=lambda kv: kv[1], reverse=True)[:10]
    bracket_json = {
        "champion_top10": [{"team": wc.teams[t].name, "probability": round(p, 4)} for t, p in champ_sorted],
        "method": "Monte-Carlo 10000 iters; R32 fixtures exact, R16+ bracket pairing approximated by sorted fixture_id.",
    }
    payload = {"standings": standings_json, "r32": r32_json, "bracket": bracket_json}
    (PRED_DIR / "worldcup-2026-predictions-after1st-group.json").write_text(json.dumps(payload, indent=2))

    # Markdown
    lines = ["# FIFA 2026 World Cup — Predictions (after group stage)", ""]
    lines.append("Group stage complete. Below: final group standings, all 16 Round-of-32 "
                 "predictions, and a Monte-Carlo bracket simulation to the champion.")
    lines.append("")
    lines.append("> **Lineup provenance:** when `API_FOOTBALL_KEY` is set, each side's most-recent")
    lines.append("> played World Cup lineup is fetched live (source: `live`); otherwise lineups")
    lines.append("> are projected from a curated formation table + squad ratings (source: `projected`).")
    lines.append("> This committed file was generated with live lineups; regenerate offline for the")
    lines.append("> projected variant. R32 pairings are the real fixtures; the R16→Final bracket")
    lines.append("> pairing is approximated (sorted by fixture_id).")
    lines.append("")
    lines.append("## Group standings")
    for g, rows in gs.items():
        lines.append(f"\n### {g}\n")
        lines.append("| Team | P | W | D | L | GF | GA | GD | Pts |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            lines.append(f"| {r.name} | {r.played} | {r.wins} | {r.draws} | {r.losses} | {r.gf} | {r.ga} | {r.gd} | {r.pts} |")
    lines.append("\n## Round of 32\n")
    for p in sim.r32_predictions:
        ko = p.kickoff.strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"- `{ko}`  **{p.home_name} {p.score_home}-{p.score_away} {p.away_name}**  "
                     f"(W {p.p_home:.0%} / D {p.p_draw:.0%} / L {p.p_away:.0%})  — {p.rationale}")
    lines.append("\n## Bracket simulation (Monte-Carlo, 10000 iters)\n")
    lines.append("R32 pairings are the real fixtures; R16→Final pairing is approximated "
                 "(sorted by fixture_id).\n")
    lines.append("**Champion probabilities (top 10):**\n")
    for t, p in champ_sorted:
        lines.append(f"- {wc.teams[t].name}: {p:.1%}")
    (PRED_DIR / "worldcup-2026-predictions-after1st-group.md").write_text("\n".join(lines))
    return 0


def _write_card(wc, rankings, strengths, forms, fetcher, home_name: str, away_name: str) -> int:
    home = _team_by_name(wc, home_name)
    away = _team_by_name(wc, away_name)
    m = next((x for x in wc.matches if x.matchday == 0 and {x.home_id, x.away_id} == {home.id, away.id}), None)
    fid = m.fixture_id if m else None
    card = build_card(wc, rankings, strengths, forms, home.id, away.id, fetcher=fetcher, fixture_id=fid)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{home.name}-vs-{away.name}"
    (PRED_DIR / f"{stem}.json").write_text(json.dumps(card.to_dict(), indent=2))
    try:
        render_card_pdf(card, PRED_DIR / f"{stem}.pdf")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
    return 0


def _write_bracket(wc, rankings, strengths, forms, fetcher) -> int:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    fc = forecast_bracket(wc, rankings, strengths, forms, fetcher=fetcher)
    payload = fc.to_dict()
    # Brief Monte-Carlo champion context.
    sim = simulate_bracket(wc, rankings, strengths, forms, fetcher=fetcher, n=10000)
    champ_top5 = sorted(sim.champion.items(), key=lambda kv: kv[1], reverse=True)[:5]
    payload["monte_carlo_champion_top5"] = [{"team": wc.teams[t].name, "probability": round(p, 4)} for t, p in champ_top5]
    payload["method"] = ("Deterministic modal bracket: each match's modal winner advances (drawn "
                         "ties go to ET/penalties). R32 = real fixtures; R16→Final pairing follows "
                         "R32 schedule order (documented). MC champion odds are a separate 10k sim.")
    (PRED_DIR / "worldcup-2026-knockout-bracket.json").write_text(json.dumps(payload, indent=2))

    lines = ["# FIFA 2026 World Cup — Knockout Bracket Forecast", ""]
    lines.append("Deterministic modal bracket: every match predicted; the modal winner advances each round.")
    lines.append("Drawn knockout ties go to extra time/penalties (marked `ET`). R32 = real fixtures; "
                 "R16→Final pairing follows R32 schedule order (best-available; dataset has no official slot map).")
    lines.append("")
    for rnd in ("R32", "R16", "QF", "SF", "3rd", "Final"):
        matches = fc.rounds.get(rnd, [])
        if not matches:
            continue
        title = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-Finals",
                 "SF": "Semi-Finals", "Final": "Final", "3rd": "Third-Place Play-off"}[rnd]
        lines.append(f"\n## {title}\n")
        for bm in matches:
            p = bm.prediction
            ko = bm.kickoff.strftime("%Y-%m-%d %H:%M UTC") if bm.kickoff else "TBD"
            et = " (ET/pen)" if bm.expected_extra_time else ""
            lines.append(
                f"- `{ko}`  **{bm.home_name} {p.score_home}-{p.score_away} {bm.away_name}**{et}  "
                f"(W {p.p_home:.0%} / D {p.p_draw:.0%} / L {p.p_away:.0%})  "
                f"-> advances: **{bm.advancing_name}**"
            )
    lines.append("\n## Champion")
    lines.append(f"**{fc.to_dict()['champion']}** (runner-up: {fc.to_dict()['runner_up']}; "
                 f"third: {fc.to_dict()['third_place']})")
    lines.append("\n### Monte-Carlo champion odds (10k sims, top 5)")
    for t, p in champ_top5:
        lines.append(f"- {wc.teams[t].name}: {p:.1%}")
    (PRED_DIR / "worldcup-2026-knockout-bracket.md").write_text("\n".join(lines))

    # Auto-generate a PDF for every predicted match, R32 through the Final + 3rd place.
    cards_dir = PRED_DIR / "bracket-cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    for rnd in ("R32", "R16", "QF", "SF", "3rd", "Final"):
        for bm in fc.rounds.get(rnd, []):
            card = build_card(wc, rankings, strengths, forms, bm.home_id, bm.away_id,
                              fetcher=fetcher, kickoff=bm.kickoff, venue=bm.venue)
            stem = f"{rnd}-{bm.match_no}-{bm.home_name}-vs-{bm.away_name}"
            (cards_dir / f"{stem}.json").write_text(json.dumps(card.to_dict(), indent=2))
            try:
                render_card_pdf(card, cards_dir / f"{stem}.pdf")
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print("usage: python -m soccer_agent.worldcup {predict|card|bracket} [...]", file=sys.stderr)
        return 2
    cmd = argv[0]
    wc, rankings, strengths, forms, fetcher = _engine()
    if cmd == "predict":
        return _write_predictions(wc, rankings, strengths, forms, fetcher)
    if cmd == "card":
        if len(argv) < 3:
            print("usage: card \"Home\" \"Away\"", file=sys.stderr)
            return 2
        return _write_card(wc, rankings, strengths, forms, fetcher, argv[1], argv[2])
    if cmd == "bracket":
        return _write_bracket(wc, rankings, strengths, forms, fetcher)
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2
