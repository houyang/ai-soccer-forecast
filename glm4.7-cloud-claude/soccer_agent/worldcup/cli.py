"""Command-line interface for the World Cup predictor.

Commands:
  predict            -> predictions/worldcup-2026-predictions-after1st-group.{md,json}
  card "Home" "Away" -> predictions/<Home>-vs-<Away>.{pdf,json}
  bracket            -> print champion + advancement odds to stdout
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from soccer_agent.worldcup.card import build_card
from soccer_agent.worldcup.cardpdf import render_card_pdf
from soccer_agent.worldcup.dataset import load_worldcup
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
    return wc, rankings, strengths, fetcher


def _team_by_name(wc, name: str):
    name_l = name.lower().strip()
    for t in wc.teams.values():
        if t.name.lower() == name_l or t.name.lower().startswith(name_l):
            return t
    raise SystemExit(f"team not found: {name}")


def _write_predictions(wc, rankings, strengths, fetcher) -> int:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    sim = simulate_bracket(wc, rankings, strengths, fetcher=fetcher, n=10000)
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


def _write_card(wc, rankings, strengths, fetcher, home_name: str, away_name: str) -> int:
    home = _team_by_name(wc, home_name)
    away = _team_by_name(wc, away_name)
    # Find an R32 fixture matching these two teams (either order).
    m = next((x for x in wc.matches if x.matchday == 0 and {x.home_id, x.away_id} == {home.id, away.id}), None)
    if m is None:
        raise SystemExit(f"no R32 fixture between {home.name} and {away.name}")
    card = build_card(wc, rankings, strengths, home.id, away.id, fetcher=fetcher, fixture_id=m.fixture_id)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{home.name}-vs-{away.name}"
    (PRED_DIR / f"{stem}.json").write_text(json.dumps(card.to_dict(), indent=2))
    try:
        render_card_pdf(card, PRED_DIR / f"{stem}.pdf")
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    if not argv:
        print("usage: python -m soccer_agent.worldcup {predict|card|bracket} [...]", file=sys.stderr)
        return 2
    cmd = argv[0]
    wc, rankings, strengths, fetcher = _engine()
    if cmd == "predict":
        return _write_predictions(wc, rankings, strengths, fetcher)
    if cmd == "card":
        if len(argv) < 3:
            print("usage: card \"Home\" \"Away\"", file=sys.stderr)
            return 2
        return _write_card(wc, rankings, strengths, fetcher, argv[1], argv[2])
    if cmd == "bracket":
        sim = simulate_bracket(wc, rankings, strengths, fetcher=fetcher, n=10000)
        for t, p in sorted(sim.champion.items(), key=lambda kv: kv[1], reverse=True)[:10]:
            print(f"{wc.teams[t].name}: {p:.1%}")
        return 0
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2
