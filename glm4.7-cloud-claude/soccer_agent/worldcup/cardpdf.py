"""Render a MatchCard to a one-page A4 PDF. reportlab is imported lazily."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from soccer_agent.worldcup.card import MatchCard, TeamCard

_INSTALL_HINT = "PDF output requires reportlab; install with: pip install 'soccer-agent[pdf]'"


def render_card_pdf(card: "MatchCard", path: str | Path) -> None:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(_INSTALL_HINT) from exc

    width, height = A4
    pdf = canvas.Canvas(str(path), pagesize=A4)
    left = 18 * mm
    right = width - 18 * mm
    cursor = height - 20 * mm

    def line(text: str, *, size: int = 10, gap: float = 5.2, x: float = left) -> None:
        nonlocal cursor
        pdf.setFont("Helvetica", size)
        pdf.drawString(x, cursor, text)
        cursor -= gap * mm

    kickoff = card.kickoff.strftime("%Y-%m-%d %H:%M %Z").strip() if card.kickoff else "TBD"
    line(f"{card.home.name}  vs  {card.away.name}", size=16, gap=8)
    line(f"{card.group}  ·  {kickoff}  ·  {card.venue}", size=9, gap=8)

    pred = card.prediction
    line(
        f"Prediction: {pred.home_name} {pred.score_home}-{pred.score_away} {pred.away_name}"
        f"   (W {pred.p_home:.0%} / D {pred.p_draw:.0%} / L {pred.p_away:.0%})",
        size=12, gap=6,
    )
    line(f"Expected goals: {pred.lambda_home:.2f} - {pred.lambda_away:.2f}", size=9)
    tops = ", ".join(f"{h}-{a} ({p:.0%})" for h, a, p in card.top_scorelines)
    line(f"Most likely scorelines: {tops}", size=9, gap=7)
    line(pred.rationale, size=8, gap=8)

    def team_block(team: "TeamCard", x: float) -> None:
        nonlocal cursor
        top = cursor
        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(x, cursor, f"{team.name}  ({team.formation})")
        cursor -= 5.5 * mm
        badge = team.source if team.source_matchday is None else f"{team.source} MD{team.source_matchday}"
        coach = team.coach_name or "?"
        record = "-".join(str(n) for n in team.coach_record) if team.coach_record else "?"
        line(f"Coach: {coach}  ({record} W-D-L)   [{badge}]", size=8, gap=5, x=x)
        line("Starting XI:", size=9, gap=5, x=x)
        for p in team.starters:
            line(f"  {p.position[:3]:<3} {p.name}  ({p.rating:.0f})", size=8, gap=4.2, x=x)
        line("Likely subs:", size=9, gap=5, x=x)
        for p in team.subs:
            line(f"  {p.position[:3]:<3} {p.name}  ({p.rating:.0f})", size=8, gap=4.2, x=x)
        cursor = top

    block_top = cursor
    team_block(card.home, left)
    cursor = block_top
    team_block(card.away, left + (right - left) / 2.0)

    pdf.showPage()
    pdf.save()
