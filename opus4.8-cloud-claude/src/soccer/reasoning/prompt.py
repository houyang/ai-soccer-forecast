from __future__ import annotations

import json

from soccer.models import MatchDossier, Outcome, normalize_probs
from soccer.reasoning.base import ReasonerError, ReasonResult

_INSTRUCTIONS = (
    "You are a football match analyst. Using the dossier, estimate the probability "
    "of each 1X2 outcome. Respond with ONLY a JSON object with keys: "
    '"home", "draw", "away" (numbers), "confidence" (0-1), "rationale" (string).'
)


def _form_line(dossier: MatchDossier, side: str) -> str:
    form = dossier.form.get(side)
    if form is None:
        return f"{side}: form unavailable"
    return (
        f"{side} ({form.team}): last={[o.value for o in form.last_n]} "
        f"pts={form.points} gf={form.gf} ga={form.ga} streak={form.streak}"
    )


def render_prompt(dossier: MatchDossier) -> str:
    m = dossier.match
    odds = dossier.odds
    odds_line = (
        "odds unavailable" if odds is None else f"odds H/D/A = {odds.home}/{odds.draw}/{odds.away}"
    )
    h2h = dossier.h2h
    h2h_line = (
        "h2h unavailable"
        if h2h is None
        else f"h2h home_wins={h2h.home_wins} draws={h2h.draws} away_wins={h2h.away_wins}"
    )
    weather = dossier.weather
    weather_line = (
        "weather unavailable"
        if weather is None
        else f"weather {weather.condition} {weather.temp_c}C wind {weather.wind_kph}kph"
    )
    lines = [
        _INSTRUCTIONS,
        f"Match: {m.home} (home) vs {m.away} (away), {m.competition} {m.season}.",
        _form_line(dossier, "home"),
        _form_line(dossier, "away"),
        h2h_line,
        weather_line,
        odds_line,
        f"Missing data: {list(dossier.missing) or 'none'}.",
    ]
    return "\n".join(lines)


def parse_reason_json(raw: str) -> ReasonResult:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReasonerError(f"reasoner did not return JSON: {exc}") from exc
    required = {"home", "draw", "away", "confidence", "rationale"}
    if not required <= set(data):
        raise ReasonerError(f"reasoner JSON missing keys: {required - set(data)}")
    try:
        raw_probs = {
            Outcome.HOME: float(data["home"]),
            Outcome.DRAW: float(data["draw"]),
            Outcome.AWAY: float(data["away"]),
        }
        confidence = float(data["confidence"])
    except (TypeError, ValueError) as exc:
        raise ReasonerError(f"reasoner returned non-numeric value: {exc}") from exc
    if any(v < 0 for v in raw_probs.values()):
        raise ReasonerError("reasoner returned negative probability")
    probs = normalize_probs(raw_probs)
    confidence = min(max(confidence, 0.0), 1.0)
    return ReasonResult(probs=probs, confidence=confidence, rationale=str(data["rationale"]))
