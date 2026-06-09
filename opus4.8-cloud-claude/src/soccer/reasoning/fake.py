from __future__ import annotations

from soccer.models import (
    MatchDossier,
    MatchOutcome,
    MatchResult,
    Outcome,
    Prediction,
    normalize_probs,
)
from soccer.reasoning.base import ReasonResult

_POINTS = {MatchOutcome.WIN: 3.0, MatchOutcome.DRAW: 1.0, MatchOutcome.LOSS: 0.0}


def _form_strength(dossier: MatchDossier, side: str) -> float:
    form = dossier.form.get(side)
    if form is None or not form.last_n:
        return 1.0
    return sum(_POINTS[o] for o in form.last_n) / len(form.last_n)


class DeterministicReasoner:
    """Blends market-implied odds with a fixed form/H2H adjustment. No randomness."""

    name = "fake"

    def predict(self, dossier: MatchDossier) -> ReasonResult:
        if dossier.odds is not None:
            base = dict(dossier.odds.implied_probs)
        else:
            base = {Outcome.HOME: 0.4, Outcome.DRAW: 0.3, Outcome.AWAY: 0.3}
        home_str = _form_strength(dossier, "home")
        away_str = _form_strength(dossier, "away")
        # Tilt toward the in-form side; +0.05 weight per point of form gap.
        gap = (home_str - away_str) * 0.05
        adjusted = {
            Outcome.HOME: max(base[Outcome.HOME] + gap, 1e-6),
            Outcome.DRAW: max(base[Outcome.DRAW], 1e-6),
            Outcome.AWAY: max(base[Outcome.AWAY] - gap, 1e-6),
        }
        probs = normalize_probs(adjusted)
        pick = max(probs, key=lambda k: probs[k])
        confidence = round(probs[pick], 4)
        rationale = (
            f"Market-implied base adjusted by form gap {gap:+.3f} "
            f"(home {home_str:.2f} vs away {away_str:.2f}); "
            f"missing data: {list(dossier.missing) or 'none'}."
        )
        return ReasonResult(probs=probs, confidence=confidence, rationale=rationale)

    def self_evaluate(self, prediction: Prediction, result: MatchResult) -> str:
        hit = "correct" if prediction.pick is result.outcome else "wrong"
        return (
            f"Prediction was {hit}: picked {prediction.pick.value} "
            f"(p={prediction.probs[prediction.pick]:.2f}), "
            f"actual {result.outcome.value} "
            f"({result.home_goals}-{result.away_goals})."
        )
