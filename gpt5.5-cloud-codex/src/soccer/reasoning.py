"""Deterministic matchup reasoning for the first agent version."""

from __future__ import annotations

from dataclasses import dataclass

from soccer.models import MatchEvidence, Outcome, Prediction


@dataclass(frozen=True)
class MatchupReasoner:
    """Turn structured evidence into a prediction.

    This is a transparent baseline, not a production-grade model. It is intentionally simple
    so the eval harness can measure whether future models are improving.
    """

    bookmaker_weight: float = 0.55

    def predict(self, evidence: MatchEvidence) -> Prediction:
        probabilities = self._baseline_probabilities(evidence)
        outcome = max(probabilities, key=lambda item: probabilities[item])
        confidence = probabilities[outcome]
        rationale = self._rationale(evidence, outcome, confidence, probabilities)
        return Prediction(
            match_id=evidence.request.match_id,
            outcome=outcome,
            confidence=round(confidence, 3),
            rationale=rationale,
            probabilities={key: round(value, 3) for key, value in probabilities.items()},
        )

    def _baseline_probabilities(self, evidence: MatchEvidence) -> dict[Outcome, float]:
        scores = {
            Outcome.HOME_WIN: 0.34,
            Outcome.DRAW: 0.28,
            Outcome.AWAY_WIN: 0.34,
        }

        form_edge = (
            evidence.home_form.points_per_match
            - evidence.away_form.points_per_match
            + evidence.home_form.goal_difference_per_match
            - evidence.away_form.goal_difference_per_match
        )
        scores[Outcome.HOME_WIN] += form_edge * 0.08
        scores[Outcome.AWAY_WIN] -= form_edge * 0.08

        injury_edge = evidence.away_injuries.impact_count - evidence.home_injuries.impact_count
        scores[Outcome.HOME_WIN] += injury_edge * 0.025
        scores[Outcome.AWAY_WIN] -= injury_edge * 0.025

        if (
            not evidence.request.neutral_site
            and evidence.venue.home_team == evidence.request.home_team
        ):
            scores[Outcome.HOME_WIN] += 0.04
            scores[Outcome.AWAY_WIN] -= 0.02

        h2h = evidence.head_to_head
        if h2h.meetings:
            scores[Outcome.HOME_WIN] += (h2h.home_team_wins / h2h.meetings - 0.33) * 0.08
            scores[Outcome.AWAY_WIN] += (h2h.away_team_wins / h2h.meetings - 0.33) * 0.08
            scores[Outcome.DRAW] += (h2h.draws / h2h.meetings - 0.28) * 0.04

        if evidence.weather.wind_kph >= 35 or evidence.weather.precipitation_mm >= 8:
            scores[Outcome.DRAW] += 0.03

        model_probabilities = self._normalize(scores)
        market_probabilities = self._market_probabilities(evidence)
        if market_probabilities is None:
            return model_probabilities

        return self._normalize(
            {
                outcome: (model_probabilities[outcome] * (1 - self.bookmaker_weight))
                + (market_probabilities[outcome] * self.bookmaker_weight)
                for outcome in Outcome
            }
        )

    def _market_probabilities(self, evidence: MatchEvidence) -> dict[Outcome, float] | None:
        if not evidence.odds:
            return None
        averaged = dict.fromkeys(Outcome, 0.0)
        for quote in evidence.odds:
            for outcome, probability in quote.implied_probabilities().items():
                averaged[outcome] += probability
        return {
            outcome: probability / len(evidence.odds) for outcome, probability in averaged.items()
        }

    @staticmethod
    def _normalize(scores: dict[Outcome, float]) -> dict[Outcome, float]:
        floored = {outcome: max(score, 0.01) for outcome, score in scores.items()}
        total = sum(floored.values())
        return {outcome: score / total for outcome, score in floored.items()}

    @staticmethod
    def _rationale(
        evidence: MatchEvidence,
        outcome: Outcome,
        confidence: float,
        probabilities: dict[Outcome, float],
    ) -> str:
        market_note = "no bookmaker odds were available"
        if evidence.odds:
            market_favorite = max(probabilities, key=lambda item: probabilities[item])
            market_note = f"market-adjusted probabilities favor {market_favorite.value}"

        return (
            f"Prediction is {outcome.value} at {confidence:.0%} confidence. "
            f"{evidence.request.home_team} form: "
            f"{evidence.home_form.wins}-{evidence.home_form.draws}-{evidence.home_form.losses}, "
            f"GD {evidence.home_form.goals_for - evidence.home_form.goals_against}; "
            f"{evidence.request.away_team} form: "
            f"{evidence.away_form.wins}-{evidence.away_form.draws}-{evidence.away_form.losses}, "
            f"GD {evidence.away_form.goals_for - evidence.away_form.goals_against}. "
            f"Injury counts are {evidence.home_injuries.impact_count} vs "
            f"{evidence.away_injuries.impact_count}. "
            f"Head-to-head: {evidence.head_to_head.summary}. "
            f"Venue is {evidence.venue.name}, {evidence.venue.city}; weather is "
            f"{evidence.weather.summary}. {market_note}."
        )
