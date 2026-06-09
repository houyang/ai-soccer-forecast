from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from soccer.models import Evaluation, MatchResult, Outcome, Prediction

_EPS = 1e-15


def brier_score(probs: dict[Outcome, float], actual: Outcome) -> float:
    return sum((probs[o] - (1.0 if o is actual else 0.0)) ** 2 for o in Outcome)


def log_loss(probs: dict[Outcome, float], actual: Outcome) -> float:
    return -math.log(min(max(probs[actual], _EPS), 1.0))


def beat_market(
    probs: dict[Outcome, float],
    market: dict[Outcome, float] | None,
    actual: Outcome,
) -> bool:
    if market is None:
        return False
    return probs[actual] > market[actual]


def score(
    prediction: Prediction,
    result: MatchResult,
    self_critique: str,
    evaluated_at: datetime,
) -> Evaluation:
    actual = result.outcome
    return Evaluation(
        prediction_id=prediction.id,
        result=result,
        correct=prediction.pick is actual,
        brier=brier_score(prediction.probs, actual),
        log_loss=log_loss(prediction.probs, actual),
        beat_market=beat_market(prediction.probs, prediction.market_probs, actual),
        self_critique=self_critique,
        evaluated_at=evaluated_at,
    )


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    observed: float  # fraction correct among predictions whose confidence is in-band


def calibration_bins(
    predictions: list[Prediction],
    outcomes: list[Outcome],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    width = 1.0 / n_bins
    bins: list[CalibrationBin] = []
    for i in range(n_bins):
        lower, upper = i * width, (i + 1) * width
        members = [
            (p, o)
            for p, o in zip(predictions, outcomes, strict=True)
            if lower <= p.confidence < upper or (i == n_bins - 1 and p.confidence == 1.0)
        ]
        count = len(members)
        observed = sum(1 for p, o in members if p.pick is o) / count if count else 0.0
        bins.append(CalibrationBin(lower=lower, upper=upper, count=count, observed=observed))
    return bins
