"""Evaluation harness for settled predictions."""

from __future__ import annotations

from dataclasses import dataclass

from soccer.models import Outcome, PredictionRecord
from soccer.storage import PredictionLog


@dataclass(frozen=True)
class EvaluationSummary:
    settled_count: int
    accuracy: float
    average_confidence: float
    brier_score: float


@dataclass(frozen=True)
class EvaluationHarness:
    prediction_log: PredictionLog

    def evaluate(self) -> EvaluationSummary:
        settled = [record for record in self.prediction_log.list_records() if record.result]
        if not settled:
            return EvaluationSummary(0, 0.0, 0.0, 0.0)

        correct = 0
        total_confidence = 0.0
        total_brier = 0.0
        for record in settled:
            result = record.result
            if result is None:
                continue
            if record.prediction.outcome == result.outcome:
                correct += 1
            total_confidence += record.prediction.confidence
            total_brier += self._brier_score(record)

        return EvaluationSummary(
            settled_count=len(settled),
            accuracy=correct / len(settled),
            average_confidence=total_confidence / len(settled),
            brier_score=total_brier / len(settled),
        )

    @staticmethod
    def _brier_score(record: PredictionRecord) -> float:
        if record.result is None:
            raise ValueError("Cannot score a prediction without a result")
        actual = record.result.outcome
        return sum(
            (
                record.prediction.probabilities.get(outcome, 0.0)
                - (1.0 if outcome == actual else 0.0)
            )
            ** 2
            for outcome in Outcome
        )
