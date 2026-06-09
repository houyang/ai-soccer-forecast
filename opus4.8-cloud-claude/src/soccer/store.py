from __future__ import annotations

import json
from pathlib import Path

from soccer.models import (
    Evaluation,
    MatchResult,
    Prediction,
    evaluation_from_dict,
    evaluation_to_dict,
    prediction_from_dict,
    prediction_to_dict,
    result_from_dict,
    result_to_dict,
)


class PredictionStore:
    def __init__(
        self,
        predictions_path: Path,
        results_path: Path,
        evaluations_path: Path,
    ) -> None:
        self._predictions = Path(predictions_path)
        self._results = Path(results_path)
        self._evaluations = Path(evaluations_path)

    @staticmethod
    def _append(path: Path, record: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    @staticmethod
    def _read(path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def append_prediction(self, prediction: Prediction) -> None:
        self._append(self._predictions, prediction_to_dict(prediction))

    def append_result(self, result: MatchResult) -> None:
        self._append(self._results, result_to_dict(result))

    def append_evaluation(self, evaluation: Evaluation) -> None:
        self._append(self._evaluations, evaluation_to_dict(evaluation))

    def load_predictions(self) -> list[Prediction]:
        return [prediction_from_dict(r) for r in self._read(self._predictions)]

    def load_results(self) -> list[MatchResult]:
        return [result_from_dict(r) for r in self._read(self._results)]

    def load_evaluations(self) -> list[Evaluation]:
        return [evaluation_from_dict(r) for r in self._read(self._evaluations)]

    def pending(self) -> list[Prediction]:
        evaluated = {e.prediction_id for e in self.load_evaluations()}
        return [p for p in self.load_predictions() if p.id not in evaluated]
