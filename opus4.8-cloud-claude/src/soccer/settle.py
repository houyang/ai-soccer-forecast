from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from soccer.evaluation import score
from soccer.models import Evaluation
from soccer.reasoning.base import Reasoner
from soccer.registry import ToolRegistry
from soccer.store import PredictionStore


def _utc_now() -> datetime:
    return datetime.now(UTC)


def settle(
    store: PredictionStore,
    registry: ToolRegistry,
    reasoner: Reasoner,
    clock: Callable[[], datetime] = _utc_now,
) -> list[Evaluation]:
    new_evaluations: list[Evaluation] = []
    for prediction in store.pending():
        result = registry.results.get_result(prediction.match_ref)
        if result is None:
            continue
        critique = reasoner.self_evaluate(prediction, result)
        evaluation = score(prediction, result, critique, evaluated_at=clock())
        store.append_result(result)
        store.append_evaluation(evaluation)
        new_evaluations.append(evaluation)
    return new_evaluations
