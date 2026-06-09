from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from soccer.dossier import build_dossier, dossier_digest
from soccer.models import MatchRef, Prediction, make_prediction_id
from soccer.reasoning.base import Reasoner
from soccer.registry import ToolRegistry


def _utc_now() -> datetime:
    return datetime.now(UTC)


class PredictionAgent:
    def __init__(
        self,
        registry: ToolRegistry,
        reasoner: Reasoner,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._registry = registry
        self._reasoner = reasoner
        self._clock = clock

    def predict(self, match: MatchRef) -> Prediction:
        dossier = build_dossier(match, self._registry)
        result = self._reasoner.predict(dossier)
        created_at = self._clock()
        pick = max(result.probs, key=lambda k: result.probs[k])
        market = dossier.odds.implied_probs if dossier.odds is not None else None
        return Prediction(
            id=make_prediction_id(match.id, created_at),
            match_ref=match,
            created_at=created_at,
            probs=result.probs,
            pick=pick,
            confidence=result.confidence,
            rationale=result.rationale,
            market_probs=market,
            dossier_digest=dossier_digest(dossier),
            reasoner_name=self._reasoner.name,
        )
