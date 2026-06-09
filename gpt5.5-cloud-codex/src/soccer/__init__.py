"""Soccer prediction agent package."""

from soccer.agent import PredictionAgent
from soccer.evaluation import EvaluationHarness
from soccer.models import MatchRequest, MatchResult, Outcome, Prediction

__all__ = [
    "EvaluationHarness",
    "MatchRequest",
    "MatchResult",
    "Outcome",
    "Prediction",
    "PredictionAgent",
]
