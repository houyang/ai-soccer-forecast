from __future__ import annotations

from dataclasses import dataclass

from soccer.agent import PredictionAgent
from soccer.evaluation import CalibrationBin, brier_score, calibration_bins, log_loss
from soccer.models import MatchRef, MatchResult, Outcome, Prediction
from soccer.registry import ToolRegistry


@dataclass(frozen=True)
class Scenario:
    name: str
    registry: ToolRegistry
    matches: list[MatchRef]
    results: dict[str, MatchResult]


@dataclass(frozen=True)
class MatchScore:
    match_id: str
    pick: Outcome
    actual: Outcome
    correct: bool
    brier: float
    log_loss: float


@dataclass(frozen=True)
class MarketBaseline:
    mean_brier: float
    mean_log_loss: float


@dataclass(frozen=True)
class EvalReport:
    scenario: str
    n: int
    accuracy: float
    mean_brier: float
    mean_log_loss: float
    calibration: list[CalibrationBin]
    market_baseline: MarketBaseline
    edge_vs_market: float
    per_match: list[MatchScore]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_scenario(scenario: Scenario, agent: PredictionAgent) -> EvalReport:
    per_match: list[MatchScore] = []
    outcomes: list[Outcome] = []
    predictions: list[Prediction] = []
    market_brier: list[float] = []
    market_log_loss: list[float] = []

    for match in scenario.matches:
        actual = scenario.results[match.id].outcome
        prediction = agent.predict(match)
        predictions.append(prediction)
        outcomes.append(actual)
        per_match.append(
            MatchScore(
                match_id=match.id,
                pick=prediction.pick,
                actual=actual,
                correct=prediction.pick is actual,
                brier=brier_score(prediction.probs, actual),
                log_loss=log_loss(prediction.probs, actual),
            )
        )
        if prediction.market_probs is not None:
            market_brier.append(brier_score(prediction.market_probs, actual))
            market_log_loss.append(log_loss(prediction.market_probs, actual))

    n = len(scenario.matches)
    mean_log = _mean([s.log_loss for s in per_match])
    market_mean_log = _mean(market_log_loss)
    return EvalReport(
        scenario=scenario.name,
        n=n,
        accuracy=_mean([1.0 if s.correct else 0.0 for s in per_match]),
        mean_brier=_mean([s.brier for s in per_match]),
        mean_log_loss=mean_log,
        calibration=calibration_bins(predictions, outcomes),
        market_baseline=MarketBaseline(
            mean_brier=_mean(market_brier), mean_log_loss=market_mean_log
        ),
        edge_vs_market=mean_log - market_mean_log,
        per_match=per_match,
    )
