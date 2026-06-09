import math
from datetime import UTC, datetime

import pytest

from soccer.evaluation import (
    beat_market,
    brier_score,
    calibration_bins,
    log_loss,
    score,
)
from soccer.models import MatchRef, MatchResult, Outcome, Prediction

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=UTC)
REF = MatchRef(
    id="m1",
    competition="UCL",
    home="A",
    away="B",
    kickoff=KICK,
    venue_id="v1",
    season="2025-26",
)


def _pred(probs: dict[Outcome, float], market: dict[Outcome, float] | None = None) -> Prediction:
    pick = max(probs, key=lambda k: probs[k])
    return Prediction(
        id="p1",
        match_ref=REF,
        created_at=KICK,
        probs=probs,
        pick=pick,
        confidence=probs[pick],
        rationale="r",
        market_probs=market,
        dossier_digest="d",
        reasoner_name="fake",
    )


HOME_RESULT = MatchResult(
    match_id="m1", home_goals=2, away_goals=0, status="finished", source="fixture"
)


def test_brier_perfect_prediction_is_zero() -> None:
    probs = {Outcome.HOME: 1.0, Outcome.DRAW: 0.0, Outcome.AWAY: 0.0}
    assert brier_score(probs, Outcome.HOME) == pytest.approx(0.0)


def test_brier_known_value() -> None:
    probs = {Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2}
    # (0.5-1)^2 + (0.3-0)^2 + (0.2-0)^2 = 0.25 + 0.09 + 0.04 = 0.38
    assert brier_score(probs, Outcome.HOME) == pytest.approx(0.38)


def test_log_loss_known_value() -> None:
    probs = {Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2}
    assert log_loss(probs, Outcome.HOME) == pytest.approx(-math.log(0.5))


def test_beat_market_true_when_model_more_confident_in_actual() -> None:
    model = {Outcome.HOME: 0.6, Outcome.DRAW: 0.25, Outcome.AWAY: 0.15}
    market = {Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2}
    assert beat_market(model, market, Outcome.HOME) is True


def test_beat_market_false_without_market() -> None:
    model = {Outcome.HOME: 0.6, Outcome.DRAW: 0.25, Outcome.AWAY: 0.15}
    assert beat_market(model, None, Outcome.HOME) is False


def test_score_builds_full_evaluation() -> None:
    pred = _pred(
        {Outcome.HOME: 0.6, Outcome.DRAW: 0.25, Outcome.AWAY: 0.15},
        market={Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2},
    )
    ev = score(pred, HOME_RESULT, "good call", evaluated_at=KICK)
    assert ev.correct is True
    assert ev.beat_market is True
    assert ev.self_critique == "good call"
    assert ev.brier == pytest.approx(0.6**2 - 2 * 0.6 + 1 + 0.25**2 + 0.15**2)


def test_calibration_bins_group_by_confidence() -> None:
    preds = [
        _pred({Outcome.HOME: 0.88, Outcome.DRAW: 0.07, Outcome.AWAY: 0.05}),
        _pred({Outcome.HOME: 0.85, Outcome.DRAW: 0.1, Outcome.AWAY: 0.05}),
    ]
    outcomes = [Outcome.HOME, Outcome.AWAY]  # one hit, one miss
    bins = calibration_bins(preds, outcomes, n_bins=10)
    high = [b for b in bins if b.count > 0 and b.lower >= 0.8][0]
    assert high.count == 2
    assert high.observed == pytest.approx(0.5)
