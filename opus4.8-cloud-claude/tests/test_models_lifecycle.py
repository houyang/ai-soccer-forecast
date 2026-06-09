# tests/test_models_lifecycle.py
from datetime import UTC, datetime

import pytest

from soccer.models import (
    Evaluation,
    MatchRef,
    MatchResult,
    Outcome,
    Prediction,
    evaluation_from_dict,
    evaluation_to_dict,
    prediction_from_dict,
    prediction_to_dict,
    result_from_dict,
    result_to_dict,
)

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=UTC)
REF = MatchRef(
    id="m1", competition="UCL", home="A", away="B", kickoff=KICK, venue_id="v1", season="2025-26"
)


def _pred() -> Prediction:
    return Prediction(
        id="abc123",
        match_ref=REF,
        created_at=KICK,
        probs={Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2},
        pick=Outcome.HOME,
        confidence=0.5,
        rationale="strong home form",
        market_probs={Outcome.HOME: 0.45, Outcome.DRAW: 0.3, Outcome.AWAY: 0.25},
        dossier_digest="deadbeef",
        reasoner_name="fake",
    )


def test_prediction_round_trip() -> None:
    p = _pred()
    assert prediction_from_dict(prediction_to_dict(p)) == p


def test_result_outcome_property() -> None:
    r = MatchResult(match_id="m1", home_goals=2, away_goals=1, status="finished", source="fixture")
    assert r.outcome is Outcome.HOME


def test_result_outcome_away() -> None:
    r = MatchResult(match_id="m1", home_goals=1, away_goals=2, status="finished", source="fixture")
    assert r.outcome is Outcome.AWAY


def test_prediction_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError):
        Prediction(
            id="abc123",
            match_ref=REF,
            created_at=KICK,
            probs={Outcome.HOME: 0.5, Outcome.DRAW: 0.3, Outcome.AWAY: 0.2},
            pick=Outcome.HOME,
            confidence=1.5,
            rationale="strong home form",
            market_probs=None,
            dossier_digest="deadbeef",
            reasoner_name="fake",
        )


def test_prediction_rejects_probs_not_summing_to_one() -> None:
    with pytest.raises(ValueError):
        Prediction(
            id="abc123",
            match_ref=REF,
            created_at=KICK,
            probs={Outcome.HOME: 0.5, Outcome.DRAW: 0.4, Outcome.AWAY: 0.4},
            pick=Outcome.HOME,
            confidence=0.5,
            rationale="strong home form",
            market_probs=None,
            dossier_digest="deadbeef",
            reasoner_name="fake",
        )


def test_result_round_trip() -> None:
    r = MatchResult(match_id="m1", home_goals=1, away_goals=1, status="finished", source="fixture")
    assert result_from_dict(result_to_dict(r)) == r
    assert r.outcome is Outcome.DRAW


def test_evaluation_round_trip() -> None:
    r = MatchResult(match_id="m1", home_goals=0, away_goals=2, status="finished", source="fixture")
    e = Evaluation(
        prediction_id="abc123",
        result=r,
        correct=False,
        brier=0.5,
        log_loss=1.2,
        beat_market=False,
        self_critique="overrated home",
        evaluated_at=KICK,
    )
    assert evaluation_from_dict(evaluation_to_dict(e)) == e
