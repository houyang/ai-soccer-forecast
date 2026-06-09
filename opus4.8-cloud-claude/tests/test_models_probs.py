# tests/test_models_probs.py
import pytest

from soccer.models import Outcome, normalize_probs, validate_probs


def test_normalize_scales_to_one() -> None:
    out = normalize_probs({Outcome.HOME: 2.0, Outcome.DRAW: 1.0, Outcome.AWAY: 1.0})
    assert out[Outcome.HOME] == pytest.approx(0.5)
    assert sum(out.values()) == pytest.approx(1.0)


def test_normalize_rejects_nonpositive_total() -> None:
    with pytest.raises(ValueError):
        normalize_probs({Outcome.HOME: 0.0, Outcome.DRAW: 0.0, Outcome.AWAY: 0.0})


def test_normalize_rejects_negative_value_with_positive_total() -> None:
    with pytest.raises(ValueError):
        normalize_probs({Outcome.HOME: -1.0, Outcome.DRAW: 3.0, Outcome.AWAY: 1.0})


def test_validate_requires_all_three_outcomes() -> None:
    with pytest.raises(ValueError):
        validate_probs({Outcome.HOME: 0.5, Outcome.DRAW: 0.5})


def test_validate_requires_sum_one() -> None:
    with pytest.raises(ValueError):
        validate_probs({Outcome.HOME: 0.5, Outcome.DRAW: 0.4, Outcome.AWAY: 0.4})
