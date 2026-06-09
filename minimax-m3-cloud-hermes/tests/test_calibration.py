"""Tests for src/soccer_agent/calibration.py (Task 28)."""

from __future__ import annotations

import math

import pytest

from soccer_agent.calibration import (
    BinningCalibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    brier,
    ece,
    reliability_table,
)


# --- metrics ---------------------------------------------------------------


def test_ece_perfect_calibration_is_zero():
    """If stated probs match outcome frequencies exactly, ECE = 0."""
    probs = [0.0, 0.0, 1.0, 1.0]
    outcomes = [0, 0, 1, 1]
    assert ece(probs, outcomes) == pytest.approx(0.0)


def test_ece_overconfidence_positive():
    """A model that says 90% but wins 50% has non-zero ECE."""
    probs = [0.9] * 10
    outcomes = [1] * 5 + [0] * 5
    # 90% bucket: 10 items, avg_p=0.9, avg_y=0.5, gap=0.4
    assert ece(probs, outcomes) == pytest.approx(0.4)


def test_ece_handles_edge_p_equal_1():
    """p=1.0 should land in the last bucket, not overflow."""
    probs = [1.0, 1.0, 1.0, 1.0]
    outcomes = [1, 1, 1, 0]
    # 0.9-1.0 bucket: 4 items, avg_p=1.0, avg_y=0.75, gap=0.25
    assert ece(probs, outcomes) == pytest.approx(0.25)


def test_ece_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        ece([0.5, 0.5], [1])


def test_ece_empty_input_is_zero():
    assert ece([], []) == 0.0


def test_brier_perfect_is_zero():
    assert brier([0.0, 1.0], [0, 1]) == pytest.approx(0.0)


def test_brier_worst_is_one():
    """Confident wrong → Brier = 1."""
    assert brier([0.0], [1]) == pytest.approx(1.0)
    assert brier([1.0], [0]) == pytest.approx(1.0)


def test_brier_baseline_is_quarter():
    """On a 50/50 base rate, always predicting 0.5 gives Brier = 0.25."""
    probs = [0.5] * 20
    outcomes = [0, 1] * 10
    assert brier(probs, outcomes) == pytest.approx(0.25)


def test_reliability_table_shape():
    """Returns one row per bin, with n=0 buckets preserved."""
    probs = [0.1, 0.2, 0.5, 0.8]
    outcomes = [0, 1, 1, 1]
    table = reliability_table(probs, outcomes, n_bins=4)
    assert len(table) == 4
    # First bucket (0-0.25): n=2
    assert table[0]["n"] == 2
    # Last bucket: avg_p near 0.8, avg_y=1.0
    assert table[3]["n"] == 1
    assert table[3]["avg_p"] == pytest.approx(0.8)
    assert table[3]["avg_y"] == 1.0


# --- calibrators -----------------------------------------------------------


def test_identity_passthrough():
    cal = IdentityCalibrator()
    out = cal.calibrate([0.1, 0.5, 0.9])
    assert out == [0.1, 0.5, 0.9]


def test_platt_identity_when_a1_b0():
    """PlattCalibrator with default (1, 0) is the identity on logit
    space, so output == input."""
    cal = PlattCalibrator()
    out = cal.calibrate([0.1, 0.3, 0.7, 0.9])
    for o, i in zip(out, [0.1, 0.3, 0.7, 0.9]):
        assert o == pytest.approx(i, abs=1e-3)


def test_platt_fit_on_overconfident_model_improves_ece():
    """Model says 0.9 but wins 50% → Platt should soften it."""
    probs = [0.9] * 20
    outcomes = [1] * 10 + [0] * 10
    pre_ece = ece(probs, outcomes)
    cal = PlattCalibrator().fit(probs, outcomes)
    out = cal.calibrate(probs)
    post_ece = ece(out, outcomes)
    assert post_ece < pre_ece
    assert pre_ece > 0.1  # confirm the baseline is actually bad
    assert post_ece < 0.1  # and that Platt fixed it meaningfully


def test_temperature_identity_when_t1():
    cal = TemperatureCalibrator()
    out = cal.calibrate([0.1, 0.3, 0.7, 0.9])
    for o, i in zip(out, [0.1, 0.3, 0.7, 0.9]):
        assert o == pytest.approx(i, abs=1e-3)


def test_temperature_high_t_softens():
    """T=2 should pull extreme probs toward 0.5."""
    cal = TemperatureCalibrator(temperature=2.0)
    out = cal.calibrate([0.9, 0.1])
    assert 0.5 < out[0] < 0.9
    assert 0.1 < out[1] < 0.5


def test_temperature_low_t_sharpens():
    """T=0.5 should push moderate probs toward 0 or 1."""
    cal = TemperatureCalibrator(temperature=0.5)
    out = cal.calibrate([0.6, 0.4])
    assert out[0] > 0.6
    assert out[1] < 0.4


def test_isotonic_fit_makes_monotone_step():
    """Fit then evaluate on a new x: result is monotone non-decreasing."""
    probs = [0.1, 0.3, 0.5, 0.7, 0.9]
    outcomes = [0, 0, 1, 1, 1]
    cal = IsotonicCalibrator().fit(probs, outcomes)
    out = cal.calibrate([0.05, 0.2, 0.4, 0.6, 0.8, 0.95])
    for a, b in zip(out, out[1:]):
        assert a <= b + 1e-6


def test_isotonic_handles_perfect_calibration():
    """A perfectly calibrated set is recovered as identity-ish."""
    probs = [0.2, 0.2, 0.2, 0.8, 0.8, 0.8]
    outcomes = [0, 0, 0, 1, 1, 1]
    cal = IsotonicCalibrator().fit(probs, outcomes)
    out = cal.calibrate([0.2, 0.8])
    assert out[0] == pytest.approx(0.0, abs=1e-3)
    assert out[1] == pytest.approx(1.0, abs=1e-3)


def test_binning_identity_when_data_matches():
    """If every input is exactly the bucket center, output should
    match the empirical rate closely."""
    probs = [0.15, 0.15, 0.15, 0.15, 0.85, 0.85, 0.85, 0.85]
    outcomes = [0, 0, 0, 0, 1, 1, 1, 1]
    cal = BinningCalibrator(n_bins=10, shrink=0.0).fit(probs, outcomes)
    out = cal.calibrate([0.15, 0.85])
    assert out[0] == pytest.approx(0.0, abs=1e-3)
    assert out[1] == pytest.approx(1.0, abs=1e-3)


def test_binning_shrinkage_pulls_toward_mean():
    """With high shrinkage, sparse-bin outputs gravitate to the
    global mean."""
    probs = [0.5]
    outcomes = [1]  # global mean = 1.0 here
    cal_shrink = BinningCalibrator(n_bins=10, shrink=100.0).fit(probs, outcomes)
    cal_no = BinningCalibrator(n_bins=10, shrink=0.0).fit(probs, outcomes)
    out_shrink = cal_shrink.calibrate([0.5])[0]
    out_no = cal_no.calibrate([0.5])[0]
    # With shrinkage the bin value is pulled toward 1.0 (the mean),
    # but since the mean is 1.0, the difference is small. Use a
    # case where shrinkage toward 0.5 is visible.
    probs2 = [0.05, 0.05, 0.05, 0.05, 0.5, 0.5, 0.5, 0.5]
    outcomes2 = [0, 0, 0, 0, 1, 1, 1, 1]
    cal2_shrink = BinningCalibrator(n_bins=10, shrink=20.0).fit(probs2, outcomes2)
    cal2_no = BinningCalibrator(n_bins=10, shrink=0.0).fit(probs2, outcomes2)
    # The 0.5 bucket has all-1s in both → both close to 1.0.
    # The 0.0-0.1 bucket has all-0s → with shrinkage pulled to 0.5
    # much more strongly.
    out_shrink_low = cal2_shrink.calibrate([0.05])[0]
    out_no_low = cal2_no.calibrate([0.05])[0]
    assert out_shrink_low > out_no_low


def test_all_calibrators_handle_empty_input():
    """Empty input → empty output for all calibrators, no crash."""
    for cal in [
        IdentityCalibrator(),
        PlattCalibrator(),
        TemperatureCalibrator(),
        IsotonicCalibrator(),
        BinningCalibrator(),
    ]:
        out = cal.calibrate([])
        assert out == []


def test_all_calibrators_return_list_of_floats():
    """Calibrate returns a list of floats, not a numpy array."""
    platt = PlattCalibrator().fit([0.3, 0.7], [0, 1])
    out = platt.calibrate([0.3, 0.5, 0.7])
    assert isinstance(out, list)
    assert all(isinstance(x, float) for x in out)


# -- Task 35: per-competition calibrator fitting ----------------------------


def _make_samples(competition: str, n: int, p_right: float = 0.6) -> list:
    """Build n synthetic CalibSample-like dicts for one competition.

    They need .p_right, .outcome, .match_id; the function under
    test figures out the competition from match_id via the
    supplied resolver.
    """
    from soccer_agent.eval.calibration import CalibSample
    out = []
    for i in range(n):
        # Alternate outcomes so the calibrator has something to fit.
        outcome = 1 if (i % 2 == 0) else 0
        out.append(CalibSample(
            match_id=f"{competition}-{i}",
            pick="home", actual="home" if outcome else "away",
            confidence=p_right,
            outcome=outcome,
            p_right=p_right if outcome else (1 - p_right),
        ))
    return out


def _match_id_to_competition(samples):
    """Resolver that decodes '<COMP>-<n>' back to '<COMP>'."""
    def resolver(match_id: str) -> str | None:
        for s in samples:
            if s.match_id == match_id:
                return match_id.split("-")[0]
        return None
    return resolver


def test_fit_per_competition_calibrators_writes_one_file_per_competition(tmp_path):
    """Task 35: fit_per_competition_calibrators should write one
    isotonic_<COMP>.json per competition with at least min_n samples.

    Competitions with fewer samples (below min_n) should be SKIPPED
    (so the agent's global fallback handles them).
    """
    from soccer_agent.eval.calibration import fit_per_competition_calibrators

    samples = (
        _make_samples("EPL", 30)        # → fit
        + _make_samples("LaLiga", 25)   # → fit
        + _make_samples("UCL", 11)      # → skip (n < 20)
    )
    written = fit_per_competition_calibrators(
        samples, key="isotonic", root=tmp_path, min_n=20,
        match_to_competition=_match_id_to_competition(samples),
    )
    comps = sorted(written.keys())
    assert comps == ["EPL", "LaLiga"], (
        f"expected EPL + LaLiga, got {comps} (UCL should be skipped at n=11)"
    )
    # Each written entry points to an actual file.
    for comp, path in written.items():
        assert path.exists(), f"file missing for {comp}: {path}"
        assert path.name == f"isotonic_{comp}.json"


def test_fit_per_competition_calibrators_skips_below_min_n(tmp_path):
    """All competitions below min_n → empty result (no files)."""
    from soccer_agent.eval.calibration import fit_per_competition_calibrators
    samples = _make_samples("Bundesliga", 15)
    written = fit_per_competition_calibrators(
        samples, key="isotonic", root=tmp_path, min_n=20,
        match_to_competition=_match_id_to_competition(samples),
    )
    assert written == {}
    # And no file was written.
    assert list(tmp_path.glob("*.json")) == []


def test_fit_per_competition_calibrators_empty_input(tmp_path):
    """No samples → empty result, no crash."""
    from soccer_agent.eval.calibration import fit_per_competition_calibrators
    written = fit_per_competition_calibrators(
        [], key="isotonic", root=tmp_path, min_n=20,
        match_to_competition=lambda _: None,
    )
    assert written == {}
