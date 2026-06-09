"""calibration_store: persist + load fitted calibrators.

These tests pin the contract that the live predict() path relies on:
- save then load round-trips the calibrator's behavior
- load returns None for a missing key (not raises)
- save rejects path traversal
- save rejects unknown calibrator classes
- fit_isotonic stores the right metadata
"""

from __future__ import annotations

from pathlib import Path

import pytest

from soccer_agent.calibration import (
    BinningCalibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
)
from soccer_agent.calibration_store import (
    _CALIBRATORS,
    fit_isotonic,
    load_calibrator,
    save_calibrator,
)


# --- round-trip tests ----------------------------------------------------

@pytest.mark.parametrize("cls", [
    IdentityCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
    IsotonicCalibrator,
    BinningCalibrator,
])
def test_save_then_load_round_trips(tmp_path: Path, cls):
    """After save -> load, the loaded calibrator returns the same values
    as the original when given the same inputs."""
    probs_in = [0.1, 0.3, 0.5, 0.7, 0.9]
    outs = [0, 0, 0, 1, 1]

    if cls is IdentityCalibrator:
        original = cls()
    elif cls is PlattCalibrator:
        original = cls().fit(probs_in, outs)
    elif cls is TemperatureCalibrator:
        original = cls().fit(probs_in, outs)
    elif cls is IsotonicCalibrator:
        original = cls().fit(probs_in, outs)
    elif cls is BinningCalibrator:
        original = cls().fit(probs_in, outs)

    save_calibrator(
        original, key="test_cal", root=tmp_path,
        competition="EPL", n_samples=5, ece=0.1, brier=0.2,
    )
    loaded = load_calibrator(key="test_cal", root=tmp_path)
    assert loaded is not None, "load returned None after save"
    # Same class.
    assert type(loaded).__name__ == cls.__name__
    # Same outputs (round-trip).
    assert loaded.calibrate(probs_in) == pytest.approx(
        original.calibrate(probs_in), abs=1e-6
    )


def test_load_missing_returns_none(tmp_path: Path):
    """A missing key is the common case. Don't raise, return None."""
    assert load_calibrator(key="never_saved", root=tmp_path) is None


# --- safety tests --------------------------------------------------------

def test_save_rejects_path_traversal(tmp_path: Path):
    """Keys with /, .., or leading dots are rejected. Defense in depth:
    even if the caller has a bug, the file never lands outside root."""
    cal = IdentityCalibrator()
    with pytest.raises(ValueError, match="invalid calibrator key"):
        save_calibrator(cal, key="../etc/passwd", root=tmp_path,
                        competition="EPL", n_samples=0, ece=0, brier=0)
    with pytest.raises(ValueError, match="invalid calibrator key"):
        save_calibrator(cal, key="a/b", root=tmp_path,
                        competition="EPL", n_samples=0, ece=0, brier=0)
    with pytest.raises(ValueError, match="invalid calibrator key"):
        save_calibrator(cal, key=".hidden", root=tmp_path,
                        competition="EPL", n_samples=0, ece=0, brier=0)


def test_save_rejects_unknown_class(tmp_path: Path):
    """If you pass a calibrator class that isn't registered, fail loud
    at save time. Better than discovering it 3 months later at load."""
    class FakeCalibrator:
        def calibrate(self, probs): return list(probs)
    with pytest.raises(ValueError, match="unknown calibrator class"):
        save_calibrator(FakeCalibrator(), key="x", root=tmp_path,
                        competition="EPL", n_samples=0, ece=0, brier=0)


# --- fit_isotonic --------------------------------------------------------

def test_fit_isotonic_persists_metadata(tmp_path: Path):
    """fit_isotonic stores the calibrator AND the metadata we want to
    show in the dashboard (n, ece, brier, competition)."""
    # Synthetic 1D data: confidence is wrong-ish in two regimes.
    samples = [
        (0.1, 1), (0.2, 1), (0.3, 0), (0.4, 1),  # low conf: 3/4 right
        (0.6, 0), (0.7, 1), (0.8, 1), (0.9, 0),  # high conf: 2/4 right
    ]
    cal, meta = fit_isotonic(samples, key="iso_epl", root=tmp_path, competition="EPL")
    assert isinstance(cal, IsotonicCalibrator)
    assert meta["competition"] == "EPL"
    assert meta["n_samples"] == 8
    assert 0.0 <= meta["ece"] <= 1.0
    assert 0.0 <= meta["brier"] <= 1.0
    # And we can load it back.
    loaded = load_calibrator(key="iso_epl", root=tmp_path)
    assert loaded is not None
    assert type(loaded).__name__ == "IsotonicCalibrator"


def test_all_known_calibrators_are_registered():
    """Every class in the registry must be in the save/load registry.
    This catches the case where someone adds a new calibrator class
    in calibration.py but forgets to register it in the store."""
    # We can't introspect the source module for Calibrator subclasses
    # without scanning, but we CAN assert that the names we depend on
    # are present.
    expected = {"IdentityCalibrator", "PlattCalibrator",
                "TemperatureCalibrator", "IsotonicCalibrator",
                "BinningCalibrator"}
    actual = {cls.__name__ for cls in _CALIBRATORS.values()}
    assert actual == expected
