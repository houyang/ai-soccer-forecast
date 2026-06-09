"""Calibrator store: load/save fitted calibrators by competition.

A fitted calibrator is the result of running
`soccer_agent.eval.calibration` on a labeled eval set. We want
to *use* that calibrator at predict-time so the agent's
`final_confidence` is already calibrated when it lands in the DB
(the raw confidence is also stored for comparison).

The store is just JSON files under `data/calibrators/<key>.json`,
one per (competition, calibrator_class) pair. We keep the
calibrator class name in the JSON so the loader can re-hydrate it.

The default location is `<project_root>/data/calibrators/`. The
harness / API pass a custom `root` so tests can use a tmp dir.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .calibration import (
    BinningCalibrator,
    IdentityCalibrator,
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureCalibrator,
)


# Registry: name -> class. Used to re-hydrate a saved calibrator.
# Order is arbitrary; just make sure every calibrator class that
# `eval/calibration.py` knows about is in here.
_CALIBRATORS: dict[str, type] = {
    "identity": IdentityCalibrator,
    "platt": PlattCalibrator,
    "temperature": TemperatureCalibrator,
    "isotonic": IsotonicCalibrator,
    "binning": BinningCalibrator,
}


def _file_for(root: Path, key: str) -> Path:
    """Path to a calibrator file. Key is the basename; we add .json."""
    # Reject path traversal in the key: no slashes, no .., no leading dots.
    if "/" in key or "\\" in key or ".." in key or key.startswith("."):
        raise ValueError(
            f"invalid calibrator key: {key!r} "
            "(must be a single filename component, no slashes or '..')"
        )
    return root / f"{key}.json"


def save_calibrator(
    calibrator: Any,
    *,
    key: str,
    root: Path,
    competition: str,
    n_samples: int,
    ece: float,
    brier: float,
) -> Path:
    """Persist a fitted calibrator to `root/<key>.json`.

    The calibrator must have a `calibrate(probs)` method (anything
    satisfying the `Calibrator` protocol works). We also stash
    metadata: competition, sample count, achieved ECE / Brier.
    The metadata is for the dashboard and the CLI to display —
    `predict()` doesn't read it.

    Returns the path written.
    """
    cls_name = type(calibrator).__name__
    if not any(cls_name == cls.__name__ for cls in _CALIBRATORS.values()):
        raise ValueError(
            f"unknown calibrator class: {cls_name!r}. "
            f"Known: {sorted(c.__name__ for c in _CALIBRATORS.values())}"
        )
    root.mkdir(parents=True, exist_ok=True)
    out_path = _file_for(root, key)
    # Each calibrator class knows how to dump itself. The simplest
    # protocol is `to_dict()` if it exists; otherwise we fall back
    # to a name+stubs payload that round-trips through Identity.
    payload: dict[str, Any] = {
        "class": cls_name,
        "competition": competition,
        "n_samples": n_samples,
        "ece": ece,
        "brier": brier,
        "state": _dump_state(calibrator),
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return out_path


def _dump_state(calibrator: Any) -> dict[str, Any]:
    """Extract the calibrator's internal state into JSON-serializable form.

    Each calibrator has a different internal layout, so we handle
    them by class name. Adding a new calibrator means adding a
    branch here AND a rehydration branch below.
    """
    name = type(calibrator).__name__
    if name == "IdentityCalibrator":
        return {}
    if name == "PlattCalibrator":
        return {
            "a": float(calibrator.a),  # type: ignore[attr-defined]
            "b": float(calibrator.b),  # type: ignore[attr-defined]
        }
    if name == "TemperatureCalibrator":
        return {"temperature": float(calibrator.temperature)}  # type: ignore[attr-defined]
    if name == "IsotonicCalibrator":
        # IsotonicCalibrator stores its step function as public
        # attributes xs[] / ys[] (see calibration.py). We serialize
        # both arrays — at query time the calibrator binary-searches
        # xs[] and interpolates in ys[].
        return {
            "xs": [float(x) for x in calibrator.xs],
            "ys": [float(y) for y in calibrator.ys],
        }
    if name == "BinningCalibrator":
        # BinningCalibrator stores its edges and per-bucket values as
        # private fields `_edges` / `_values` (see calibration.py).
        # We serialize them by name even though the underscore makes
        # it look private — that's the calibrator's contract.
        return {
            "edges": list(calibrator._edges),  # type: ignore[attr-defined]
            "values": list(calibrator._values),  # type: ignore[attr-defined]
        }
    raise ValueError(f"don't know how to dump calibrator: {name}")


def load_calibrator(*, key: str, root: Path) -> Any | None:
    """Load a fitted calibrator from `root/<key>.json`, or None.

    Returns None (not raises) if the file is missing — that's the
    common case for fresh installs and is the same signal as "no
    calibrator has been fitted yet for this key".
    """
    path = _file_for(root, key)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    cls_name = payload["class"]  # e.g. "IsotonicCalibrator"
    # Registry is keyed by lowercase basename ("isotonic" -> IsotonicCalibrator).
    # Match by suffix so we don't require the save side to know the
    # internal registry key format.
    cls = None
    for c in _CALIBRATORS.values():
        if c.__name__ == cls_name:
            cls = c
            break
    if cls is None:
        raise ValueError(f"unknown calibrator class in {path}: {cls_name!r}")
    state = payload.get("state", {})
    cal = cls()
    _hydrate_state(cal, state, cls.__name__)
    return cal


def _hydrate_state(cal: Any, state: dict[str, Any], cls_name: str) -> None:
    """Reverse of `_dump_state`. Set internal attributes so `calibrate()`
    returns the right values."""
    if cls_name == "IdentityCalibrator":
        return
    if cls_name == "PlattCalibrator":
        cal.a = float(state["a"])  # type: ignore[attr-defined]
        cal.b = float(state["b"])  # type: ignore[attr-defined]
        return
    if cls_name == "TemperatureCalibrator":
        cal.temperature = float(state["temperature"])  # type: ignore[attr-defined]
        return
    if cls_name == "IsotonicCalibrator":
        # Mirrors the dump side. The calibrator's calibrate() reads
        # xs[]/ys[]; we just need to put the saved arrays back in.
        cal.xs = list(state["xs"])  # type: ignore[attr-defined]
        cal.ys = list(state["ys"])  # type: ignore[attr-defined]
        cal._fitted = True  # type: ignore[attr-defined]
        return
    if cls_name == "BinningCalibrator":
        # BinningCalibrator stores its data in private fields
        # (leading underscore). They're public-by-contract; see the
        # dump side for the same rationale.
        cal._edges = list(state["edges"])  # type: ignore[attr-defined]
        cal._values = list(state["values"])  # type: ignore[attr-defined]
        cal._global_mean = float(  # type: ignore[attr-defined]
            sum(state["values"]) / max(len(state["values"]), 1)
        )
        cal._fitted = True  # type: ignore[attr-defined]
        return
    raise ValueError(f"don't know how to hydrate: {cls_name}")


def fit_isotonic(
    samples: list[tuple[float, int]],
    *,
    key: str,
    root: Path,
    competition: str,
) -> tuple[Any, dict[str, Any]]:
    """Fit an IsotonicCalibrator from (confidence, outcome) pairs and save.

    Returns (calibrator, metadata). Metadata is a dict you can
    pass to the dashboard.
    """
    from .calibration import IsotonicCalibrator, brier, ece
    probs = [c for c, _ in samples]
    outs = [o for _, o in samples]
    cal = IsotonicCalibrator().fit(probs, outs)
    cal_probs = cal.calibrate(probs)
    meta = {
        "n_samples": len(samples),
        "ece": ece(cal_probs, outs),
        "brier": brier(cal_probs, outs),
        "competition": competition,
    }
    save_calibrator(
        cal, key=key, root=root,
        competition=competition,
        n_samples=meta["n_samples"],
        ece=meta["ece"], brier=meta["brier"],
    )
    return cal, meta
