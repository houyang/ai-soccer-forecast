"""Tests for src/soccer_agent/eval/calibration.py (Task 28)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from soccer_agent.db import Database, init_db
from soccer_agent.eval.calibration import (
    CalibSample,
    CalibrationReport,
    collect_samples,
    loo_eval,
    run_calibration_report,
)
from soccer_agent.eval.dataset import EVAL_CASES


def _seed_predictions(db_path: Path, picks_and_confs: list[tuple[str, str, float]]) -> None:
    """Insert predictions for the eval cases.

    `picks_and_confs` is a list of (match_id, pick, confidence) — the
    test picks a subset of EVAL_CASES.
    """
    init_db(db_path)
    import uuid
    with Database(db_path) as db:
        for mid, pick, conf in picks_and_confs:
            db.insert_prediction({
                "prediction_id": str(uuid.uuid4()),
                "match_id": mid,
                "created_at": "2024-12-01T00:00:00Z",
                "signals": {},
                "reasoner_outputs": [],
                "final_pick": pick,
                "final_probs": {pick: 1.0},
                "final_confidence": float(conf),
                "final_rationale": "test fixture",
                "warnings": [],
                "model_versions": {"test": "1"},
            })


def test_collect_samples_empty_db(tmp_path):
    db = tmp_path / "calib.db"
    init_db(db)
    samples = collect_samples(db)
    assert samples == []


def test_collect_samples_filters_unknown_matches(tmp_path):
    """Predictions for match_ids that aren't in EVAL_CASES are dropped."""
    db = tmp_path / "calib.db"
    _seed_predictions(db, [
        ("ucl_gs_2024_bayern_barca", "home", 0.7),
        ("some_random_match_id", "home", 0.7),
    ])
    samples = collect_samples(db)
    assert len(samples) == 1
    assert samples[0].match_id == "ucl_gs_2024_bayern_barca"


def test_collect_samples_reduces_to_p_right(tmp_path):
    """When the pick matches the actual winner, p_right = confidence.
    When it doesn't, p_right = 1 - confidence."""
    case = next(c for c in EVAL_CASES if c.match_id == "ucl_gs_2024_bayern_barca")
    # bayern beat barca 4-2, so actual_winner is "home" (= bayern).
    db = tmp_path / "calib.db"
    _seed_predictions(db, [(case.match_id, "home", 0.7)])
    samples = collect_samples(db)
    assert len(samples) == 1
    s = samples[0]
    assert s.pick == "home"
    assert s.actual == "home"
    assert s.outcome == 1
    assert s.p_right == pytest.approx(0.7)


def test_collect_samples_wrong_pick_flips_confidence(tmp_path):
    """Wrong pick → outcome=0 → p_right = 1 - confidence."""
    case = next(c for c in EVAL_CASES if c.match_id == "ucl_gs_2024_bayern_barca")
    # predict "away" with confidence 0.6 (wrong) → p_right = 0.4
    db = tmp_path / "calib.db"
    _seed_predictions(db, [(case.match_id, "away", 0.6)])
    samples = collect_samples(db)
    s = samples[0]
    assert s.outcome == 0
    assert s.p_right == pytest.approx(0.4)


def test_run_calibration_report_shape(tmp_path):
    """Seeded with 5 matches; the report has n=5, raw ECE, and a
    per-method LOO block."""
    db = tmp_path / "calib.db"
    cases = EVAL_CASES[:5]
    # All correct picks for the first 5 to give a non-trivial raw.
    seeds = []
    for c in cases:
        seeds.append((c.match_id, c.actual_winner, 0.6))
    _seed_predictions(db, seeds)
    report = run_calibration_report(db)
    assert report.n_samples == 5
    assert 0.0 <= report.raw_ece <= 1.0
    assert 0.0 <= report.raw_brier <= 1.0
    assert "identity" in report.loo
    assert "platt" in report.loo
    assert "temperature" in report.loo
    assert "isotonic" in report.loo
    assert "binning" in report.loo
    for name, v in report.loo.items():
        assert 0.0 <= v["ece"] <= 1.0
        assert 0.0 <= v["brier"] <= 1.0


def test_run_calibration_report_writes_json(tmp_path):
    db = tmp_path / "calib.db"
    cases = EVAL_CASES[:3]
    _seed_predictions(db, [(c.match_id, c.actual_winner, 0.5) for c in cases])
    out = tmp_path / "report.json"
    report = run_calibration_report(db)
    out.write_text(json.dumps(report.to_dict(), indent=2))
    data = json.loads(out.read_text())
    assert data["n_samples"] == 3
    assert "raw" in data
    assert "loo" in data
    assert len(data["samples"]) == 3


def test_loo_eval_returns_all_methods(tmp_path):
    """loo_eval returns identity + 4 calibrators, each with ece/brier."""
    db = tmp_path / "calib.db"
    cases = EVAL_CASES[:5]
    _seed_predictions(db, [(c.match_id, c.actual_winner, 0.6) for c in cases])
    samples = collect_samples(db)
    loo = loo_eval(samples)
    assert set(loo.keys()) == {
        "identity", "platt", "temperature", "isotonic", "binning",
    }
    for v in loo.values():
        assert "ece" in v and "brier" in v and "calibrated" in v


def test_run_calibration_report_summary_includes_raw_and_loo(tmp_path):
    """The summary text mentions ECE, Brier, and all 5 methods."""
    db = tmp_path / "calib.db"
    cases = EVAL_CASES[:5]
    _seed_predictions(db, [(c.match_id, c.actual_winner, 0.6) for c in cases])
    report = run_calibration_report(db)
    s = report.summary()
    assert "ECE" in s and "Brier" in s
    for m in ["identity", "platt", "temperature", "isotonic", "binning"]:
        assert m in s
