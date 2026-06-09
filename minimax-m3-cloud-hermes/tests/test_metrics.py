"""Tests for Task 17: eval metrics.

These metrics feed the harness (Task 18) and the eventual dashboard.
Inputs are lists of `Prediction` rows joined with their `Result` rows.
We test against a tiny hand-built dataset so the math is checkable by
hand.

Convention: Brier is `sum((p - y)^2) / 2` for 3 classes, so 0.0 is
perfect, 1.0 is worst (e.g. 100% on wrong class).

Confidence: 0..1 unit interval, where 1.0 means "fully confident".
Accuracy is the simple `pick == actual_winner` rate.
"""
from __future__ import annotations

import math

import pytest

from soccer_agent.eval.metrics import (
    accuracy,
    brier_mean,
    calibration_bins,
    log_loss,
    metric_summary,
    per_class_metrics,
    row_from_db,
    rows_from_db,
    top_factor_hit_rate,
)
from soccer_agent.eval.scoring import brier as brier_one


def _row(match_id, pick, actual, conf, probs, top_factor_hit=1):
    """Build a minimal harness row (dict, Pydantic-agnostic)."""
    return {
        "match_id": match_id,
        "pick": pick,
        "actual": actual,
        "confidence": conf,
        "probs": probs,
        "top_factor_hit": top_factor_hit,
    }


# -- accuracy -----------------------------------------------------------------

def test_accuracy_all_correct():
    rows = [
        _row("m1", "home", "home", 0.7, {"home": 0.7, "draw": 0.2, "away": 0.1}),
        _row("m2", "draw", "draw", 0.5, {"home": 0.3, "draw": 0.5, "away": 0.2}),
    ]
    assert accuracy(rows) == 1.0


def test_accuracy_half_correct():
    rows = [
        _row("m1", "home", "home", 0.7, {"home": 0.7, "draw": 0.2, "away": 0.1}),
        _row("m2", "home", "away", 0.6, {"home": 0.6, "draw": 0.2, "away": 0.2}),
    ]
    assert accuracy(rows) == 0.5


def test_accuracy_empty_returns_nan():
    import math as _m
    assert _m.isnan(accuracy([]))


def test_accuracy_only_counts_resolved_rows():
    """A prediction without an `actual` field can't be graded — skip it.
    If *all* rows are unresolved, the metric is undefined (NaN)."""
    rows = [
        _row("m1", "home", None, 0.7, {"home": 0.7, "draw": 0.2, "away": 0.1}),
        _row("m2", "home", "home", 0.7, {"home": 0.7, "draw": 0.2, "away": 0.1}),
    ]
    assert accuracy(rows) == 1.0  # only the resolved row counts
    assert math.isnan(accuracy([_row("m1", "home", None, 0.7, {})]))


# -- brier mean ----------------------------------------------------------------

def test_brier_mean_perfect_predictions():
    """Brier=0.0 for every row when predicted probs are one-hot on the
    correct class. brier_mean should be 0.0."""
    rows = [
        _row("m1", "home", "home", 1.0, {"home": 1.0, "draw": 0.0, "away": 0.0}),
        _row("m2", "away", "away", 1.0, {"home": 0.0, "draw": 0.0, "away": 1.0}),
    ]
    assert brier_mean(rows) == 0.0


def test_brier_mean_hand_computed():
    """Hand-compute against two rows and the brier() helper.

    Row 1: p=(home:0.7, draw:0.2, away:0.1), actual=home
        one-hot y = (1, 0, 0)
        brier = ((0.7-1)^2 + (0.2-0)^2 + (0.1-0)^2) / 2
              = (0.09 + 0.04 + 0.01) / 2 = 0.14 / 2 = 0.07

    Row 2: p=(home:0.6, draw:0.2, away:0.2), actual=away
        one-hot y = (0, 0, 1)
        brier = ((0.6-0)^2 + (0.2-0)^2 + (0.2-1)^2) / 2
              = (0.36 + 0.04 + 0.64) / 2 = 1.04 / 2 = 0.52

    Mean = (0.07 + 0.52) / 2 = 0.295
    """
    rows = [
        _row("m1", "home", "home", 0.7, {"home": 0.7, "draw": 0.2, "away": 0.1}),
        _row("m2", "home", "away", 0.6, {"home": 0.6, "draw": 0.2, "away": 0.2}),
    ]
    expected = (brier_one(rows[0]["probs"], "home") + brier_one(rows[1]["probs"], "away")) / 2
    assert math.isclose(brier_mean(rows), expected, rel_tol=1e-9)
    assert math.isclose(brier_mean(rows), 0.295, rel_tol=1e-9)


def test_brier_mean_skips_unresolved():
    rows = [
        _row("m1", "home", None, 0.7, {"home": 0.7, "draw": 0.2, "away": 0.1}),
        _row("m2", "home", "home", 1.0, {"home": 1.0, "draw": 0.0, "away": 0.0}),
    ]
    assert brier_mean(rows) == 0.0


# -- log loss ------------------------------------------------------------------

def test_log_loss_perfect():
    """Log loss -> 0 when predicted probs are one-hot on the correct class.
    Clipped to a small epsilon to avoid log(0) = -inf."""
    rows = [
        _row("m1", "home", "home", 1.0, {"home": 0.9999, "draw": 0.0001, "away": 0.0}),
    ]
    assert log_loss(rows) == pytest.approx(0.0, abs=1e-3)


def test_log_loss_known_value():
    """For p=0.5, log loss = -ln(0.5) = ln(2) ≈ 0.6931."""
    rows = [
        _row("m1", "home", "home", 0.5, {"home": 0.5, "draw": 0.3, "away": 0.2}),
    ]
    assert log_loss(rows) == pytest.approx(math.log(2), rel=1e-3)


# -- top factor hit rate -------------------------------------------------------

def test_top_factor_hit_rate_counts_only_known_hits():
    """top_factor_hit: 1 = hit, 0 = miss, None = unknown (skip)."""
    rows = [
        _row("m1", "home", "home", 0.5, {}, top_factor_hit=1),
        _row("m2", "home", "away", 0.5, {}, top_factor_hit=0),
        _row("m3", "home", "home", 0.5, {}, top_factor_hit=None),  # skip
    ]
    assert top_factor_hit_rate(rows) == 0.5  # 1 hit of 2 known


def test_top_factor_hit_rate_all_unknown_is_nan():
    rows = [_row("m1", "home", "home", 0.5, {}, top_factor_hit=None)]
    assert math.isnan(top_factor_hit_rate(rows))


# -- per-class metrics ---------------------------------------------------------

def test_per_class_metrics_hand_computed():
    """Confusion matrix hand-built for 3 classes (home, draw, away):
        actual:  home, away, draw, away, away
        pick:    home, home, home, away, away
        -> TP_home=1, FP_home=2, FN_home=1  -> precision=1/3, recall=1/2
        -> TP_draw=0, FP_draw=0, FN_draw=1  -> precision=NaN, recall=0
        -> TP_away=2, FP_away=0, FN_away=1  -> precision=1.0, recall=2/3
    """
    rows = [
        _row("m1", "home", "home", 0.5, {}),  # TP_home
        _row("m2", "home", "away", 0.5, {}),  # FN_away, FP_home
        _row("m3", "home", "draw", 0.5, {}),  # FN_draw, FP_home
        _row("m4", "away", "away", 0.5, {}),  # TP_away
        _row("m5", "away", "away", 0.5, {}),  # TP_away
    ]
    p = per_class_metrics(rows)
    # home: tp=1, fp=2, fn=0 -> precision=1/3, recall=1.0
    assert math.isclose(p["home"]["precision"], 1/3)
    assert p["home"]["recall"] == 1.0
    # draw: never picked but 1 actual -> recall=0, precision=NaN
    assert math.isnan(p["draw"]["precision"])
    assert p["draw"]["recall"] == 0.0
    # away: tp=2, fp=0, fn=1 -> precision=1.0, recall=2/3
    assert p["away"]["precision"] == 1.0
    assert math.isclose(p["away"]["recall"], 2/3)


# -- calibration ---------------------------------------------------------------

def test_calibration_bins_basic():
    """Two bins, [0.0, 0.5) and [0.5, 1.0], each with 2 predictions.
    In bin 1: confidence avg=0.4, actual win rate=0.5  -> off by 0.1
    In bin 2: confidence avg=0.7, actual win rate=0.5  -> off by 0.2
    ECE = mean(|conf - actual|) weighted by bin size
        = (|0.4-0.5|*2 + |0.7-0.5|*2) / 4 = (0.2 + 0.4) / 4 = 0.15
    """
    rows = [
        _row("m1", "home", "home", 0.4, {}),  # 0.4 bin, correct
        _row("m2", "home", "away", 0.4, {}),  # 0.4 bin, wrong
        _row("m3", "home", "home", 0.7, {}),  # 0.7 bin, correct
        _row("m4", "home", "away", 0.7, {}),  # 0.7 bin, wrong
    ]
    bins = calibration_bins(rows, n_bins=2)
    assert len(bins) == 2
    assert math.isclose(bins[0]["count"], 2)
    assert math.isclose(bins[1]["count"], 2)
    ece = sum(abs(b["mean_confidence"] - b["mean_actual"]) * b["count"] for b in bins) / 4
    assert math.isclose(ece, 0.15, rel_tol=1e-9)


# -- summary -------------------------------------------------------------------

def test_metric_summary_keys_present():
    rows = [
        _row("m1", "home", "home", 0.7, {"home": 0.7, "draw": 0.2, "away": 0.1}),
        _row("m2", "home", "away", 0.6, {"home": 0.6, "draw": 0.2, "away": 0.2}),
    ]
    s = metric_summary(rows)
    for key in ("n_resolved", "n_total", "accuracy", "brier_mean", "log_loss",
                "top_factor_hit_rate", "per_class", "calibration_ece"):
        assert key in s
    assert s["n_resolved"] == 2
    assert s["n_total"] == 2
    assert s["accuracy"] == 0.5


# -- row_from_db adapter -------------------------------------------------------

def _dbrow(**kw):
    """Build a sqlite3.Row-like dict with the columns list_predictions joins."""
    import json
    base = {
        "prediction_id": "p-1",
        "match_id": "m1",
        "final_pick": "home",
        "final_probs": json.dumps({"home": 0.7, "draw": 0.2, "away": 0.1}),
        "final_confidence": 0.7,
        "top_factor_hit": 1,
        "home_goals": None,
        "away_goals": None,
    }
    base.update(kw)
    return base


def test_row_from_db_resolves_actual_from_goals():
    """`actual` should be 'home'/'draw'/'away' when both goals are set."""
    r = row_from_db(_dbrow(home_goals=2, away_goals=1))
    assert r["pick"] == "home"
    assert r["actual"] == "home"
    assert r["confidence"] == 0.7
    assert r["probs"]["home"] == 0.7
    assert r["top_factor_hit"] == 1

    r2 = row_from_db(_dbrow(home_goals=1, away_goals=1))
    assert r2["actual"] == "draw"

    r3 = row_from_db(_dbrow(home_goals=0, away_goals=2))
    assert r3["actual"] == "away"


def test_row_from_db_unresolved_has_actual_none():
    """When goals are null, the row is unresolved (actual=None)."""
    r = row_from_db(_dbrow())
    assert r["actual"] is None


def test_rows_from_db_pipes_through_metric_summary():
    """End-to-end: db-shaped rows -> summary dict."""
    rows = rows_from_db([
        # perfect prediction: one-hot on home, actual=home
        _dbrow(match_id="m1", final_pick="home",
               final_probs='{"home":1.0,"draw":0.0,"away":0.0}',
               final_confidence=1.0, home_goals=2, away_goals=1),
        # worst-case home: predicted home, actual away
        _dbrow(match_id="m2", final_pick="home",
               final_probs='{"home":0.6,"draw":0.2,"away":0.2}',
               final_confidence=0.6, home_goals=0, away_goals=2),
    ])
    s = metric_summary(rows)
    assert s["n_resolved"] == 2
    assert s["accuracy"] == 0.5
    # m1 (one-hot on actual) = 0
    # m2 (worst case) = (0.36+0.04+0.64)/2 = 0.52
    assert math.isclose(s["brier_mean"], 0.52 / 2, rel_tol=1e-9)
