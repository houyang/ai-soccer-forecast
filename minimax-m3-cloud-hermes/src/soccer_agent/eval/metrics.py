"""Eval metrics for the harness (Task 18) and the dashboard.

Inputs: a list of `row` dicts, one per prediction, each containing:
  - `match_id`   (str)
  - `pick`       (Literal['home','draw','away'])
  - `actual`     (Literal['home','draw','away'] | None)  -- None if ungraded
  - `confidence` (float in [0, 1])
  - `probs`      (dict[Literal['home','draw','away'] -> float])
  - `top_factor_hit` (0 | 1 | None)

The functions are dict-in / value-out so they don't depend on the
Pydantic models — easier to test, easier to call from the harness
which builds rows from sqlite3.Row objects.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from .scoring import brier as _brier_one

CLASSES = ("home", "draw", "away")
_LOG_EPS = 1e-15  # avoid log(0)


def row_from_db(row: Mapping[str, Any]) -> dict[str, Any]:
    """Adapter: turn a `sqlite3.Row` (or any dict-like) joined row
    into the dict shape the metric functions expect.

    Expected columns (subset of what `list_predictions` joins):
        prediction_id, match_id, final_pick, final_probs (JSON str),
        final_confidence, top_factor_hit, home_goals, away_goals.

    `actual` is derived from `home_goals`/`away_goals` (both
    present and non-null = resolved). If one is null, `actual` is
    None and the row is skipped by every rate metric.
    """
    import json
    probs = row["final_probs"]
    if isinstance(probs, str):
        probs = json.loads(probs)
    hg = row["home_goals"] if "home_goals" in row.keys() else None
    ag = row["away_goals"] if "away_goals" in row.keys() else None
    if hg is not None and ag is not None:
        if hg > ag:
            actual = "home"
        elif hg < ag:
            actual = "away"
        else:
            actual = "draw"
    else:
        actual = None
    return {
        "match_id": row["match_id"],
        "pick": row["final_pick"],
        "actual": actual,
        "confidence": float(row["final_confidence"]),
        "probs": probs or {},
        "top_factor_hit": row["top_factor_hit"] if "top_factor_hit" in row.keys() else None,
    }


def rows_from_db(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [row_from_db(r) for r in rows]


def _resolved(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in rows if r.get("actual") in CLASSES]


# -- simple rate-style metrics -------------------------------------------------

def accuracy(rows: list[dict[str, Any]]) -> float:
    """Fraction of resolved predictions where pick == actual.
    NaN if no rows are resolved (caller decides how to render)."""
    resolved = _resolved(rows)
    if not resolved:
        return float("nan")
    hits = sum(1 for r in resolved if r["pick"] == r["actual"])
    return hits / len(resolved)


def top_factor_hit_rate(rows: list[dict[str, Any]]) -> float:
    """Mean of top_factor_hit across rows where it's known (0 or 1).
    Skips rows with top_factor_hit=None (not scored).
    NaN if no rows have a known hit."""
    known = [r for r in rows if r.get("top_factor_hit") in (0, 1)]
    if not known:
        return float("nan")
    return sum(int(r["top_factor_hit"]) for r in known) / len(known)


# -- probability-based metrics ------------------------------------------------

def brier_mean(rows: list[dict[str, Any]]) -> float:
    """Mean 3-class Brier score across resolved rows.
    Brier is `sum((p - y)^2) / 2` — 0.0 perfect, 1.0 worst."""
    resolved = _resolved(rows)
    if not resolved:
        return float("nan")
    total = 0.0
    for r in resolved:
        total += _brier_one(r["probs"], r["actual"])
    return total / len(resolved)


def log_loss(rows: list[dict[str, Any]]) -> float:
    """Multinomial log loss: -mean(log p_actual).

    p_actual is clipped to [_LOG_EPS, 1] to avoid log(0) = -inf on
    perfect-but-slightly-off probs. This is the same convention
    scikit-learn uses."""
    resolved = _resolved(rows)
    if not resolved:
        return float("nan")
    total = 0.0
    for r in resolved:
        p = r["probs"].get(r["actual"], 0.0)
        p = min(max(p, _LOG_EPS), 1.0)
        total += -math.log(p)
    return total / len(resolved)


# -- per-class metrics --------------------------------------------------------

def per_class_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Precision + recall + F1 for each of the 3 classes.
    A class with 0 predictions gets NaN for both metrics (not 0,
    which would imply a perfect-but-untested class)."""
    resolved = _resolved(rows)
    out: dict[str, dict[str, float]] = {}
    for c in CLASSES:
        tp = fp = fn = 0
        for r in resolved:
            if r["pick"] == c and r["actual"] == c:
                tp += 1
            elif r["pick"] == c:
                fp += 1
            elif r["actual"] == c:
                fn += 1
        # precision is undefined when the class was never picked (0/0);
        # recall is undefined when the class never actually occurred
        # (0/0). We render those as NaN. The mixed case (tp=0, fp>0)
        # gives precision=0, which is meaningful.
        if tp + fp == 0:
            prec = float("nan")
        else:
            prec = tp / (tp + fp)
        if tp + fn == 0:
            rec = float("nan")
        else:
            rec = tp / (tp + fn)
        f1 = (
            float("nan")
            if (math.isnan(prec) or math.isnan(rec) or (prec + rec) == 0)
            else 2 * prec * rec / (prec + rec)
        )
        out[c] = {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "support": tp + fn,
            "predicted_count": tp + fp,
        }
    return out


# -- calibration ---------------------------------------------------------------

def calibration_bins(
    rows: list[dict[str, Any]], n_bins: int = 10
) -> list[dict[str, float]]:
    """Bin rows by confidence, return per-bin stats.

    Each row contributes `1.0` to `mean_actual` if pick == actual
    (i.e. the agent got it right), else 0.0. This is a binary
    calibration: how well does confidence in the *top pick* track
    actual win rate? For 3-class calibration (the richer thing),
    see the Brier score above.

    Bins are `[0, 1/n), [1/n, 2/n), ..., [(n-1)/n, 1]`. Rows with
    confidence == 1.0 fall in the last bin (inclusive upper bound).
    """
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    out: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [
            r
            for r in rows
            if lo <= float(r.get("confidence", 0.0)) <= hi
        ]
        if not in_bin:
            out.append({
                "lo": lo, "hi": hi,
                "count": 0, "mean_confidence": float("nan"),
                "mean_actual": float("nan"),
            })
            continue
        mean_conf = sum(float(r["confidence"]) for r in in_bin) / len(in_bin)
        mean_actual = sum(
            1.0 if r.get("actual") in CLASSES and r["pick"] == r["actual"] else 0.0
            for r in in_bin
        ) / len(in_bin)
        out.append({
            "lo": lo, "hi": hi,
            "count": len(in_bin),
            "mean_confidence": mean_conf,
            "mean_actual": mean_actual,
        })
    return out


def expected_calibration_error(
    rows: list[dict[str, Any]], n_bins: int = 10
) -> float:
    """ECE = weighted mean |confidence - actual| across bins.
    Uses `calibration_bins` internally. NaN if there are no rows."""
    if not rows:
        return float("nan")
    bins = calibration_bins(rows, n_bins=n_bins)
    total = sum(b["count"] for b in bins)
    if total == 0:
        return float("nan")
    ece = 0.0
    for b in bins:
        if b["count"] == 0:
            continue
        ece += abs(b["mean_confidence"] - b["mean_actual"]) * b["count"]
    return ece / total


# -- summary -------------------------------------------------------------------

def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """All metrics in one dict — what the harness logs / what the
    dashboard displays. Keys are stable so the dashboard can rely on them."""
    resolved = _resolved(rows)
    return {
        "n_total": len(rows),
        "n_resolved": len(resolved),
        "accuracy": accuracy(rows),
        "brier_mean": brier_mean(rows),
        "log_loss": log_loss(rows),
        "top_factor_hit_rate": top_factor_hit_rate(rows),
        "per_class": per_class_metrics(rows),
        "calibration_ece": expected_calibration_error(resolved),
        "calibration_bins": calibration_bins(resolved),
    }
