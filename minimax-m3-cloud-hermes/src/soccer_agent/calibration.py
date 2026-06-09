"""Calibration module (Task 28).

Converts a stated probability into a well-calibrated one. Provides:

- `ece()` and `brier()` — quality metrics.
- `reliability_table()` — per-bucket stats for the reliability diagram.
- `IdentityCalibrator` — no-op baseline.
- `PlattCalibrator` — 1D logistic regression (a, b) on logit(p).
- `IsotonicCalibrator` — isotonic regression, non-parametric.
- `TemperatureCalibrator` — single scalar T, applied to logits.
- `BinningCalibrator` — per-bin empirical win rate, with width
  shrinkage to avoid sparse-bin issues.
- `Calibrator.fit_isotonic`, `Calibrator.fit_platt` — fit a calibrator
  on a (probs, outcomes) sample and apply it to new probs.

All calibrators share the same `calibrate(probs) -> probs` interface
so they're swappable in the eval harness.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


# --- metrics ---------------------------------------------------------------


def ece(
    probs: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Splits the [0, 1] probability range into `n_bins` equal-width
    buckets. For each bucket, computes `|avg_pred - avg_outcome|`
    weighted by bucket size. Returns a value in [0, 1]; lower is
    better.

    `probs` are the predicted probabilities of the *positive* class.
    `outcomes` are 0/1 actuals.
    """
    if len(probs) != len(outcomes):
        raise ValueError(
            f"length mismatch: {len(probs)} probs vs {len(outcomes)} outcomes"
        )
    n = len(probs)
    if n == 0:
        return 0.0
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, outcomes):
        # Edge case: p == 1.0 should land in the last bucket.
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    ece_val = 0.0
    for b in bins:
        if not b:
            continue
        avg_p = sum(p for p, _ in b) / len(b)
        avg_y = sum(y for _, y in b) / len(b)
        ece_val += (len(b) / n) * abs(avg_p - avg_y)
    return ece_val


def brier(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Brier score = mean squared error of (p - y)^2.

    Lower is better; a perfectly calibrated + sharp model on a
    50/50 base rate gets ~0.25.
    """
    if len(probs) != len(outcomes):
        raise ValueError(
            f"length mismatch: {len(probs)} probs vs {len(outcomes)} outcomes"
        )
    n = len(probs)
    if n == 0:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / n


def reliability_table(
    probs: Sequence[float],
    outcomes: Sequence[int],
    n_bins: int = 10,
) -> list[dict]:
    """Per-bucket stats: count, avg predicted, avg outcome, gap.

    Useful for plotting a reliability diagram.
    """
    if len(probs) != len(outcomes):
        raise ValueError(
            f"length mismatch: {len(probs)} probs vs {len(outcomes)} outcomes"
        )
    bins: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(probs, outcomes):
        idx = min(int(p * n_bins), n_bins - 1)
        bins[idx].append((p, y))
    table = []
    for lo in range(n_bins):
        hi = (lo + 1) / n_bins
        b = bins[lo]
        if not b:
            table.append({
                "lo": lo / n_bins, "hi": hi, "n": 0,
                "avg_p": None, "avg_y": None, "gap": None,
            })
            continue
        avg_p = sum(p for p, _ in b) / len(b)
        avg_y = sum(y for _, y in b) / len(b)
        table.append({
            "lo": lo / n_bins, "hi": hi, "n": len(b),
            "avg_p": avg_p, "avg_y": avg_y,
            "gap": avg_p - avg_y,
        })
    return table


# --- calibrator protocol ---------------------------------------------------


class Calibrator(Protocol):
    def calibrate(self, probs: Sequence[float]) -> list[float]: ...


# --- identity --------------------------------------------------------------


@dataclass
class IdentityCalibrator:
    """No-op. p -> p. Useful as a baseline."""

    def fit(self, probs: Sequence[float], outcomes: Sequence[int]) -> "IdentityCalibrator":
        # Identity has no parameters. `fit` is here only so the
        # calibrators can be used interchangeably.
        return self

    def calibrate(self, probs: Sequence[float]) -> list[float]:
        return [float(p) for p in probs]


# --- platt (1D logistic) ---------------------------------------------------


@dataclass
class PlattCalibrator:
    """1D Platt scaling: fit p' = sigmoid(a * logit(p) + b).

    This is a 2-parameter logistic regression on the logit of the
    predicted probability. It assumes the original probabilities are
    already shaped by a model and only the *scale* and *bias* of the
    logit need to be corrected. When the input is a hand-weighted
    blend (no real logits), this is a sensible default.
    """

    a: float = 1.0
    b: float = 0.0
    n_iter: int = 200
    lr: float = 0.05
    l2: float = 1e-4
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, probs: Sequence[float], outcomes: Sequence[int]) -> "PlattCalibrator":
        """Fit a, b by gradient descent on the Bernoulli NLL.

        Initial a=1, b=0 is the identity. We add L2 regularization
        toward that point so we don't overfit on tiny calibration
        sets.
        """
        if len(probs) != len(outcomes):
            raise ValueError("length mismatch")
        if not probs:
            self._fitted = True
            return self
        a, b = float(self.a), float(self.b)
        for _ in range(self.n_iter):
            grad_a = 0.0
            grad_b = 0.0
            for p, y in zip(probs, outcomes):
                p = min(max(p, 1e-6), 1.0 - 1e-6)
                z = a * math.log(p / (1.0 - p)) + b
                pred = 1.0 / (1.0 + math.exp(-z))
                # dNLL/da = (pred - y) * z' / 1, where z' = logit(p)
                grad_a += (pred - y) * math.log(p / (1.0 - p))
                grad_b += (pred - y)
            grad_a /= len(probs)
            grad_b /= len(probs)
            # L2 toward (1, 0)
            grad_a += self.l2 * (a - 1.0)
            grad_b += self.l2 * b
            a -= self.lr * grad_a
            b -= self.lr * grad_b
        self.a = a
        self.b = b
        self._fitted = True
        return self

    def calibrate(self, probs: Sequence[float]) -> list[float]:
        out = []
        for p in probs:
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            z = self.a * math.log(p / (1.0 - p)) + self.b
            out.append(1.0 / (1.0 + math.exp(-z)))
        return out


# --- temperature -----------------------------------------------------------


@dataclass
class TemperatureCalibrator:
    """Single-scalar temperature on logits: p' = softmax(logit(p) / T).

    T > 1 softens (less confident), T < 1 sharpens (more confident).
    T = 1 is identity. When the upstream model already produces good
    probabilities and just needs a global temperature fix, this is
    the right tool.
    """

    temperature: float = 1.0
    n_iter: int = 200
    lr: float = 0.05
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, probs: Sequence[float], outcomes: Sequence[int]) -> "TemperatureCalibrator":
        if len(probs) != len(outcomes):
            raise ValueError("length mismatch")
        if not probs:
            self._fitted = True
            return self
        # 1D line search on T to minimize NLL.
        t = float(self.temperature)
        for _ in range(self.n_iter):
            grad = 0.0
            for p, y in zip(probs, outcomes):
                p = min(max(p, 1e-6), 1.0 - 1e-6)
                z = math.log(p / (1.0 - p)) / t
                pred = 1.0 / (1.0 + math.exp(-z))
                # dNLL/dT = (pred - y) * -z / T
                grad += (pred - y) * (-math.log(p / (1.0 - p)) / (t * t))
            grad /= len(probs)
            t -= self.lr * grad
            t = min(max(t, 0.05), 20.0)  # sane range
        self.temperature = t
        self._fitted = True
        return self

    def calibrate(self, probs: Sequence[float]) -> list[float]:
        out = []
        for p in probs:
            p = min(max(p, 1e-6), 1.0 - 1e-6)
            z = math.log(p / (1.0 - p)) / self.temperature
            out.append(1.0 / (1.0 + math.exp(-z)))
        return out


# --- isotonic --------------------------------------------------------------


@dataclass
class IsotonicCalibrator:
    """Isotonic regression: a non-decreasing step function from input
    probability to actual outcome rate, fit on (probs, outcomes) by
    the pool adjacent violators algorithm (PAVA).

    Most flexible of the calibrators — it can correct any monotonic
    miscalibration. Risk of overfitting on tiny sets.
    """

    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, probs: Sequence[float], outcomes: Sequence[int]) -> "IsotonicCalibrator":
        if len(probs) != len(outcomes):
            raise ValueError("length mismatch")
        if not probs:
            self._fitted = True
            return self
        # Sort by x.
        pairs = sorted(zip(probs, outcomes), key=lambda t: t[0])
        # PAVA: pool adjacent violators until monotone non-decreasing.
        values: list[list[tuple[float, int]]] = [[p, y] for p, y in pairs]
        # Working buffer of (sum_x, sum_y, n) blocks
        blocks: list[tuple[float, float, int]] = []
        for x, y in pairs:
            sx, sy, n = float(x), float(y), 1
            while blocks and (sy / n) < (blocks[-1][1] / blocks[-1][2]):
                psx, psy, pn = blocks.pop()
                sx += psx
                sy += psy
                n += pn
            blocks.append((sx, sy, n))
        # Final step function.
        self.xs = []
        self.ys = []
        # Each block represents a constant output. To turn it into a
        # step function we need the x-range. Use cumulative means to
        # get a monotonically non-decreasing y-axis.
        prev_x = -math.inf
        for sx, sy, n in blocks:
            avg_x = sx / n  # not used; we need the start of the block
            avg_y = sy / n
            self.xs.append(avg_x)
            self.ys.append(avg_y)
            prev_x = avg_x
        # Simpler approach: store (x_lo, x_hi, y) boundaries.
        # Rebuild: for each block, x_hi is the next block's avg_x (or +inf for last).
        self.xs = []
        self.ys = []
        for i, (sx, sy, n) in enumerate(blocks):
            x_lo = sx / n  # mean of x in block
            y_val = sy / n
            # The "step" starts at the i-th mean; we just store the
            # means. At query time we binary-search for the first
            # x[i] >= p.
            self.xs.append(x_lo)
            self.ys.append(y_val)
        self._fitted = True
        return self

    def calibrate(self, probs: Sequence[float]) -> list[float]:
        if not self._fitted or not self.xs:
            return [float(p) for p in probs]
        out = []
        xs_arr = self.xs
        ys_arr = self.ys
        for p in probs:
            # Find first i such that xs_arr[i] >= p.
            lo, hi = 0, len(xs_arr)
            while lo < hi:
                mid = (lo + hi) // 2
                if xs_arr[mid] < p:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= len(xs_arr):
                out.append(ys_arr[-1])
            elif lo == 0:
                out.append(ys_arr[0])
            else:
                # Linear interpolation between the bracketing points.
                x0, x1 = xs_arr[lo - 1], xs_arr[lo]
                y0, y1 = ys_arr[lo - 1], ys_arr[lo]
                if x1 == x0:
                    out.append(y0)
                else:
                    t = (p - x0) / (x1 - x0)
                    out.append(y0 + t * (y1 - y0))
        return out


# --- binning ---------------------------------------------------------------


@dataclass
class BinningCalibrator:
    """Per-bucket empirical win rate with width shrinkage.

    For each of `n_bins` buckets, compute the average outcome in
    that bucket (with optional shrinkage toward the global mean for
    sparse buckets). Faster and more interpretable than isotonic
    for very small datasets.
    """

    n_bins: int = 10
    shrink: float = 5.0  # pseudo-counts of "average outcome"
    _edges: list[float] = field(default_factory=list, init=False, repr=False)
    _values: list[float] = field(default_factory=list, init=False, repr=False)
    _global_mean: float = field(default=0.5, init=False, repr=False)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, probs: Sequence[float], outcomes: Sequence[int]) -> "BinningCalibrator":
        if len(probs) != len(outcomes):
            raise ValueError("length mismatch")
        self._fitted = True
        if not probs:
            return self
        n = len(probs)
        self._global_mean = sum(outcomes) / n
        # Bin points (closed-left, open-right except last).
        edges = [i / self.n_bins for i in range(self.n_bins + 1)]
        edges[-1] = 1.0
        sums = [0.0] * self.n_bins
        counts = [0] * self.n_bins
        for p, y in zip(probs, outcomes):
            idx = min(int(p * self.n_bins), self.n_bins - 1)
            sums[idx] += y
            counts[idx] += 1
        values = []
        denom = self.shrink
        for s, c in zip(sums, counts):
            # Shrunk mean: (s + shrink * global) / (c + shrink).
            # If the bucket is empty and shrinkage is also zero, fall
            # back to the global mean so we never divide by zero.
            if c == 0 and self.shrink == 0:
                values.append(self._global_mean)
            else:
                v = (s + self.shrink * self._global_mean) / (c + denom)
                values.append(v)
        self._edges = edges
        self._values = values
        return self

    def calibrate(self, probs: Sequence[float]) -> list[float]:
        if not self._fitted:
            return [float(p) for p in probs]
        out = []
        for p in probs:
            idx = min(int(p * self.n_bins), self.n_bins - 1)
            out.append(self._values[idx])
        return out
