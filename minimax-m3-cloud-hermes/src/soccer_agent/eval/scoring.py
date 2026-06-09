"""Pure scoring helpers used by the self-eval loop and the eval harness.

These are intentionally tiny and stateless — they take a ReasonerOutput
(or its factors) and an actual outcome string, and return a scalar that
the database can store. They live in `eval/` rather than `reasoners/`
because the *judge* role is evaluation, not prediction.

Brier score convention: 3-class (home/draw/away) on the simplex,
sum((p_i - y_i)^2) / 2 where y is one-hot. Range: 0.0 (perfect) to
1.0 (totally wrong with certainty).

`top_factor_hit` is a cheap proxy: did *any* factor whose sign agrees
with the actual outcome have non-trivial weight? Returns True/False/None
(None when there are no factors at all).
"""
from __future__ import annotations

from typing import Any, Mapping

from ..models import ReasonerOutput


def brier(probs: Mapping[str, float], actual: str) -> float:
    """3-way Brier score on home/draw/away simplex.

    `probs` is any mapping; missing keys are treated as 0.0. Robust
    to reasoners that only emit a subset of the three outcomes.
    """
    classes = ("home", "draw", "away")
    one_hot = {c: 0.0 for c in classes}
    one_hot[actual] = 1.0
    s = 0.0
    for c in classes:
        p = float(probs.get(c, 0.0))
        y = one_hot[c]
        s += (p - y) ** 2
    return s / 2.0


def top_factor_hit(
    reasoner_output: ReasonerOutput,
    actual: str,
    *,
    weight_threshold: float = 0.0,
) -> bool | None:
    """Did any factor with sign aligned to the actual outcome contribute?

    Heuristic — useful for "which signals were right?" debugging, not
    a real attribution score. Returns None when the reasoner emitted
    no factors at all (so the eval harness can filter nulls out).
    """
    factors: list[Any] = list(reasoner_output.factors)
    if not factors:
        return None
    # Map outcome → the sign we'd expect a winning factor to carry.
    expected_sign = {"home": "positive", "draw": "neutral", "away": "negative"}
    want = expected_sign.get(actual)
    if want is None:
        return None
    for f in factors:
        if float(f.weight) <= weight_threshold:
            continue
        if f.sign == want:
            return True
    # No aligned factor found — but we did have factors, so this is a
    # *miss*, not an "unknown". Distinguishing the two is useful in UI.
    return False
