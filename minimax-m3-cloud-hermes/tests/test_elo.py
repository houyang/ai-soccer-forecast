"""Tests for src/soccer_agent/elo.py (Task 27).

The numeric reasoner's Elo path currently uses a placeholder
(DEFAULT_ELO for every team), so it carries no information. Task 27
replaces that with a proper Elo model that:

  - Maintains **separate home and away ratings** per team
    (because home advantage is real, but varies by team).
  - Uses a **form window** of last N matches to weight recent
    matches more heavily in the expected-outcome calculation.
  - Supports **load from / dump to JSON** so the state is
    shareable across runs (compute once, reuse forever).

These tests pin the core invariants of the Elo math and the
home/away split, plus the I/O contract.
"""

from __future__ import annotations

import json
import math
import pytest

from soccer_agent.elo import (
    DEFAULT_ELO,
    EloRating,
    EloState,
    MatchResult,
    expected,
    update_state,
    predict_proba,
    weight_for,
)


# ---------- core expected() ----------------------------------------------

def test_expected_is_symmetric_at_1500():
    """Two equal teams → 0.5/0.5."""
    assert expected(1500, 1500) == pytest.approx(0.5, abs=1e-9)


def test_expected_round_trip():
    """expected(A,B) + expected(B,A) == 1.0 (always)."""
    for a, b in [(1500, 1500), (1600, 1450), (1200, 1800), (2000, 1000)]:
        assert expected(a, b) + expected(b, a) == pytest.approx(1.0, abs=1e-9)


def test_expected_higher_rating_wins():
    """Higher-rated team has > 0.5 expected probability."""
    assert expected(1600, 1500) > 0.5
    assert expected(1700, 1500) > expected(1600, 1500)
    # 100-elo gap ≈ 0.640; 200-elo gap ≈ 0.760
    assert expected(1600, 1500) == pytest.approx(0.640, abs=0.005)
    assert expected(1700, 1500) == pytest.approx(0.760, abs=0.005)


def test_expected_100_elo_gap():
    """Classic result: 100-elo gap ≈ 64% win rate."""
    assert expected(1600, 1500) == pytest.approx(0.640, abs=0.005)


# ---------- update_state() -----------------------------------------------

def test_update_increases_winner_decreases_loser():
    """H=0 so the math is the textbook Elo case: 50% expected, K=20
    per side, → +10/-10."""
    state = EloState(home_advantage=0.0)
    state.ensure("a"); state.ensure("b")
    update_state(
        state,
        MatchResult(home_id="a", away_id="b", home_goals=2, away_goals=0, k=20.0),
    )
    # K=20, actual=1, expected=0.5 → home rating up 10, away rating down 10.
    assert state.ratings["a"].home == pytest.approx(1510.0, abs=1e-9)
    assert state.ratings["b"].away == pytest.approx(1490.0, abs=1e-9)
    # b's home is untouched (still 1500) so overall is the mean.
    assert state.ratings["b"].overall == pytest.approx(1495.0, abs=1e-9)


def test_update_k_zero_means_no_change():
    state = EloState(home_advantage=0.0)
    state.ensure("a"); state.ensure("b")
    a0 = state.ratings["a"].overall
    b0 = state.ratings["b"].overall
    update_state(
        state,
        MatchResult(home_id="a", away_id="b", home_goals=2, away_goals=0, k=0.0),
    )
    assert state.ratings["a"].overall == a0
    assert state.ratings["b"].overall == b0


def test_update_draw_balances_ratings():
    """H=0 + 50/50 expected + draw → both ratings stay put."""
    state = EloState(home_advantage=0.0)
    state.ensure("a"); state.ensure("b")
    a0 = state.ratings["a"].overall
    b0 = state.ratings["b"].overall
    update_state(
        state,
        MatchResult(home_id="a", away_id="b", home_goals=1, away_goals=1, k=20.0),
    )
    # When actual=0.5 for both and expected=0.5 for both, deltas are 0.
    assert state.ratings["a"].overall == pytest.approx(a0, abs=1e-9)
    assert state.ratings["b"].overall == pytest.approx(b0, abs=1e-9)


# ---------- home / away split --------------------------------------------

def test_home_away_split_after_repeated_home_wins():
    """If team A keeps winning at home, A's home_elo rises but A's
    away_elo stays flat. This is the key reason to split ratings."""
    state = EloState(k=10.0)
    state.ensure("a"); state.ensure("b")
    for _ in range(20):
        update_state(
            state,
            MatchResult(home_id="a", away_id="b", home_goals=2, away_goals=0, k=10.0),
        )
    a = state.ratings["a"]
    # Home rating should converge near the asymptotic limit (a few
    # hundred points above 1500) and stay well above the away rating
    # (which has never moved from 1500).
    assert a.home > a.away
    assert a.home > 1550
    assert a.away == pytest.approx(1500.0, abs=1e-9)  # never played as away


def test_home_away_split_uses_home_advantage_delta_in_expected():
    """The expected-probability for an upcoming home game should be
    higher than the overall-elo-based expected probability, by the
    configured HOME_ADVANTAGE."""
    state = EloState(k=20.0, home_advantage=50.0)
    state.ensure("a"); state.ensure("b")
    # Both teams have played no games → both at 1500/1500.
    p_home, p_away, p_draw = predict_proba(state, "a", "b")
    # With a 50-elo home advantage, home is slightly favoured
    # over a coin flip. 50-elo raw = 1/(1+10^-0.125) ≈ 0.5715.
    # We carve out a 0.27 draw bucket → 0.5715 * 0.73 ≈ 0.4172.
    # So: p_home > p_away, p_home > 0.4, and the simplex sums to 1.
    assert p_home > p_away
    assert p_home > 0.4
    assert p_home < 0.5
    assert p_home + p_away + p_draw == pytest.approx(1.0, abs=1e-9)


# ---------- form window ---------------------------------------------------

def test_weight_for_decay():
    """weight_for(n) is 1.0 at n=0 and 0.0 beyond the form window."""
    assert weight_for(0, window=5) == pytest.approx(1.0, abs=1e-9)
    assert weight_for(2, window=5) == pytest.approx(0.6, abs=0.01)  # 1 - 2/5
    assert weight_for(4, window=5) == pytest.approx(0.2, abs=0.01)
    assert weight_for(5, window=5) == pytest.approx(0.0, abs=1e-9)
    assert weight_for(10, window=5) == pytest.approx(0.0, abs=1e-9)


# ---------- persistence ---------------------------------------------------

def test_state_round_trip_through_json(tmp_path):
    state = EloState(k=24.0, home_advantage=60.0)
    state.ensure("man_city")
    state.ensure("real_madrid")
    state.ratings["man_city"].overall = 1700.0
    state.ratings["man_city"].home = 1750.0
    state.ratings["man_city"].away = 1650.0
    state.ratings["real_madrid"].overall = 1650.0
    state.ratings["real_madrid"].home = 1670.0
    state.ratings["real_madrid"].away = 1630.0

    path = tmp_path / "elo_state.json"
    state.to_json(path)
    loaded = EloState.from_json(path)
    assert loaded.k == 24.0
    assert loaded.home_advantage == 60.0
    assert loaded.ratings["man_city"].overall == 1700.0
    assert loaded.ratings["man_city"].home == 1750.0
    assert loaded.ratings["man_city"].away == 1650.0
    assert loaded.ratings["real_madrid"].home == 1670.0


def test_state_ensure_is_idempotent():
    state = EloState()
    state.ensure("x")
    s_before = state.ratings["x"].overall
    state.ensure("x")
    assert state.ratings["x"].overall == s_before
    assert len(state.ratings) == 1


# ---------- predict_proba() ----------------------------------------------

def test_predict_proba_sums_to_one():
    state = EloState()
    for t in ("a", "b", "c"):
        state.ensure(t)
    p_h, p_a, p_d = predict_proba(state, "a", "b")
    assert p_h + p_a + p_d == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= p_h <= 1.0
    assert 0.0 <= p_a <= 1.0
    assert 0.0 <= p_d <= 1.0


def test_predict_proba_swaps_home_away():
    """Predicting 'a at home' vs 'a away' should differ once each
    team has built up venue-specific ratings."""
    state = EloState(k=20.0, home_advantage=80.0)
    state.ensure("a"); state.ensure("b")
    # Make a's home rating higher than b's away rating.
    state.ratings["a"].home = 1600.0
    state.ratings["b"].away = 1500.0
    # Predict a as home (gets a.home=1600 + H=80 boost) vs
    # a as away (uses a.away, default 1500, no boost).
    p_a_home, _, _ = predict_proba(state, "a", "b")
    p_a_away, _, _ = predict_proba(state, "b", "a")
    assert p_a_home > p_a_away
