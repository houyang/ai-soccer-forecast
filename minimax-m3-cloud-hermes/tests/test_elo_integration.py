"""Tests for the Elo integration in the numeric reasoner (Task 27).

These tests pin the *wiring* — that the reasoner actually uses the
provided EloState, and falls back gracefully when none is provided.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from soccer_agent.elo import EloState
from soccer_agent.models import (
    Match,
    MatchContext,
    Signal,
    Team,
)
from soccer_agent.reasoners.numeric import (
    DEFAULT_ELO,
    NumericReasoner,
    _elo_expected,
    run,
)


def _match(home_id: str = "a", away_id: str = "b") -> Match:
    return Match(
        match_id=f"{home_id}_vs_{away_id}",
        competition="TEST",
        round="test",
        venue_id="test_venue",
        kickoff=datetime(2025, 1, 1, 20, 0, tzinfo=__import__("datetime").timezone.utc),
        home=Team(id=home_id, name=home_id),
        away=Team(id=away_id, name=away_id),
    )


def _ctx_with_elo(elo: EloState | None) -> MatchContext:
    return MatchContext(match=_match(), signals={}, elo_state=elo)


# ---------- backward compatibility ---------------------------------------

def test_no_elo_state_means_placeholder_behavior():
    """Without an elo_state, the reasoner should fall back to the
    1500/1500 placeholder (its historical behavior)."""
    out = run(_ctx_with_elo(None))
    # The probability math should look the same as before.
    assert 0.30 < out.probs["home"] < 0.45
    assert 0.20 < out.probs["draw"] < 0.30
    assert 0.30 < out.probs["away"] < 0.45


def test_elo_state_with_unseen_teams_uses_defaults():
    """An elo_state that has no entries for the match's teams should
    fall back to the placeholder behavior (auto-ensures at 1500)."""
    state = EloState()
    state.ensure("some_other_team")
    out = run(_ctx_with_elo(state))
    assert 0.30 < out.probs["home"] < 0.45


# ---------- the new behavior --------------------------------------------

def test_strong_home_team_picked_home_with_higher_confidence():
    """When a's home rating is 1700 and b's away is 1500, the home
    team should be picked and have a confidence > the baseline."""
    state = EloState(home_advantage=0.0)
    state.ensure("a"); state.ensure("b")
    state.ratings["a"].home = 1700.0  # a is a beast at home
    state.ratings["a"].away = 1500.0
    state.ratings["b"].home = 1500.0
    state.ratings["b"].away = 1500.0
    out = run(_ctx_with_elo(state))
    # 200-elo gap (a.home=1700, b.away=1500) with H=0 → ~0.76 raw
    # minus draw residual → ~0.55. Pick should be 'home'.
    assert out.pick == "home"
    assert out.probs["home"] > 0.50
    assert out.probs["home"] > out.probs["away"]


def test_strong_away_team_picked_away():
    """Mirror case: b is much stronger away. a is at home but weak."""
    state = EloState(home_advantage=0.0)
    state.ensure("a"); state.ensure("b")
    state.ratings["a"].home = 1300.0
    state.ratings["b"].away = 1700.0
    out = run(_ctx_with_elo(state))
    # a.home=1300, b.away=1700 → exp(a) = 1/(1+10^(400/400)) = 0.10.
    # 10% raw × 0.73 leftover = 7.3% home. Pick must be 'away'.
    assert out.pick == "away"
    assert out.probs["away"] > out.probs["home"]


def test_elo_state_replaces_hardcoded_placeholder():
    """A 50-elo difference should produce a visible shift in the
    reasoner output. Pre-Task 27 the reasoner ignored Elo entirely;
    the two outputs should differ."""
    state = EloState(home_advantage=0.0)
    state.ensure("a"); state.ensure("b")
    # equal teams: baseline
    out_equal = run(_ctx_with_elo(state))
    # now give a a real home-rating boost
    state.ratings["a"].home = 1600.0
    out_boosted = run(_ctx_with_elo(state))
    assert out_boosted.probs["home"] > out_equal.probs["home"]
