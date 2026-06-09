"""Elo ratings with home/away splits (Task 27).

This module replaces the placeholder Elo in the numeric reasoner
(where every team was implicitly at 1500) with a proper model that:

  - Tracks **separate home and away ratings per team**. Home
    advantage is real (≈ 0.6 goals in modern football) but it
    varies by team (e.g. Liverpool at Anfield is much stronger
    than Burnley at Turf Moor). A single rating loses that signal.

  - Supports a **form window**: matches more than N games back
    are down-weighted in the form aggregation. We use a linear
    decay (1 - n/window) which is simple and stable. Exponential
    decay (Elo's own K-factor already does something similar)
    is an option for later.

  - Is **persistable to JSON** so the ratings can be computed
    once from a season's worth of matches and shipped with the
    repo. The eval set is too small to derive ratings from
    scratch, so users can pre-compute on their host.

This module does NOT depend on httpx, FastAPI, the LLM, or the
DB — it's pure math + persistence. Tested in tests/test_elo.py.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_ELO: float = 1500.0
DEFAULT_K: float = 20.0
DEFAULT_HOME_ADVANTAGE: float = 50.0   # in Elo points
DEFAULT_FORM_WINDOW: int = 5           # matches


def expected(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score for A vs B.

    P(A wins) = 1 / (1 + 10^((B - A) / 400))

    With 100-elo gap: 0.640. With 200: 0.760. With 0: 0.500.
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def weight_for(matches_back: int, window: int = DEFAULT_FORM_WINDOW) -> float:
    """Linear decay for form aggregation. 1.0 at n=0, 0.0 at n>=window.

    A 5-match window gives weights [1.0, 0.8, 0.6, 0.4, 0.2].
    The 6th-and-older match contributes 0.0 (i.e. is ignored).
    """
    if matches_back < 0:
        return 0.0
    if matches_back >= window:
        return 0.0
    return 1.0 - (matches_back / window)


@dataclass
class EloRating:
    """Three ratings per team: overall, home, away.

    `overall` is kept for callers that want a single number; it's
    updated as (home + away) / 2 after each game. The form / predict
    functions use `home` or `away` depending on venue.
    """
    overall: float = DEFAULT_ELO
    home: float = DEFAULT_ELO
    away: float = DEFAULT_ELO


@dataclass
class MatchResult:
    """A single completed match for Elo update.

    k is the per-game K-factor (can be higher for newer teams or
    lower for established ones — we use a constant for simplicity).
    """
    home_id: str
    away_id: str
    home_goals: int
    away_goals: int
    k: float = DEFAULT_K


@dataclass
class EloState:
    """The full Elo state: per-team ratings + global config.

    Construct with default values, then call .ensure(team_id) for
    every team you want to track, then .update() with matches in
    chronological order. Finally, .to_json(path) for persistence.
    """
    k: float = DEFAULT_K
    home_advantage: float = DEFAULT_HOME_ADVANTAGE
    form_window: int = DEFAULT_FORM_WINDOW
    ratings: dict[str, EloRating] = field(default_factory=dict)

    def ensure(self, team_id: str) -> None:
        """Idempotently register a team at DEFAULT_ELO if not present."""
        if team_id not in self.ratings:
            self.ratings[team_id] = EloRating()

    def to_json(self, path: str | Path) -> None:
        """Dump to a JSON file. Idempotent — overwrites."""
        payload: dict[str, Any] = {
            "k": self.k,
            "home_advantage": self.home_advantage,
            "form_window": self.form_window,
            "ratings": {
                tid: {"overall": r.overall, "home": r.home, "away": r.away}
                for tid, r in self.ratings.items()
            },
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True))

    @classmethod
    def from_json(cls, path: str | Path) -> "EloState":
        """Load from a JSON file produced by to_json."""
        payload = json.loads(Path(path).read_text())
        state = cls(
            k=payload.get("k", DEFAULT_K),
            home_advantage=payload.get("home_advantage", DEFAULT_HOME_ADVANTAGE),
            form_window=payload.get("form_window", DEFAULT_FORM_WINDOW),
        )
        for tid, r in payload.get("ratings", {}).items():
            state.ratings[tid] = EloRating(
                overall=r["overall"], home=r["home"], away=r["away"],
            )
        return state


def update_state(state: EloState, match: MatchResult) -> None:
    """Update ratings in-place after a match.

    Algorithm:
      1. Compute expected scores using each team's appropriate
         rating (home for the home team, away for the away team),
         plus a home-advantage delta in the home team's favour.
      2. Determine the actual score (1.0 for win, 0.5 for draw, 0.0
         for loss) for the home team and the mirror for away.
      3. Update the home team's HOME rating and the away team's
         AWAY rating. Also update each team's OVERALL rating as
         the simple mean of their home+away.
    """
    state.ensure(match.home_id)
    state.ensure(match.away_id)
    h = state.ratings[match.home_id]
    a = state.ratings[match.away_id]

    # Expected, accounting for home advantage.
    # The home team gets the full home_advantage boost added to their
    # home rating; the away team uses their plain away rating.
    # (The H/2-split form would be equivalent on a fresh game but
    # would double-count when an asymmetry has built up in the
    # per-venue ratings themselves.)
    h_eff = h.home + state.home_advantage
    a_eff = a.away
    exp_home = expected(h_eff, a_eff)
    exp_away = 1.0 - exp_home

    # Actual.
    if match.home_goals > match.away_goals:
        act_home, act_away = 1.0, 0.0
    elif match.home_goals < match.away_goals:
        act_home, act_away = 0.0, 1.0
    else:
        act_home, act_away = 0.5, 0.5

    # Update home and away ratings separately.
    h.home += match.k * (act_home - exp_home)
    a.away += match.k * (act_away - exp_away)
    # Recompute overall as the mean of the two.
    h.overall = (h.home + h.away) / 2.0
    a.overall = (a.home + a.away) / 2.0


def predict_proba(
    state: EloState, home_id: str, away_id: str,
) -> tuple[float, float, float]:
    """Predict (p_home, p_away, p_draw) for an upcoming match.

    1. Compute raw home-win probability from Elo+home_advantage.
    2. Add a draw residual (draw rate is roughly 0.27 in soccer
       and decreases as the Elo gap grows).
    3. Return the resulting 3-way simplex.

    The teams are auto-ensured (will appear at DEFAULT_ELO if not
    seen before — which is fine for early-season predictions).
    """
    state.ensure(home_id)
    state.ensure(away_id)
    h = state.ratings[home_id]
    a = state.ratings[away_id]
    h_eff = h.home + state.home_advantage
    a_eff = a.away
    p_home_raw = expected(h_eff, a_eff)
    p_away_raw = 1.0 - p_home_raw
    # Draw rate. A 0.27 baseline for evenly-matched teams; shrinks
    # as the gap grows. Clamp to [0.15, 0.40] so we never predict
    # no-draw or always-draw.
    gap = abs(p_home_raw - p_away_raw)
    p_draw = max(0.15, min(0.40, 0.27 - gap * 0.3))
    leftover = 1.0 - p_draw
    return (
        p_home_raw * leftover,
        p_away_raw * leftover,
        p_draw,
    )
