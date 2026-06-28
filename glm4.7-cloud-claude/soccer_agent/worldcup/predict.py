# soccer_agent/worldcup/predict.py
"""Independent-Poisson scoreline model with Dixon-Coles low-score correction.

Effective rating = blend of recalibrated team strength and projected-XI mean player rating,
adjusted for host-nation home field, inter-confederation travel, and hot-venue weather.
The rating gap becomes a goal supremacy that splits a baseline match total into two Poisson
means; the scoreline matrix yields the modal exact score and W/D/L probabilities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.lineup import ProjectedLineup
from soccer_agent.worldcup.ranking import Rankings

BASE_MATCH_GOALS = 2.6
SUPREMACY_PER_10 = 0.62
LAMBDA_FLOOR = 0.18
MAX_GOALS = 8
DRAW_RHO = -0.15
HOST_HOME_FIELD = 4.0
TRAVEL_PENALTY = {"CONCACAF": 0.0, "CONMEBOL": 0.5, "UEFA": 1.0, "CAF": 1.5, "AFC": 2.0, "OFC": 2.5}
WEATHER_PENALTY = 0.8
_HOT_VENUE_HINTS = ("Miami", "Houston", "Dallas", "Arlington", "Atlanta", "Monterrey", "Guadalajara", "Kansas City")
_HEAT_SENSITIVE = {"UEFA"}
_NEUTRAL = 50.0
W_XI = 0.5  # weight on projected-XI mean rating vs team strength


@dataclass(frozen=True)
class MatchPrediction:
    fixture_id: int
    matchday: int
    group: str
    kickoff: datetime
    venue: str
    home_id: int
    away_id: int
    home_name: str
    away_name: str
    lambda_home: float
    lambda_away: float
    score_home: int
    score_away: int
    p_home: float
    p_draw: float
    p_away: float
    rationale: str
    home_adjustment: float = 0.0
    away_adjustment: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id, "matchday": self.matchday, "group": self.group,
            "kickoff": self.kickoff.isoformat(), "venue": self.venue,
            "home_id": self.home_id, "away_id": self.away_id,
            "home_name": self.home_name, "away_name": self.away_name,
            "lambda_home": round(self.lambda_home, 3), "lambda_away": round(self.lambda_away, 3),
            "score_home": self.score_home, "score_away": self.score_away,
            "p_home": round(self.p_home, 4), "p_draw": round(self.p_draw, 4), "p_away": round(self.p_away, 4),
            "rationale": self.rationale,
            "home_adjustment": round(self.home_adjustment, 3), "away_adjustment": round(self.away_adjustment, 3),
        }


def _poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def scoreline_matrix(lh: float, la: float) -> list[list[float]]:
    """9x9 (0..MAX_GOALS) scoreline probability matrix with Dixon-Coles low-score correction."""
    mat = [[_poisson(i, lh) * _poisson(j, la) for j in range(MAX_GOALS + 1)] for i in range(MAX_GOALS + 1)]
    # Dixon-Coles: adjust 0-0, 1-0, 0-1, 1-1.
    tau = lambda i, j: 1.0 - DRAW_RHO * _poisson(i, lh) * _poisson(j, la) if (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1)] else 1.0  # noqa: E731
    adj = [[mat[i][j] * tau(i, j) for j in range(MAX_GOALS + 1)] for i in range(MAX_GOALS + 1)]
    total = sum(sum(row) for row in adj)
    return [[v / total for v in row] for row in adj]


def top_scorelines(lh: float, la: float, n: int) -> list[tuple[int, int, float]]:
    mat = scoreline_matrix(lh, la)
    cells = [(i, j, mat[i][j]) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)]
    cells.sort(key=lambda c: c[2], reverse=True)
    return [(i, j, p) for i, j, p in cells[:n]]


def _xi_mean_rating(wc: WorldCup, rankings: Rankings, lineup: ProjectedLineup) -> float:
    if not lineup.start_ids:
        return _NEUTRAL
    vals = [rankings.players.get(pid, _NEUTRAL) for pid in lineup.start_ids]
    return sum(vals) / len(vals)


def effective_rating(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float],
    team_id: int, lineup: ProjectedLineup, is_home: bool, venue: str,
) -> tuple[float, float]:
    """Return (effective_rating, adjustment) for one side."""
    team = wc.teams[team_id]
    base = W_XI * strengths.get(team_id, _NEUTRAL) + (1 - W_XI) * _xi_mean_rating(wc, rankings, lineup)
    adj = 0.0
    if team.is_host and is_home:
        adj += HOST_HOME_FIELD
    adj -= TRAVEL_PENALTY.get(team.confederation, 1.0)
    if any(hint in venue for hint in _HOT_VENUE_HINTS) and team.confederation in _HEAT_SENSITIVE:
        adj -= WEATHER_PENALTY
    return base + adj, adj


def predict_one(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float],
    fixture_id: int, home_lu: ProjectedLineup, away_lu: ProjectedLineup,
) -> MatchPrediction:
    m = next((x for x in wc.matches if x.fixture_id == fixture_id), None)
    if m is None:
        raise ValueError(f"fixture {fixture_id} not found")
    eff_h, adj_h = effective_rating(wc, rankings, strengths, m.home_id, home_lu, True, m.venue)
    eff_a, adj_a = effective_rating(wc, rankings, strengths, m.away_id, away_lu, False, m.venue)

    supremacy = (eff_h - eff_a) / 10.0 * SUPREMACY_PER_10
    total = BASE_MATCH_GOALS
    lh = max(LAMBDA_FLOOR, total / 2.0 + supremacy / 2.0)
    la = max(LAMBDA_FLOOR, total / 2.0 - supremacy / 2.0)

    mat = scoreline_matrix(lh, la)
    p_home = sum(mat[i][j] for i in range(MAX_GOALS + 1) for j in range(i))
    p_away = sum(mat[i][j] for i in range(MAX_GOALS + 1) for j in range(i + 1, MAX_GOALS + 1))
    p_draw = sum(mat[i][i] for i in range(MAX_GOALS + 1))
    # modal exact score:
    best = max(((i, j) for i in range(MAX_GOALS + 1) for j in range(MAX_GOALS + 1)), key=lambda ij: mat[ij[0]][ij[1]])
    sh, sa = best
    rationale = (
        f"Eff {eff_h:.1f} vs {eff_a:.1f} -> supremacy {supremacy:+.2f}; "
        f"xG {lh:.2f}-{la:.2f}; adj {adj_h:+.1f}/{adj_a:+.1f}."
    )
    return MatchPrediction(
        fixture_id=m.fixture_id, matchday=m.matchday, group=m.group, kickoff=m.kickoff,
        venue=m.venue, home_id=m.home_id, away_id=m.away_id,
        home_name=wc.teams[m.home_id].name, away_name=wc.teams[m.away_id].name,
        lambda_home=lh, lambda_away=la, score_home=sh, score_away=sa,
        p_home=p_home, p_draw=p_draw, p_away=p_away, rationale=rationale,
        home_adjustment=adj_h, away_adjustment=adj_a,
    )
