"""Poisson scoreline model for the group-stage matches.

Each national-team rating (0-100) is turned into an effective rating after per-match
host/home, travel (jet lag), and weather adjustments, then the rating gap becomes a goal
supremacy that splits a baseline match total into two Poisson means. The independent
Poisson scoreline matrix yields the modal exact score and the win/draw/win probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from soccer.worldcup.entities import WcMatch, WorldCup
from soccer.worldcup.ranking import Rankings

BASE_MATCH_GOALS = 2.6  # typical World Cup goals per match, split between the sides
SUPREMACY_PER_10 = 0.62  # goal supremacy added per 10 effective rating points of edge
LAMBDA_FLOOR = 0.18  # no team's expected goals drops below this
MAX_GOALS = 8  # scoreline matrix dimension (0..MAX_GOALS each side)

HOST_HOME_FIELD = 4.0  # host nation playing in its own country
# Effective-rating penalty (points) by travel distance to the North American hosts.
TRAVEL_PENALTY = {
    "CONCACAF": 0.0,
    "CONMEBOL": 0.5,
    "UEFA": 1.0,
    "CAF": 1.5,
    "AFC": 2.0,
    "OFC": 2.5,
}
WEATHER_PENALTY = 0.8  # cool-climate side playing in a hot/humid venue
_HOT_VENUE_HINTS = (
    "Miami",
    "Houston",
    "Dallas",
    "Arlington",
    "Atlanta",
    "Monterrey",
    "Guadalajara",
    "Kansas City",
)
_HEAT_SENSITIVE = {"UEFA"}


@dataclass(frozen=True)
class MatchPrediction:
    fixture_id: int
    matchday: int
    group: str
    kickoff: datetime
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "matchday": self.matchday,
            "group": self.group,
            "kickoff": self.kickoff.isoformat(),
            "home_id": self.home_id,
            "away_id": self.away_id,
            "home_name": self.home_name,
            "away_name": self.away_name,
            "lambda_home": self.lambda_home,
            "lambda_away": self.lambda_away,
            "score_home": self.score_home,
            "score_away": self.score_away,
            "p_home": self.p_home,
            "p_draw": self.p_draw,
            "p_away": self.p_away,
            "rationale": self.rationale,
        }


def _is_hot(venue: str) -> bool:
    return any(hint in venue for hint in _HOT_VENUE_HINTS)


def _effective_rating(
    wc: WorldCup, team_id: int, base: float, *, is_home: bool, venue: str
) -> float:
    team = wc.teams[team_id]
    rating = base
    if team.is_host and is_home:
        rating += HOST_HOME_FIELD
    rating -= TRAVEL_PENALTY.get(team.confederation, 1.0)
    if _is_hot(venue) and team.confederation in _HEAT_SENSITIVE:
        rating -= WEATHER_PENALTY
    return rating


def _poisson_pmf(lam: float, k: int) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def _scoreline_matrix(lam_home: float, lam_away: float) -> list[list[float]]:
    home = [_poisson_pmf(lam_home, i) for i in range(MAX_GOALS + 1)]
    away = [_poisson_pmf(lam_away, j) for j in range(MAX_GOALS + 1)]
    return [[h * a for a in away] for h in home]


def _outcome_probs(matrix: list[list[float]]) -> tuple[float, float, float]:
    p_home = p_draw = p_away = 0.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p
    total = p_home + p_draw + p_away
    if total <= 0:
        return 0.0, 1.0, 0.0
    return p_home / total, p_draw / total, p_away / total


def _modal_score(matrix: list[list[float]]) -> tuple[int, int]:
    best = (0, 0)
    best_p = -1.0
    for i, row in enumerate(matrix):
        for j, p in enumerate(row):
            if p > best_p:
                best_p, best = p, (i, j)
    return best


def predict_match(
    wc: WorldCup,
    rankings: Rankings,
    fixture_id: int,
) -> MatchPrediction:
    match = next(m for m in wc.matches if m.fixture_id == fixture_id)
    return _predict(wc, rankings, match)


def _predict(wc: WorldCup, rankings: Rankings, match: WcMatch) -> MatchPrediction:
    home = wc.teams[match.home_id]
    away = wc.teams[match.away_id]
    base_h = rankings.teams.get(match.home_id, 50.0)
    base_a = rankings.teams.get(match.away_id, 50.0)
    eff_h = _effective_rating(wc, match.home_id, base_h, is_home=True, venue=match.venue)
    eff_a = _effective_rating(wc, match.away_id, base_a, is_home=False, venue=match.venue)

    supremacy = SUPREMACY_PER_10 * (eff_h - eff_a) / 10.0
    lam_home = max(BASE_MATCH_GOALS / 2.0 + supremacy / 2.0, LAMBDA_FLOOR)
    lam_away = max(BASE_MATCH_GOALS / 2.0 - supremacy / 2.0, LAMBDA_FLOOR)

    matrix = _scoreline_matrix(lam_home, lam_away)
    p_home, p_draw, p_away = _outcome_probs(matrix)
    score_home, score_away = _modal_score(matrix)
    rationale = (
        f"Effective rating {eff_h:.1f} vs {eff_a:.1f} -> supremacy {supremacy:+.2f}; "
        f"xG {lam_home:.2f}-{lam_away:.2f}."
    )
    return MatchPrediction(
        fixture_id=match.fixture_id,
        matchday=match.matchday,
        group=match.group,
        kickoff=match.kickoff,
        home_id=match.home_id,
        away_id=match.away_id,
        home_name=home.name,
        away_name=away.name,
        lambda_home=round(lam_home, 3),
        lambda_away=round(lam_away, 3),
        score_home=score_home,
        score_away=score_away,
        p_home=round(p_home, 4),
        p_draw=round(p_draw, 4),
        p_away=round(p_away, 4),
        rationale=rationale,
    )


def predict_group_stage(wc: WorldCup, rankings: Rankings) -> list[MatchPrediction]:
    ordered = sorted(wc.matches, key=lambda m: (m.matchday, m.group, m.fixture_id))
    return [_predict(wc, rankings, m) for m in ordered]
