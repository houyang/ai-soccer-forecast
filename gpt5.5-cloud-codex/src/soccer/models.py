"""Core data models for soccer predictions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class Outcome(StrEnum):
    """A three-way soccer match outcome."""

    HOME_WIN = "home_win"
    DRAW = "draw"
    AWAY_WIN = "away_win"


@dataclass(frozen=True)
class MatchRequest:
    """A match the agent should evaluate."""

    match_id: str
    competition: str
    home_team: str
    away_team: str
    kickoff: datetime
    neutral_site: bool = False


@dataclass(frozen=True)
class TeamForm:
    team: str
    matches: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int

    @property
    def points_per_match(self) -> float:
        if self.matches == 0:
            return 0.0
        return ((self.wins * 3) + self.draws) / self.matches

    @property
    def goal_difference_per_match(self) -> float:
        if self.matches == 0:
            return 0.0
        return (self.goals_for - self.goals_against) / self.matches


@dataclass(frozen=True)
class InjuryReport:
    team: str
    unavailable: tuple[str, ...] = ()
    doubtful: tuple[str, ...] = ()
    source: str = "unknown"

    @property
    def impact_count(self) -> int:
        return len(self.unavailable) + len(self.doubtful)


@dataclass(frozen=True)
class HeadToHeadRecord:
    home_team_wins: int
    draws: int
    away_team_wins: int
    meetings: int
    summary: str


@dataclass(frozen=True)
class Venue:
    name: str
    city: str
    country: str
    home_team: str | None = None


@dataclass(frozen=True)
class Weather:
    temperature_c: float
    wind_kph: float
    precipitation_mm: float
    summary: str


@dataclass(frozen=True)
class OddsQuote:
    bookmaker: str
    home_win: float
    draw: float
    away_win: float

    def implied_probabilities(self) -> dict[Outcome, float]:
        raw = {
            Outcome.HOME_WIN: 1 / self.home_win,
            Outcome.DRAW: 1 / self.draw,
            Outcome.AWAY_WIN: 1 / self.away_win,
        }
        total = sum(raw.values())
        return {outcome: probability / total for outcome, probability in raw.items()}


@dataclass(frozen=True)
class MatchEvidence:
    request: MatchRequest
    home_form: TeamForm
    away_form: TeamForm
    home_injuries: InjuryReport
    away_injuries: InjuryReport
    head_to_head: HeadToHeadRecord
    venue: Venue
    weather: Weather
    odds: tuple[OddsQuote, ...]


@dataclass(frozen=True)
class Prediction:
    match_id: str
    outcome: Outcome
    confidence: float
    rationale: str
    probabilities: dict[Outcome, float]


@dataclass(frozen=True)
class MatchResult:
    match_id: str
    home_score: int
    away_score: int
    completed_at: datetime

    @property
    def outcome(self) -> Outcome:
        if self.home_score > self.away_score:
            return Outcome.HOME_WIN
        if self.home_score < self.away_score:
            return Outcome.AWAY_WIN
        return Outcome.DRAW


@dataclass(frozen=True)
class PredictionRecord:
    request: MatchRequest
    evidence: MatchEvidence
    prediction: Prediction
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    result: MatchResult | None = None
