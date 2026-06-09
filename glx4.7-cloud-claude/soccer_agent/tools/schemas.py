from dataclasses import dataclass
from typing import Optional


@dataclass
class FormSummary:
    team_id: str
    last_n_matches: int
    record: dict[str, int]  # {"win": 3, "draw": 1, "loss": 1}
    goals_scored: int
    goals_conceded: int
    momentum_score: float  # -1.0 to 1.0
    last_5: list[dict]


@dataclass
class H2HSummary:
    team_a_id: str
    team_b_id: str
    team_a_wins: int
    draws: int
    team_b_wins: int
    recent_meetings: list[dict]


@dataclass
class InjuryReport:
    team_id: str
    key_out: list[dict]  # {player, position, severity, return_date}
    doubtful: list[dict]
    impact_score: float  # 0-1


@dataclass
class OddsSummary:
    match_id: str
    home_win_odds: dict[str, float]
    draw_odds: dict[str, float]
    away_win_odds: dict[str, float]
    implied_prob_home: float
    value_detected: bool


@dataclass
class WeatherForecast:
    venue_id: str
    temperature_celsius: float
    condition: str  # 'clear', 'rain', 'cloudy', 'snow'
    wind_speed_kmh: float


@dataclass
class VenueInfo:
    id: str
    name: str
    capacity: Optional[int] = None
    surface: Optional[str] = None
    city: Optional[str] = None