"""Pydantic models for the prediction agent.

The agent's data shapes. JSON-serializable throughout, so Pydantic models
double as the on-the-wire format for FastAPI and the on-disk format for
SQLite JSON columns.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# -- primitives ----------------------------------------------------------------


class Team(BaseModel):
    id: str
    name: str
    country: str = "ENG"


class Venue(BaseModel):
    id: str
    name: str
    city: str
    country: str
    capacity: int
    surface: Literal["grass", "hybrid", "artificial"] = "grass"
    is_neutral: bool = False
    altitude_m: int = 0
    is_dome: bool = False
    lat: float = 0.0
    lon: float = 0.0


class Match(BaseModel):
    match_id: str
    home: Team
    away: Team
    kickoff: datetime
    venue_id: str
    competition: str = "UCL"
    round: str | None = None


# -- tool outputs --------------------------------------------------------------


class FormEntry(BaseModel):
    played: int
    won: int
    drawn: int
    lost: int
    gf: int
    ga: int
    points: int
    last5_form_string: str  # e.g. "WWDWL"


class FormOutput(BaseModel):
    home: FormEntry
    away: FormEntry


class InjuryReport(BaseModel):
    player: str
    status: Literal["out", "doubt", "questionable"]
    reported_at: datetime
    source: str
    summary: str | None = None


class InjuryOutput(BaseModel):
    home: list[InjuryReport]
    away: list[InjuryReport]


class H2HMeeting(BaseModel):
    date: datetime
    home: str  # team_id
    away: str
    home_goals: int
    away_goals: int
    competition: str


class H2HOutput(BaseModel):
    meetings: list[H2HMeeting]
    home_wins: int
    away_wins: int
    draws: int
    last_meeting: datetime | None
    last_winner: Literal["home", "away", "draw"] | None


class WeatherOutput(BaseModel):
    temp_c: float
    precip_mm: float
    wind_kph: float
    conditions: str
    is_dome: bool
    playability_risk: Literal["low", "medium", "high"]


class BookmakerOdds(BaseModel):
    name: str
    home: float
    draw: float
    away: float


class OddsOutput(BaseModel):
    bookmakers: list[BookmakerOdds]
    implied_probs: dict[str, float]  # home/draw/away
    market_consensus_pick: Literal["home", "draw", "away"]


# -- signal envelope -----------------------------------------------------------


class Signal(BaseModel):
    tool: str
    ok: bool = True
    data: dict[str, Any]
    # 'live' / 'fixture' = normal data sources;
    # 'tool' / 'registry' = ok=False error-path sources — kept in the literal
    # so the UI can colour them differently and we never silently drop them.
    source: Literal["live", "fixture", "tool", "registry"] = "live"
    error: "ToolErrorPayload | None" = None
    warnings: list[str] = Field(default_factory=list)


class ToolErrorPayload(BaseModel):
    source: str
    message: str
    retriable: bool = True


# -- reasoner outputs ----------------------------------------------------------


class Factor(BaseModel):
    name: str
    value: float
    sign: Literal["positive", "negative", "neutral"]
    weight: float


class ReasonerOutput(BaseModel):
    reasoner: str
    pick: Literal["home", "draw", "away"]
    probs: dict[str, float]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    rationale: str
    factors: list[Factor] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _probs_sum_to_one(self) -> "ReasonerOutput":
        s = sum(self.probs.values())
        if not (0.99 <= s <= 1.01):
            raise ValueError(f"probs must sum to ~1.0, got {s}")
        for k in ("home", "draw", "away"):
            if k not in self.probs:
                raise ValueError(f"probs missing key {k}")
        return self


# -- context, prediction, result -----------------------------------------------


class MatchContext(BaseModel):
    match: Match
    signals: dict[str, Signal] = Field(default_factory=dict)
    venue: Venue | None = None
    # Optional: a pre-built EloState, typically loaded from disk.
    # When present, the numeric reasoner uses it (per-team home/away
    # ratings + form-window) instead of the 1500/1500 placeholder.
    # None is the safe default for the eval harness; the agent
    # orchestrator fills it in from the configured state file.
    # Type is `Any` (not `EloState`) to avoid a circular import —
    # `elo.py` is a leaf module. The reasoner validates the type
    # at use-time.
    elo_state: Any | None = None


class Prediction(BaseModel):
    prediction_id: str
    match_id: str
    created_at: datetime
    signals: dict[str, Signal]
    reasoner_outputs: list[ReasonerOutput]
    final_pick: Literal["home", "draw", "away"]
    final_probs: dict[str, float]
    final_confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    # Task 31: raw (un-calibrated) confidence, for the dashboard's
    # before/after comparison. Same range; may equal final_confidence
    # if no calibrator is configured.
    raw_confidence: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    # Which calibrator (if any) produced final_confidence.
    # Format: "<class_key>@<competition>" — e.g. "isotonic@EPL".
    # None when no calibrator was available.
    calibrator: str | None = None
    final_rationale: str
    warnings: list[str] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)

    # Set by `evaluate()` once the result is known.
    actual: Literal["home", "draw", "away"] | None = None
    correct: bool | None = None
    brier: float | None = None

    # Convenience aliases — match the v2 plan's semantic language
    # (e.g. "what is the pick?") without forcing callers to remember
    # the `final_` prefix on the persisted fields.
    @property
    def pick(self) -> Literal["home", "draw", "away"]:
        return self.final_pick

    @property
    def confidence(self) -> float:
        return self.final_confidence

    @property
    def rationale(self) -> str:
        return self.final_rationale

    model_config = ConfigDict(extra="forbid")


class Result(BaseModel):
    match_id: str
    home_goals: int
    away_goals: int
    decided_at: datetime

    @property
    def winner(self) -> Literal["home", "draw", "away"]:
        if self.home_goals > self.away_goals:
            return "home"
        if self.home_goals < self.away_goals:
            return "away"
        return "draw"


class EvalRun(BaseModel):
    eval_id: str
    started_at: datetime
    finished_at: datetime | None
    dataset_path: str
    n_matches: int
    n_with_results: int
    metrics: dict[str, Any]  # {"numeric": {...}, "llm": {...}, "final": {...}}
    judge_score: float | None = None
    config: dict[str, Any] = Field(default_factory=dict)


# Resolve forward reference for Signal.error
Signal.model_rebuild()
