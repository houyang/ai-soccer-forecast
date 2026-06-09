# soccer_agent/workflows/state.py
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from soccer_agent.tools.schemas import (
    FormSummary, H2HSummary, InjuryReport,
    OddsSummary, WeatherForecast, VenueInfo
)


class PredictionState(BaseModel):
    """State for the prediction workflow."""

    match_id: str
    competition_id: str
    stage: str = Field(description="Stage: group, knockout, final")

    # Tool outputs
    team_a_form: Optional[FormSummary] = None
    team_b_form: Optional[FormSummary] = None
    h2h_history: Optional[H2HSummary] = None
    injuries_a: Optional[InjuryReport] = None
    injuries_b: Optional[InjuryReport] = None
    odds: Optional[OddsSummary] = None
    weather: Optional[WeatherForecast] = None
    venue: Optional[VenueInfo] = None

    # Reasoning
    context_analysis: Optional[str] = None
    synthesized_rationale: Optional[str] = None

    # Output
    predicted_outcome: Optional[str] = Field(default=None, description="home, draw, away")
    confidence_score: Optional[float] = Field(default=None, ge=0, le=100)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EvaluationState(BaseModel):
    """State for the evaluation workflow."""

    pending_predictions: list[dict] = Field(default_factory=list)
    evaluated_predictions: list[dict] = Field(default_factory=list)
    metrics_updated: bool = False