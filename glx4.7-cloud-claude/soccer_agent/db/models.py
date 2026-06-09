from datetime import datetime
from typing import Optional
from sqlalchemy import JSON, String, Integer, Float, DateTime, ForeignKey, Index, Boolean, Date
from sqlalchemy.orm import Mapped, mapped_column
from soccer_agent.db.base import Base


class Competition(Base):
    __tablename__ = "competitions"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'league', 'tournament'
    api_source: Mapped[str] = mapped_column(String(50), nullable=False)
    current_season: Mapped[Optional[str]] = mapped_column(String(10))


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    logo_url: Mapped[Optional[str]] = mapped_column(String(255))
    api_source: Mapped[str] = mapped_column(String(50), nullable=False)


class Venue(Base):
    __tablename__ = "venues"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    city: Mapped[Optional[str]] = mapped_column(String(100))
    country: Mapped[Optional[str]] = mapped_column(String(100))
    capacity: Mapped[Optional[int]] = mapped_column(Integer)
    surface: Mapped[Optional[str]] = mapped_column(String(50))  # 'grass', 'hybrid'
    latitude: Mapped[Optional[float]] = mapped_column(Float)
    longitude: Mapped[Optional[float]] = mapped_column(Float)


class Match(Base):
    __tablename__ = "matches"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)
    competition_id: Mapped[str] = mapped_column(ForeignKey("competitions.id"), nullable=False)
    stage: Mapped[Optional[str]] = mapped_column(String(50))  # 'group_a', 'final', etc.
    home_team_id: Mapped[Optional[str]] = mapped_column(ForeignKey("teams.id"))
    away_team_id: Mapped[Optional[str]] = mapped_column(ForeignKey("teams.id"))
    venue_id: Mapped[Optional[str]] = mapped_column(ForeignKey("venues.id"))
    kickoff_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    home_score: Mapped[Optional[int]] = mapped_column(Integer)
    away_score: Mapped[Optional[int]] = mapped_column(Integer)
    winner: Mapped[Optional[str]] = mapped_column(String(20))  # 'home', 'away', 'draw'
    status: Mapped[str] = mapped_column(String(20), default="upcoming")  # 'upcoming', 'live', 'finished'
    temperature_celsius: Mapped[Optional[float]] = mapped_column(Float)
    weather_condition: Mapped[Optional[str]] = mapped_column(String(50))
    wind_speed_kmh: Mapped[Optional[float]] = mapped_column(Float)

    __table_args__ = (
        Index("idx_kickoff", "kickoff_utc"),
        Index("idx_competition", "competition_id"),
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), unique=True, nullable=False)
    predicted_outcome: Mapped[str] = mapped_column(String(20), nullable=False)  # 'home', 'draw', 'away'
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    rationale: Mapped[str] = mapped_column(String, nullable=False)
    reasoning_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    tools_used: Mapped[list] = mapped_column(JSON, nullable=False)
    model_version: Mapped[Optional[str]] = mapped_column(String(50))

    __table_args__ = (
        Index("idx_match", "match_id"),
        Index("idx_timestamp", "timestamp_utc"),
    )


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), unique=True, nullable=False)
    actual_outcome: Mapped[str] = mapped_column(String(20), nullable=False)
    correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    calibrated_confidence: Mapped[Optional[float]] = mapped_column(Float)
    self_reflection: Mapped[Optional[str]] = mapped_column(String)
    reflection_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    tools_used_correctly: Mapped[Optional[dict]] = mapped_column(JSON)
    missed_factors: Mapped[Optional[list]] = mapped_column(JSON)

    __table_args__ = (
        Index("idx_prediction", "prediction_id"),
        Index("idx_correct", "correct"),
    )


class Metrics(Base):
    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    competition_id: Mapped[Optional[str]] = mapped_column(String(50))
    period_start: Mapped[datetime] = mapped_column(Date, nullable=False)
    period_end: Mapped[datetime] = mapped_column(Date, nullable=False)
    total_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    correct_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    accuracy_rate: Mapped[float] = mapped_column(Float, nullable=False)
    avg_confidence: Mapped[Optional[float]] = mapped_column(Float)
    avg_confidence_when_correct: Mapped[Optional[float]] = mapped_column(Float)
    avg_confidence_when_wrong: Mapped[Optional[float]] = mapped_column(Float)
    home_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    draw_accuracy: Mapped[Optional[float]] = mapped_column(Float)
    away_accuracy: Mapped[Optional[float]] = mapped_column(Float)

    __table_args__ = (
        Index("idx_competition_period", "competition_id", "period_start", "period_end"),
    )


class ToolError(Base):
    __tablename__ = "tool_errors"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tool_name: Mapped[str] = mapped_column(String(50), nullable=False)
    match_id: Mapped[Optional[str]] = mapped_column(String(50))
    error_message: Mapped[str] = mapped_column(String, nullable=False)
    timestamp_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)