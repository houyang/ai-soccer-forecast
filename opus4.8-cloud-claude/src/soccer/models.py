from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class Outcome(StrEnum):
    HOME = "HOME"
    DRAW = "DRAW"
    AWAY = "AWAY"


class MatchOutcome(StrEnum):
    WIN = "W"
    DRAW = "D"
    LOSS = "L"


def normalize_probs(probs: dict[Outcome, float]) -> dict[Outcome, float]:
    if any(v < 0 for v in probs.values()):
        raise ValueError("probabilities must be non-negative")
    total = sum(probs.values())
    if total <= 0:
        raise ValueError(f"probability total must be positive, got {total}")
    return {k: v / total for k, v in probs.items()}


def validate_probs(probs: dict[Outcome, float]) -> dict[Outcome, float]:
    if set(probs) != set(Outcome):
        raise ValueError(f"probs must cover all outcomes, got {set(probs)}")
    if any(v < 0 for v in probs.values()):
        raise ValueError("probabilities must be non-negative")
    if abs(sum(probs.values()) - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1.0, got {sum(probs.values())}")
    return probs


@dataclass(frozen=True)
class MatchRef:
    id: str
    competition: str
    home: str
    away: str
    kickoff: datetime
    venue_id: str
    season: str


@dataclass(frozen=True)
class PlayerStatus:
    name: str
    status: str  # "out" | "doubtful"
    reason: str


@dataclass(frozen=True)
class TeamForm:
    team: str
    last_n: tuple[MatchOutcome, ...]
    gf: int
    ga: int
    points: int
    streak: str
    as_of: datetime
    source: str


@dataclass(frozen=True)
class InjuryReport:
    team: str
    out: tuple[PlayerStatus, ...]
    doubtful: tuple[PlayerStatus, ...]
    as_of: datetime
    source: str


@dataclass(frozen=True)
class PastMeeting:
    date: datetime
    home: str
    away: str
    home_goals: int
    away_goals: int


@dataclass(frozen=True)
class H2HRecord:
    home: str
    away: str
    meetings: tuple[PastMeeting, ...]
    home_wins: int
    draws: int
    away_wins: int
    source: str


@dataclass(frozen=True)
class WeatherReport:
    venue_id: str
    kickoff: datetime
    temp_c: float
    wind_kph: float
    precip_mm: float
    condition: str
    source: str


@dataclass(frozen=True)
class VenueInfo:
    venue_id: str
    name: str
    city: str
    surface: str
    capacity: int
    altitude_m: int
    home_advantage_hint: float
    source: str


@dataclass(frozen=True)
class OddsSnapshot:
    bookmaker: str
    home: float
    draw: float
    away: float
    as_of: datetime
    source: str

    def __post_init__(self) -> None:
        for label, value in (("home", self.home), ("draw", self.draw), ("away", self.away)):
            if value <= 0:
                raise ValueError(f"decimal odds must be > 0, got {label}={value}")

    @property
    def implied_probs(self) -> dict[Outcome, float]:
        raw = {
            Outcome.HOME: 1.0 / self.home,
            Outcome.DRAW: 1.0 / self.draw,
            Outcome.AWAY: 1.0 / self.away,
        }
        return normalize_probs(raw)


@dataclass(frozen=True)
class MatchDossier:
    match: MatchRef
    form: dict[str, TeamForm | None]
    injuries: dict[str, InjuryReport | None]
    h2h: H2HRecord | None
    weather: WeatherReport | None
    venue: VenueInfo | None
    odds: OddsSnapshot | None
    missing: tuple[str, ...]


@dataclass(frozen=True)
class Prediction:
    id: str
    match_ref: MatchRef
    created_at: datetime
    probs: dict[Outcome, float]
    pick: Outcome
    confidence: float
    rationale: str
    market_probs: dict[Outcome, float] | None
    dossier_digest: str
    reasoner_name: str

    def __post_init__(self) -> None:
        validate_probs(self.probs)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")


@dataclass(frozen=True)
class MatchResult:
    match_id: str
    home_goals: int
    away_goals: int
    status: str
    source: str

    @property
    def outcome(self) -> Outcome:
        if self.home_goals > self.away_goals:
            return Outcome.HOME
        if self.home_goals < self.away_goals:
            return Outcome.AWAY
        return Outcome.DRAW


@dataclass(frozen=True)
class Evaluation:
    prediction_id: str
    result: MatchResult
    correct: bool
    brier: float
    log_loss: float
    beat_market: bool
    self_critique: str
    evaluated_at: datetime


def make_prediction_id(match_id: str, created_at: datetime) -> str:
    raw = f"{match_id}:{created_at.isoformat()}".encode()
    return hashlib.sha256(raw).hexdigest()[:12]


def _probs_to_dict(probs: dict[Outcome, float] | None) -> dict[str, float] | None:
    if probs is None:
        return None
    return {k.value: v for k, v in probs.items()}


def _probs_from_dict(raw: dict[str, float] | None) -> dict[Outcome, float] | None:
    if raw is None:
        return None
    return {Outcome(k): v for k, v in raw.items()}


def _ref_to_dict(ref: MatchRef) -> dict[str, Any]:
    return {
        "id": ref.id,
        "competition": ref.competition,
        "home": ref.home,
        "away": ref.away,
        "kickoff": ref.kickoff.isoformat(),
        "venue_id": ref.venue_id,
        "season": ref.season,
    }


def _ref_from_dict(raw: dict[str, Any]) -> MatchRef:
    return MatchRef(
        id=raw["id"],
        competition=raw["competition"],
        home=raw["home"],
        away=raw["away"],
        kickoff=datetime.fromisoformat(raw["kickoff"]),
        venue_id=raw["venue_id"],
        season=raw["season"],
    )


def prediction_to_dict(p: Prediction) -> dict[str, Any]:
    return {
        "id": p.id,
        "match_ref": _ref_to_dict(p.match_ref),
        "created_at": p.created_at.isoformat(),
        "probs": _probs_to_dict(p.probs),
        "pick": p.pick.value,
        "confidence": p.confidence,
        "rationale": p.rationale,
        "market_probs": _probs_to_dict(p.market_probs),
        "dossier_digest": p.dossier_digest,
        "reasoner_name": p.reasoner_name,
    }


def prediction_from_dict(raw: dict[str, Any]) -> Prediction:
    return Prediction(
        id=raw["id"],
        match_ref=_ref_from_dict(raw["match_ref"]),
        created_at=datetime.fromisoformat(raw["created_at"]),
        probs=_probs_from_dict(raw["probs"]),  # type: ignore[arg-type]
        pick=Outcome(raw["pick"]),
        confidence=raw["confidence"],
        rationale=raw["rationale"],
        market_probs=_probs_from_dict(raw["market_probs"]),
        dossier_digest=raw["dossier_digest"],
        reasoner_name=raw["reasoner_name"],
    )


def result_to_dict(r: MatchResult) -> dict[str, Any]:
    return {
        "match_id": r.match_id,
        "home_goals": r.home_goals,
        "away_goals": r.away_goals,
        "status": r.status,
        "source": r.source,
    }


def result_from_dict(raw: dict[str, Any]) -> MatchResult:
    return MatchResult(
        match_id=raw["match_id"],
        home_goals=raw["home_goals"],
        away_goals=raw["away_goals"],
        status=raw["status"],
        source=raw["source"],
    )


def evaluation_to_dict(e: Evaluation) -> dict[str, Any]:
    return {
        "prediction_id": e.prediction_id,
        "result": result_to_dict(e.result),
        "correct": e.correct,
        "brier": e.brier,
        "log_loss": e.log_loss,
        "beat_market": e.beat_market,
        "self_critique": e.self_critique,
        "evaluated_at": e.evaluated_at.isoformat(),
    }


def evaluation_from_dict(raw: dict[str, Any]) -> Evaluation:
    return Evaluation(
        prediction_id=raw["prediction_id"],
        result=result_from_dict(raw["result"]),
        correct=raw["correct"],
        brier=raw["brier"],
        log_loss=raw["log_loss"],
        beat_market=raw["beat_market"],
        self_critique=raw["self_critique"],
        evaluated_at=datetime.fromisoformat(raw["evaluated_at"]),
    )
