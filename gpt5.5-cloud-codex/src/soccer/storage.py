"""Prediction persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Protocol, TypeVar, cast

from soccer.models import (
    HeadToHeadRecord,
    InjuryReport,
    MatchEvidence,
    MatchRequest,
    MatchResult,
    OddsQuote,
    Outcome,
    Prediction,
    PredictionRecord,
    TeamForm,
    Venue,
    Weather,
)


class PredictionLog(Protocol):
    def append(self, record: PredictionRecord) -> None:
        """Store a prediction record."""

    def attach_result(self, result: MatchResult) -> PredictionRecord:
        """Attach a final result to an existing prediction."""

    def list_records(self) -> list[PredictionRecord]:
        """Return all stored prediction records."""


class JsonlPredictionLog:
    """Append-only JSONL prediction log."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: PredictionRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(_to_json(record), sort_keys=True) + "\n")

    def attach_result(self, result: MatchResult) -> PredictionRecord:
        records = self.list_records()
        updated: PredictionRecord | None = None
        next_records: list[PredictionRecord] = []
        for record in records:
            if record.request.match_id == result.match_id:
                updated = replace(record, result=result)
                next_records.append(updated)
            else:
                next_records.append(record)
        if updated is None:
            raise LookupError(f"No prediction found for match_id={result.match_id!r}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as file:
            for record in next_records:
                file.write(json.dumps(_to_json(record), sort_keys=True) + "\n")
        return updated

    def list_records(self) -> list[PredictionRecord]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    records.append(_record_from_json(json.loads(line)))
        return records


def _to_json(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_to_json(item) for item in value]
    if isinstance(value, dict):
        return {str(_to_json(key)): _to_json(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _to_json(item) for key, item in asdict(value).items()}
    return value


T = TypeVar("T")


def _as_tuple(data: object) -> tuple[str, ...]:
    if not isinstance(data, list):
        return ()
    return tuple(str(item) for item in data)


def _typed_dict(data: object) -> dict[str, object]:
    if not isinstance(data, dict):
        raise TypeError("Expected stored JSON object to be a dictionary")
    return cast(dict[str, object], data)


def _required(data: dict[str, object], key: str, expected_type: type[T]) -> T:
    value = data[key]
    if not isinstance(value, expected_type):
        raise TypeError(f"Expected {key!r} to be {expected_type.__name__}")
    return value


def _request_from_json(data: object) -> MatchRequest:
    item = _typed_dict(data)
    return MatchRequest(
        match_id=_required(item, "match_id", str),
        competition=_required(item, "competition", str),
        home_team=_required(item, "home_team", str),
        away_team=_required(item, "away_team", str),
        kickoff=datetime.fromisoformat(_required(item, "kickoff", str)),
        neutral_site=_required(item, "neutral_site", bool),
    )


def _form_from_json(data: object) -> TeamForm:
    item = _typed_dict(data)
    return TeamForm(
        team=_required(item, "team", str),
        matches=_required(item, "matches", int),
        wins=_required(item, "wins", int),
        draws=_required(item, "draws", int),
        losses=_required(item, "losses", int),
        goals_for=_required(item, "goals_for", int),
        goals_against=_required(item, "goals_against", int),
    )


def _injuries_from_json(data: object) -> InjuryReport:
    item = _typed_dict(data)
    return InjuryReport(
        team=_required(item, "team", str),
        unavailable=_as_tuple(item.get("unavailable")),
        doubtful=_as_tuple(item.get("doubtful")),
        source=_required(item, "source", str),
    )


def _h2h_from_json(data: object) -> HeadToHeadRecord:
    item = _typed_dict(data)
    return HeadToHeadRecord(
        home_team_wins=_required(item, "home_team_wins", int),
        draws=_required(item, "draws", int),
        away_team_wins=_required(item, "away_team_wins", int),
        meetings=_required(item, "meetings", int),
        summary=_required(item, "summary", str),
    )


def _venue_from_json(data: object) -> Venue:
    item = _typed_dict(data)
    home_team = item.get("home_team")
    return Venue(
        name=_required(item, "name", str),
        city=_required(item, "city", str),
        country=_required(item, "country", str),
        home_team=home_team if isinstance(home_team, str) else None,
    )


def _weather_from_json(data: object) -> Weather:
    item = _typed_dict(data)
    return Weather(
        temperature_c=float(_required(item, "temperature_c", float)),
        wind_kph=float(_required(item, "wind_kph", float)),
        precipitation_mm=float(_required(item, "precipitation_mm", float)),
        summary=_required(item, "summary", str),
    )


def _odds_from_json(data: object) -> OddsQuote:
    item = _typed_dict(data)
    return OddsQuote(
        bookmaker=_required(item, "bookmaker", str),
        home_win=float(_required(item, "home_win", float)),
        draw=float(_required(item, "draw", float)),
        away_win=float(_required(item, "away_win", float)),
    )


def _prediction_from_json(data: object) -> Prediction:
    item = _typed_dict(data)
    probabilities = _typed_dict(item["probabilities"])
    return Prediction(
        match_id=_required(item, "match_id", str),
        outcome=Outcome(_required(item, "outcome", str)),
        confidence=float(_required(item, "confidence", float)),
        rationale=_required(item, "rationale", str),
        probabilities={
            Outcome(key): float(value)
            for key, value in probabilities.items()
            if isinstance(value, int | float)
        },
    )


def _evidence_from_json(data: object) -> MatchEvidence:
    item = _typed_dict(data)
    odds_data = item.get("odds", [])
    if not isinstance(odds_data, list):
        odds_data = []
    return MatchEvidence(
        request=_request_from_json(item["request"]),
        home_form=_form_from_json(item["home_form"]),
        away_form=_form_from_json(item["away_form"]),
        home_injuries=_injuries_from_json(item["home_injuries"]),
        away_injuries=_injuries_from_json(item["away_injuries"]),
        head_to_head=_h2h_from_json(item["head_to_head"]),
        venue=_venue_from_json(item["venue"]),
        weather=_weather_from_json(item["weather"]),
        odds=tuple(_odds_from_json(quote) for quote in odds_data),
    )


def _result_from_json(data: object) -> MatchResult | None:
    if data is None:
        return None
    item = _typed_dict(data)
    return MatchResult(
        match_id=_required(item, "match_id", str),
        home_score=_required(item, "home_score", int),
        away_score=_required(item, "away_score", int),
        completed_at=datetime.fromisoformat(_required(item, "completed_at", str)),
    )


def _record_from_json(data: object) -> PredictionRecord:
    item = _typed_dict(data)
    return PredictionRecord(
        request=_request_from_json(item["request"]),
        evidence=_evidence_from_json(item["evidence"]),
        prediction=_prediction_from_json(item["prediction"]),
        created_at=datetime.fromisoformat(_required(item, "created_at", str)),
        result=_result_from_json(item.get("result")),
    )
