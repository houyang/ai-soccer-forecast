# src/soccer/tools/http_stubs.py
"""Real HTTP adapters. Wiring is sketched; endpoints are intentionally not
chosen yet (AGENTS.md: do not invent endpoints). Filling in a concrete API must
not change any caller — the Protocol signatures are identical to the fixture
providers."""

from __future__ import annotations

from datetime import datetime

from soccer.models import (
    H2HRecord,
    InjuryReport,
    MatchRef,
    MatchResult,
    OddsSnapshot,
    TeamForm,
    VenueInfo,
    WeatherReport,
)

_NOT_WIRED = "HTTP provider not wired to a concrete API yet; use provider_mode=fixture"


class HttpFormProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_form(self, team: str, as_of: datetime) -> TeamForm:
        raise NotImplementedError(_NOT_WIRED)


class HttpInjuryProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_injuries(self, team: str, as_of: datetime) -> InjuryReport:
        raise NotImplementedError(_NOT_WIRED)


class HttpH2HProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_h2h(self, home: str, away: str) -> H2HRecord:
        raise NotImplementedError(_NOT_WIRED)


class HttpWeatherProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_weather(self, venue_id: str, kickoff: datetime) -> WeatherReport:
        raise NotImplementedError(_NOT_WIRED)


class HttpVenueProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_venue(self, venue_id: str) -> VenueInfo:
        raise NotImplementedError(_NOT_WIRED)


class HttpOddsProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_odds(self, match: MatchRef) -> OddsSnapshot:
        raise NotImplementedError(_NOT_WIRED)


class HttpResultProvider:
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    def get_result(self, match: MatchRef) -> MatchResult | None:
        raise NotImplementedError(_NOT_WIRED)
