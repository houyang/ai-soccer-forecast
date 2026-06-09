"""Tool protocols used by the prediction agent."""

from __future__ import annotations

from typing import Protocol

from soccer.models import (
    HeadToHeadRecord,
    InjuryReport,
    MatchRequest,
    OddsQuote,
    TeamForm,
    Venue,
    Weather,
)


class FormTool(Protocol):
    def recent_form(self, team: str, competition: str) -> TeamForm:
        """Return recent form for a team in or near the requested competition."""


class InjuryNewsTool(Protocol):
    def injury_report(self, team: str, match: MatchRequest) -> InjuryReport:
        """Return current injury and availability news for a team."""


class HeadToHeadTool(Protocol):
    def head_to_head(self, home_team: str, away_team: str) -> HeadToHeadRecord:
        """Return head-to-head history for the matchup."""


class VenueTool(Protocol):
    def venue_for(self, match: MatchRequest) -> Venue:
        """Return venue details for the match."""


class WeatherTool(Protocol):
    def forecast(self, venue: Venue, match: MatchRequest) -> Weather:
        """Return weather forecast for the venue at kickoff."""


class OddsTool(Protocol):
    def odds_for(self, match: MatchRequest) -> tuple[OddsQuote, ...]:
        """Return bookmaker odds for the match."""
