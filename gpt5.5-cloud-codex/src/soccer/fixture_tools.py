"""Local fixture-backed tools for demos and tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from soccer.agent import PredictionAgent
from soccer.models import (
    HeadToHeadRecord,
    InjuryReport,
    MatchRequest,
    OddsQuote,
    TeamForm,
    Venue,
    Weather,
)
from soccer.reasoning import MatchupReasoner
from soccer.storage import PredictionLog


@dataclass(frozen=True)
class FixtureCatalog:
    requests: dict[str, MatchRequest]
    competitions: dict[str, tuple[str, ...]]
    forms: dict[str, TeamForm]
    injuries: dict[str, InjuryReport]
    head_to_heads: dict[tuple[str, str], HeadToHeadRecord]
    venues: dict[str, Venue]
    weather: dict[str, Weather]
    odds: dict[str, tuple[OddsQuote, ...]]

    def requests_for_competition(self, competition_id: str) -> tuple[MatchRequest, ...]:
        match_ids = self.competitions[competition_id]
        return tuple(self.requests[match_id] for match_id in match_ids)


class FixtureTools:
    """Implement all data-source protocols from a local fixture catalog."""

    def __init__(self, catalog: FixtureCatalog) -> None:
        self.catalog = catalog

    def recent_form(self, team: str, competition: str) -> TeamForm:
        del competition
        return self.catalog.forms[team]

    def injury_report(self, team: str, match: MatchRequest) -> InjuryReport:
        del match
        return self.catalog.injuries.get(team, InjuryReport(team=team, source="fixture"))

    def head_to_head(self, home_team: str, away_team: str) -> HeadToHeadRecord:
        return self.catalog.head_to_heads.get(
            (home_team, away_team),
            HeadToHeadRecord(
                home_team_wins=0,
                draws=0,
                away_team_wins=0,
                meetings=0,
                summary="No recent meetings found in fixtures",
            ),
        )

    def venue_for(self, match: MatchRequest) -> Venue:
        return self.catalog.venues[match.match_id]

    def forecast(self, venue: Venue, match: MatchRequest) -> Weather:
        del venue
        return self.catalog.weather[match.match_id]

    def odds_for(self, match: MatchRequest) -> tuple[OddsQuote, ...]:
        return self.catalog.odds.get(match.match_id, ())


def default_catalog() -> FixtureCatalog:
    """Return placeholder scenarios for architectural demos."""

    requests = {
        "ucl-final-2026": MatchRequest(
            match_id="ucl-final-2026",
            competition="UEFA Champions League 2025/26",
            home_team="European Club A",
            away_team="European Club B",
            kickoff=datetime(2026, 5, 30, 19, 0, tzinfo=UTC),
            neutral_site=True,
        ),
        "world-cup-final-2026": MatchRequest(
            match_id="world-cup-final-2026",
            competition="FIFA World Cup 2026",
            home_team="Nation A",
            away_team="Nation B",
            kickoff=datetime(2026, 7, 19, 22, 0, tzinfo=UTC),
            neutral_site=True,
        ),
        "wc-2026-match-001": MatchRequest(
            match_id="wc-2026-match-001",
            competition="FIFA World Cup 2026",
            home_team="Mexico",
            away_team="Group A Opponent",
            kickoff=datetime(2026, 6, 11, 19, 0, tzinfo=UTC),
            neutral_site=False,
        ),
        "wc-2026-match-010": MatchRequest(
            match_id="wc-2026-match-010",
            competition="FIFA World Cup 2026",
            home_team="Curacao",
            away_team="Germany",
            kickoff=datetime(2026, 6, 14, 19, 0, tzinfo=UTC),
            neutral_site=True,
        ),
        "wc-2026-final": MatchRequest(
            match_id="wc-2026-final",
            competition="FIFA World Cup 2026",
            home_team="Finalist A",
            away_team="Finalist B",
            kickoff=datetime(2026, 7, 19, 22, 0, tzinfo=UTC),
            neutral_site=True,
        ),
    }
    competitions = {
        "world-cup-2026": (
            "wc-2026-match-001",
            "wc-2026-match-010",
            "wc-2026-final",
        ),
        "demo-finals": (
            "ucl-final-2026",
            "world-cup-final-2026",
        ),
    }

    forms = {
        "European Club A": TeamForm("European Club A", 6, 4, 1, 1, 13, 6),
        "European Club B": TeamForm("European Club B", 6, 3, 2, 1, 10, 7),
        "Nation A": TeamForm("Nation A", 6, 5, 0, 1, 12, 4),
        "Nation B": TeamForm("Nation B", 6, 4, 1, 1, 11, 5),
        "Mexico": TeamForm("Mexico", 6, 3, 2, 1, 9, 5),
        "Group A Opponent": TeamForm("Group A Opponent", 6, 2, 2, 2, 7, 7),
        "Curacao": TeamForm("Curacao", 6, 3, 1, 2, 8, 8),
        "Germany": TeamForm("Germany", 6, 4, 1, 1, 14, 6),
        "Finalist A": TeamForm("Finalist A", 6, 4, 1, 1, 11, 5),
        "Finalist B": TeamForm("Finalist B", 6, 4, 1, 1, 10, 6),
    }
    injuries = {
        "European Club A": InjuryReport(
            "European Club A",
            unavailable=("First-choice fullback",),
            doubtful=(),
            source="fixture",
        ),
        "European Club B": InjuryReport(
            "European Club B",
            unavailable=("Rotation midfielder",),
            doubtful=("Starting winger",),
            source="fixture",
        ),
        "Nation A": InjuryReport("Nation A", unavailable=(), doubtful=(), source="fixture"),
        "Nation B": InjuryReport(
            "Nation B",
            unavailable=("Starting center back",),
            doubtful=(),
            source="fixture",
        ),
        "Mexico": InjuryReport(
            "Mexico",
            unavailable=(),
            doubtful=("Veteran forward",),
            source="fixture",
        ),
        "Group A Opponent": InjuryReport("Group A Opponent", source="fixture"),
        "Curacao": InjuryReport(
            "Curacao",
            unavailable=("Defensive midfielder",),
            source="fixture",
        ),
        "Germany": InjuryReport("Germany", unavailable=(), doubtful=(), source="fixture"),
        "Finalist A": InjuryReport("Finalist A", unavailable=(), doubtful=(), source="fixture"),
        "Finalist B": InjuryReport("Finalist B", unavailable=(), doubtful=(), source="fixture"),
    }
    head_to_heads = {
        ("European Club A", "European Club B"): HeadToHeadRecord(
            home_team_wins=2,
            draws=1,
            away_team_wins=1,
            meetings=4,
            summary="European Club A leads recent meetings 2-1-1",
        ),
        ("Nation A", "Nation B"): HeadToHeadRecord(
            home_team_wins=1,
            draws=2,
            away_team_wins=1,
            meetings=4,
            summary="The nations are even across the last four meetings",
        ),
        ("Mexico", "Group A Opponent"): HeadToHeadRecord(
            home_team_wins=0,
            draws=0,
            away_team_wins=0,
            meetings=0,
            summary="No matchup history in the local fixture catalog",
        ),
        ("Curacao", "Germany"): HeadToHeadRecord(
            home_team_wins=0,
            draws=0,
            away_team_wins=1,
            meetings=1,
            summary="Germany leads the limited local history",
        ),
        ("Finalist A", "Finalist B"): HeadToHeadRecord(
            home_team_wins=1,
            draws=1,
            away_team_wins=1,
            meetings=3,
            summary="The finalists are even in the placeholder history",
        ),
    }
    venues = {
        "ucl-final-2026": Venue("Puskas Arena", "Budapest", "Hungary"),
        "world-cup-final-2026": Venue("MetLife Stadium", "East Rutherford", "United States"),
        "wc-2026-match-001": Venue("Estadio Azteca", "Mexico City", "Mexico", "Mexico"),
        "wc-2026-match-010": Venue("Houston Stadium", "Houston", "United States"),
        "wc-2026-final": Venue("MetLife Stadium", "East Rutherford", "United States"),
    }
    weather = {
        "ucl-final-2026": Weather(21.0, 12.0, 0.5, "mild and mostly dry"),
        "world-cup-final-2026": Weather(28.0, 16.0, 2.0, "warm with a small shower risk"),
        "wc-2026-match-001": Weather(22.0, 9.0, 3.0, "mild with possible showers"),
        "wc-2026-match-010": Weather(30.0, 18.0, 1.0, "hot with light wind"),
        "wc-2026-final": Weather(29.0, 14.0, 2.0, "warm with isolated showers possible"),
    }
    odds: dict[str, tuple[OddsQuote, ...]] = {
        "ucl-final-2026": (
            OddsQuote("FixtureBook", home_win=2.25, draw=3.35, away_win=3.15),
            OddsQuote("ExampleOdds", home_win=2.30, draw=3.25, away_win=3.10),
        ),
        "world-cup-final-2026": (
            OddsQuote("FixtureBook", home_win=2.05, draw=3.20, away_win=3.85),
            OddsQuote("ExampleOdds", home_win=2.10, draw=3.10, away_win=3.75),
        ),
        "wc-2026-match-001": (
            OddsQuote("FixtureBook", home_win=1.95, draw=3.35, away_win=4.10),
            OddsQuote("ExampleOdds", home_win=2.00, draw=3.30, away_win=3.95),
        ),
        "wc-2026-match-010": (
            OddsQuote("FixtureBook", home_win=8.00, draw=4.75, away_win=1.38),
            OddsQuote("ExampleOdds", home_win=7.50, draw=4.60, away_win=1.42),
        ),
        "wc-2026-final": (
            OddsQuote("FixtureBook", home_win=2.35, draw=3.05, away_win=3.25),
            OddsQuote("ExampleOdds", home_win=2.40, draw=3.00, away_win=3.20),
        ),
    }
    return FixtureCatalog(
        requests,
        competitions,
        forms,
        injuries,
        head_to_heads,
        venues,
        weather,
        odds,
    )


def build_fixture_agent(
    prediction_log: PredictionLog,
    catalog: FixtureCatalog | None = None,
) -> PredictionAgent:
    tools = FixtureTools(catalog or default_catalog())
    return PredictionAgent(
        form_tool=tools,
        injury_tool=tools,
        head_to_head_tool=tools,
        venue_tool=tools,
        weather_tool=tools,
        odds_tool=tools,
        reasoner=MatchupReasoner(),
        prediction_log=prediction_log,
    )
