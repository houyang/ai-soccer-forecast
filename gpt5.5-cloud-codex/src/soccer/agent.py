"""Agent orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from soccer.models import MatchEvidence, MatchRequest, MatchResult, PredictionRecord
from soccer.reasoning import MatchupReasoner
from soccer.storage import PredictionLog
from soccer.tools import FormTool, HeadToHeadTool, InjuryNewsTool, OddsTool, VenueTool, WeatherTool


@dataclass(frozen=True)
class PredictionAgent:
    """Coordinate tools, reasoning, logging, and result updates."""

    form_tool: FormTool
    injury_tool: InjuryNewsTool
    head_to_head_tool: HeadToHeadTool
    venue_tool: VenueTool
    weather_tool: WeatherTool
    odds_tool: OddsTool
    reasoner: MatchupReasoner
    prediction_log: PredictionLog

    def predict(self, request: MatchRequest) -> PredictionRecord:
        venue = self.venue_tool.venue_for(request)
        evidence = MatchEvidence(
            request=request,
            home_form=self.form_tool.recent_form(request.home_team, request.competition),
            away_form=self.form_tool.recent_form(request.away_team, request.competition),
            home_injuries=self.injury_tool.injury_report(request.home_team, request),
            away_injuries=self.injury_tool.injury_report(request.away_team, request),
            head_to_head=self.head_to_head_tool.head_to_head(
                request.home_team,
                request.away_team,
            ),
            venue=venue,
            weather=self.weather_tool.forecast(venue, request),
            odds=self.odds_tool.odds_for(request),
        )
        record = PredictionRecord(
            request=request,
            evidence=evidence,
            prediction=self.reasoner.predict(evidence),
        )
        self.prediction_log.append(record)
        return record

    def predict_many(self, requests: Iterable[MatchRequest]) -> list[PredictionRecord]:
        return [self.predict(request) for request in requests]

    def record_result(self, result: MatchResult) -> PredictionRecord:
        return self.prediction_log.attach_result(result)

    def record_results(self, results: Iterable[MatchResult]) -> list[PredictionRecord]:
        return [self.record_result(result) for result in results]
