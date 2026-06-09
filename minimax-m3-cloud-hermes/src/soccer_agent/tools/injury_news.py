"""injury_news tool.

Returns a per-team list of injury reports. The reasoner uses these to
penalize a team's strength — see Phase 1 plan §Tool design.

Live: would scrape news + parse club injury pages. Phase 1: fixtures only.
Fixture: `fixtures/injury/<home>__<away>__<kickoff_date>.json`
"""

from __future__ import annotations

from pydantic import BaseModel

from ..models import InjuryOutput
from . import ToolError
from ._fixtures import load_json


class InjuryInput(BaseModel):
    home_team_id: str
    away_team_id: str
    kickoff_date: str  # ISO date, used as a snapshot key


async def _run_injury_news_live(payload: InjuryInput) -> InjuryOutput:
    raise ToolError(
        source="live",
        message="injury_news live path not implemented in Phase 1",
        retriable=False,
    )


class InjuryNewsTool:
    name = "injury_news"
    description = "Known injuries for both teams near kickoff"
    input_model = InjuryInput
    output_model = InjuryOutput

    async def run(self, payload: InjuryInput) -> InjuryOutput:  # type: ignore[override]
        try:
            return await _run_injury_news_live(payload)
        except ToolError as e:
            if e.source != "live" or e.retriable:
                raise
        data = load_json(
            "injury",
            f"{payload.home_team_id}__{payload.away_team_id}__{payload.kickoff_date}.json",
        )
        if data is None:
            raise ToolError(
                source="fixture",
                message=f"no injury fixture for {payload.home_team_id}__"
                        f"{payload.away_team_id}__{payload.kickoff_date}",
                retriable=False,
            )
        return InjuryOutput.model_validate(data)
