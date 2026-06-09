"""h2h_history tool.

Returns the historical head-to-head record between two teams (regardless
of venue), useful for rivalry context.

Fixture: `fixtures/h2h/<home>__<away>.json`
"""

from __future__ import annotations

from pydantic import BaseModel

from ..models import H2HOutput
from . import ToolError
from ._fixtures import load_json


class H2HInput(BaseModel):
    home_team_id: str
    away_team_id: str
    n_meetings: int = 10


async def _run_h2h_live(payload: H2HInput) -> H2HOutput:
    raise ToolError(source="live", message="h2h_history live not implemented (Phase 1)", retriable=False)


class H2HHistoryTool:
    name = "h2h_history"
    description = "All-time head-to-head record (W/D/L counts + last meeting)"
    input_model = H2HInput
    output_model = H2HOutput

    async def run(self, payload: H2HInput) -> H2HOutput:  # type: ignore[override]
        try:
            return await _run_h2h_live(payload)
        except ToolError as e:
            if e.source != "live" or e.retriable:
                raise
        data = load_json("h2h", f"{payload.home_team_id}__{payload.away_team_id}.json")
        if data is None:
            raise ToolError(
                source="fixture",
                message=f"no h2h fixture for {payload.home_team_id}__{payload.away_team_id}",
                retriable=False,
            )
        return H2HOutput.model_validate(data)
