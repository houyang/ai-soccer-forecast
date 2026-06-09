"""form_recent tool.

Recent form (W/D/L string) for both teams over the last N matches.
Live path: would call API-Football `/fixtures` + `/teams/statistics`.
Fixture path: `fixtures/form/<home>__<away>__<season>.json`.

Output schema is the shared `FormOutput` from soccer_agent.models so
the reasoner can consume it without per-tool type juggling.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import FormEntry, FormOutput
from . import ToolError
from ._fixtures import load_json


class FormInput(BaseModel):
    home_team_id: str
    away_team_id: str
    season: str = Field(default="2024-2025", description="e.g. '2024-2025'")
    n_matches: int = Field(default=5, ge=1, le=38)


async def run_form_recent(payload: FormInput) -> FormOutput:
    """Live stub: always falls back to fixture for now (Phase 1).

    Phase 2 will insert a httpx call to API-Football here; the public
    contract (FormOutput) will not change.
    """
    raise ToolError(
        source="live",
        message="form_recent live path not implemented in Phase 1",
        retriable=False,
    )


class FormRecentTool:
    name = "form_recent"
    description = "Last-N-match form (W/D/L string + points) for both teams"
    input_model = FormInput
    output_model = FormOutput

    async def run(self, payload: FormInput) -> FormOutput:  # type: ignore[override]
        """Try live; on any failure return fixture data; on missing fixture, raise."""
        try:
            return await run_form_recent(payload)
        except ToolError as e:
            if e.source != "live" or e.retriable:
                raise
            # non-retriable live-not-implemented → fixture
            pass

        data = load_json(
            "form",
            f"{payload.home_team_id}__{payload.away_team_id}__{payload.season}.json",
        )
        if data is None:
            raise ToolError(
                source="fixture",
                message=f"no form fixture for {payload.home_team_id}__"
                        f"{payload.away_team_id}__{payload.season}",
                retriable=False,
            )
        return FormOutput.model_validate(data)
