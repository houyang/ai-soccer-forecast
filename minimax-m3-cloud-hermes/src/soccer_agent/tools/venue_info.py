"""venue_info tool.

Returns venue metadata (capacity, surface, neutral, dome, lat/lon).
This is the only tool that is *always* a fixture in Phase 1 — venues
are static facts and the live "stadium lookup" path adds no value.

Fixture: `fixtures/venues/venue_<id>.json`
"""

from __future__ import annotations

from pydantic import BaseModel

from ..models import Venue
from . import ToolError
from ._fixtures import load_json


class VenueInput(BaseModel):
    venue_id: str


async def _run_venue_live(payload: VenueInput) -> Venue:
    raise ToolError(source="live", message="venue_info live not used (Phase 1)", retriable=False)


class VenueInfoTool:
    name = "venue_info"
    description = "Venue metadata (capacity, surface, neutral, dome, geo)"
    input_model = VenueInput
    output_model = Venue

    async def run(self, payload: VenueInput) -> Venue:  # type: ignore[override]
        try:
            return await _run_venue_live(payload)
        except ToolError as e:
            if e.source != "live" or e.retriable:
                raise
        data = load_json("venues", f"venue_{payload.venue_id}.json")
        if data is None:
            raise ToolError(
                source="fixture",
                message=f"no venue fixture for {payload.venue_id}",
                retriable=False,
            )
        return Venue.model_validate(data)
