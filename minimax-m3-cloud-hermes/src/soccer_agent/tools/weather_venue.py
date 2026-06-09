"""weather_venue tool.

Returns the forecast (or dome-no-op) for the venue on the kickoff date.
Combines venue info (dome / altitude) with a weather lookup.

Fixture: `fixtures/weather/<venue_id>__<date>.json`
"""

from __future__ import annotations

from pydantic import BaseModel

from ..models import WeatherOutput
from . import ToolError
from ._fixtures import load_json


class WeatherInput(BaseModel):
    venue_id: str
    date: str  # ISO date
    is_dome: bool = False


async def _run_weather_live(payload: WeatherInput) -> WeatherOutput:
    raise ToolError(source="live", message="weather_venue live not implemented (Phase 1)", retriable=False)


class WeatherVenueTool:
    name = "weather_venue"
    description = "Weather forecast for the venue on the kickoff date"
    input_model = WeatherInput
    output_model = WeatherOutput

    async def run(self, payload: WeatherInput) -> WeatherOutput:  # type: ignore[override]
        try:
            return await _run_weather_live(payload)
        except ToolError as e:
            if e.source != "live" or e.retriable:
                raise
        data = load_json("weather", f"{payload.venue_id}__{payload.date}.json")
        if data is None:
            raise ToolError(
                source="fixture",
                message=f"no weather fixture for {payload.venue_id}__{payload.date}",
                retriable=False,
            )
        return WeatherOutput.model_validate(data)
