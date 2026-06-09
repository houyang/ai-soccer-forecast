# src/soccer/tools/weather.py
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from soccer.models import WeatherReport
from soccer.tools.fixtures import FixtureStore


class WeatherProvider(Protocol):
    def get_weather(self, venue_id: str, kickoff: datetime) -> WeatherReport: ...


class FixtureWeatherProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_weather(self, venue_id: str, kickoff: datetime) -> WeatherReport:
        raw = self._store.get("weather", venue_id)
        return WeatherReport(
            venue_id=raw["venue_id"],
            kickoff=kickoff,
            temp_c=raw["temp_c"],
            wind_kph=raw["wind_kph"],
            precip_mm=raw["precip_mm"],
            condition=raw["condition"],
            source="fixture",
        )
