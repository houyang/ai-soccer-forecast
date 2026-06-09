# src/soccer/registry.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from soccer.tools.base import Tool
from soccer.tools.fixtures import FixtureStore
from soccer.tools.form import FixtureFormProvider, FormProvider
from soccer.tools.head_to_head import FixtureH2HProvider, H2HProvider
from soccer.tools.injuries import FixtureInjuryProvider, InjuryProvider
from soccer.tools.odds import FixtureOddsProvider, OddsProvider
from soccer.tools.results import FixtureResultProvider, ResultProvider
from soccer.tools.venue import FixtureVenueProvider, VenueProvider
from soccer.tools.weather import FixtureWeatherProvider, WeatherProvider


@dataclass(frozen=True)
class ToolRegistry:
    form: FormProvider
    injuries: InjuryProvider
    h2h: H2HProvider
    weather: WeatherProvider
    venue: VenueProvider
    odds: OddsProvider
    results: ResultProvider

    def as_tools(self) -> list[Tool]:
        return [
            Tool("form", "recent team form", self.form.get_form),
            Tool("injuries", "injury/availability report", self.injuries.get_injuries),
            Tool("h2h", "head-to-head history", self.h2h.get_h2h),
            Tool("weather", "match-time weather", self.weather.get_weather),
            Tool("venue", "venue characteristics", self.venue.get_venue),
            Tool("odds", "bookmaker odds", self.odds.get_odds),
            Tool("results", "final result lookup", self.results.get_result),
        ]


def build_fixture_registry(fixture_path: Path) -> ToolRegistry:
    store = FixtureStore(fixture_path)
    return ToolRegistry(
        form=FixtureFormProvider(store),
        injuries=FixtureInjuryProvider(store),
        h2h=FixtureH2HProvider(store),
        weather=FixtureWeatherProvider(store),
        venue=FixtureVenueProvider(store),
        odds=FixtureOddsProvider(store),
        results=FixtureResultProvider(store),
    )
