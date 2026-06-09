# soccer_agent/tools/weather.py
import httpx
from typing import Optional

from soccer_agent.tools.schemas import WeatherForecast


class FetchWeatherTool:
    """Tool for fetching weather data from OpenWeatherMap."""

    def __init__(self, api_key: Optional[str], base_url: str = "https://api.openweathermap.org/data/2.5"):
        self.api_key = api_key
        self.base_url = base_url

    async def arun(
        self,
        venue_id: str,
        latitude: float,
        longitude: float,
        match_time_utc: Optional[str] = None
    ) -> WeatherForecast:
        """Fetch weather for a venue."""
        if not self.api_key:
            return WeatherForecast(
                venue_id=venue_id,
                temperature_celsius=0.0,
                condition="unknown",
                wind_speed_kmh=0.0
            )

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/weather",
                params={
                    "lat": latitude,
                    "lon": longitude,
                    "appid": self.api_key,
                    "units": "metric"
                }
            )
            response.raise_for_status()
            data = response.json()

        main = data.get("main", {})
        weather_list = data.get("weather", [])
        wind = data.get("wind", {})

        condition = "unknown"
        if weather_list:
            main_condition = weather_list[0].get("main", "").lower()
            if "clear" in main_condition:
                condition = "clear"
            elif "cloud" in main_condition:
                condition = "cloudy"
            elif "rain" in main_condition or "drizzle" in main_condition:
                condition = "rain"
            elif "snow" in main_condition:
                condition = "snow"

        temperature_celsius = main.get("temp", 0.0)
        wind_speed_ms = wind.get("speed", 0.0)
        wind_speed_kmh = wind_speed_ms * 3.6  # Convert m/s to km/h

        return WeatherForecast(
            venue_id=venue_id,
            temperature_celsius=temperature_celsius,
            condition=condition,
            wind_speed_kmh=wind_speed_kmh
        )


def create_weather_tool(api_key: Optional[str]) -> FetchWeatherTool:
    """Create weather tool."""
    return FetchWeatherTool(api_key)