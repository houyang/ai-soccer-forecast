from typing import Dict

class WeatherFetcher:
    def fetch_conditions(self, venue: str) -> Dict[str, Any]:
        """Fetches weather conditions at a specific venue."""
        # Mock implementation
        return {"temp": 22, "condition": "Clear"}
