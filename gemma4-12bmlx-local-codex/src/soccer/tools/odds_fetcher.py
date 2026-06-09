from typing import Dict

class OddsFetcher:
    def fetch_current_odds(self, match_id: str) -> Dict[str, float]:
        """Fetches current bookmaker odds."""
        # Mock implementation
        return {"home": 2.10, "draw": 3.40, "away": 3.80}
