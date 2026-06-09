# soccer_agent/tools/odds.py
import httpx
from typing import Optional

from soccer_agent.tools.schemas import OddsSummary


class FetchOddsTool:
    """Tool for fetching betting odds from Odds API."""

    def __init__(self, api_key: str, base_url: str = "https://api.the-odds-api.com/v4"):
        self.api_key = api_key
        self.base_url = base_url

    async def arun(
        self,
        match_id: str,
        our_predicted_implied_prob: Optional[float] = None
    ) -> OddsSummary:
        """Fetch odds for a match."""
        if not self.api_key:
            return OddsSummary(
                match_id=match_id,
                home_win_odds={},
                draw_odds={},
                away_win_odds={},
                implied_prob_home=0.0,
                value_detected=False
            )

        headers = {"X-API-KEY": self.api_key}

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/sports/soccer/odds",
                headers=headers,
                params={"regions": "uk", "markets": "h2h", "oddsFormat": "decimal"}
            )
            response.raise_for_status()
            data = await response.json()

        # Find the match
        match_data = None
        for match in data:
            if str(match.get("id")) == match_id:
                match_data = match
                break

        if not match_data:
            return OddsSummary(
                match_id=match_id,
                home_win_odds={},
                draw_odds={},
                away_win_odds={},
                implied_prob_home=0.0,
                value_detected=False
            )

        home_win_odds = {}
        draw_odds = {}
        away_win_odds = {}
        all_home_odds = []

        for bookmaker in match_data.get("bookmakers", []):
            bookmaker_key = bookmaker.get("key")
            for market in bookmaker.get("markets", []):
                if market.get("key") == "h2h":
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Home":
                            home_win_odds[bookmaker_key] = outcome.get("price")
                            all_home_odds.append(outcome.get("price"))
                        elif outcome.get("name") == "Draw":
                            draw_odds[bookmaker_key] = outcome.get("price")
                        elif outcome.get("name") == "Away":
                            away_win_odds[bookmaker_key] = outcome.get("price")

        # Calculate average implied probability
        if all_home_odds:
            avg_home_odds = sum(all_home_odds) / len(all_home_odds)
            implied_prob_home = 1.0 / avg_home_odds
        else:
            implied_prob_home = 0.0

        # Value detection: if our predicted probability > implied probability by margin
        value_detected = False
        if our_predicted_implied_prob is not None and implied_prob_home > 0:
            # 5% margin for value
            value_detected = our_predicted_implied_prob > implied_prob_home + 0.05

        return OddsSummary(
            match_id=match_id,
            home_win_odds=home_win_odds,
            draw_odds=draw_odds,
            away_win_odds=away_win_odds,
            implied_prob_home=implied_prob_home,
            value_detected=value_detected
        )


def create_odds_tool(api_key: str | None) -> FetchOddsTool:
    """Create odds tool."""
    return FetchOddsTool(api_key or "")