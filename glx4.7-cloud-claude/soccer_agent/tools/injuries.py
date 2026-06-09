# soccer_agent/tools/injuries.py
import httpx
from bs4 import BeautifulSoup
from typing import Optional

from soccer_agent.tools.schemas import InjuryReport


class FetchInjuriesTool:
    """Tool for fetching injury information via web scraping."""

    def __init__(self):
        self.bbc_url = "https://www.bbc.co.uk/sport/football"
        self.espn_url = "https://www.espn.com/soccer/team/_/id/"

    async def arun(
        self,
        team_id: str,
        team_name: Optional[str] = None
    ) -> InjuryReport:
        """Fetch injury report for a team."""
        key_out = []
        doubtful = []
        total_impact = 0.0

        # Try BBC first
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.bbc_url}/injuries")
                response.raise_for_status()
                html = response.text

            soup = BeautifulSoup(html, "html.parser")

            # Look for injury tables
            tables = soup.find_all("table", class_=lambda x: x and "injury" in x.lower())

            for table in tables:
                rows = table.find_all("tr")
                for row in rows[1:]:  # Skip header
                    cols = row.find_all("td")
                    if len(cols) >= 2:
                        player_name = cols[0].get_text(strip=True)
                        injury_type = cols[1].get_text(strip=True)
                        return_date = cols[2].get_text(strip=True) if len(cols) > 2 else "Unknown"

                        # Determine severity based on keywords
                        severity = "medium"
                        if any(word in injury_type.lower() for word in ["broken", "fracture", "rupture", "torn"]):
                            severity = "high"
                            total_impact += 1.0
                        elif any(word in injury_type.lower() for word in ["knock", "minor", "questionable"]):
                            severity = "low"
                            total_impact += 0.3
                        else:
                            total_impact += 0.5

                        # Determine if doubtful vs definitely out
                        if any(word in return_date.lower() for word in ["doubtful", "question", "50/50"]):
                            doubtful.append({
                                "player": player_name,
                                "position": "unknown",
                                "severity": severity,
                                "return_date": return_date
                            })
                        else:
                            key_out.append({
                                "player": player_name,
                                "position": "unknown",
                                "severity": severity,
                                "return_date": return_date
                            })

        except (httpx.HTTPError, Exception):
            # Log error but don't fail
            pass

        # Cap impact score at 1.0
        impact_score = min(total_impact / 5.0, 1.0)  # Assuming ~5 key players max impact

        return InjuryReport(
            team_id=team_id,
            key_out=key_out,
            doubtful=doubtful,
            impact_score=impact_score
        )


def create_injuries_tool() -> FetchInjuriesTool:
    """Create injuries tool."""
    return FetchInjuriesTool()