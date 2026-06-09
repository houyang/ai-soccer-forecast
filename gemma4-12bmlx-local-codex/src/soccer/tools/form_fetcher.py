from typing import List
from ..models.prediction import FormResult

class FormFetcher:
    def fetch_recent_form(self, team_name: str) -> List[FormResult]:
        """
        Fetches recent form for a given team.
        In production, this calls the Sportmonk/Opta API via BaseConnector.
        """
        # Mock implementation (retained for now until real endpoints are provided)
        return [
            FormResult("2026-05-01", "3-1", "Team B", "W"),
            FormResult("2026-04-28", "0-0", "Team C", "D")
        ]
