# soccer_agent/tools/api_football.py
import httpx
from typing import Literal

from soccer_agent.tools.schemas import FormSummary, H2HSummary


class FetchTeamFormTool:
    """Tool for fetching team form from API-Football."""

    def __init__(self, api_key: str, base_url: str = "https://api-football-v1.p.rapidapi.com"):
        self.api_key = api_key
        self.base_url = base_url

    async def arun(
        self,
        team_id: str,
        last_n_matches: int = 5,
        context_mode: Literal["standard", "group_stage", "knockout"] = "standard"
    ) -> FormSummary:
        """Fetch team form data."""
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/fixtures",
                headers=headers,
                params={"team": team_id, "last": last_n_matches, "season": 2024}
            )
            response.raise_for_status()
            data = await response.json()

        matches = data.get("response", [])

        if not matches:
            return FormSummary(
                team_id=team_id,
                last_n_matches=0,
                record={"win": 0, "draw": 0, "loss": 0},
                goals_scored=0,
                goals_conceded=0,
                momentum_score=0.0,
                last_5=[]
            )

        # Process matches
        wins = draws = losses = 0
        goals_scored = goals_conceded = 0
        last_5 = []
        momentum_values = []

        for match in matches[:last_n_matches]:
            teams = match.get("teams", {})
            goals = match.get("goals", {})
            is_home = teams.get("home", {}).get("id") == int(team_id)

            team_goals = goals.get("home", 0) if is_home else goals.get("away", 0)
            opp_goals = goals.get("away", 0) if is_home else goals.get("home", 0)

            goals_scored += team_goals
            goals_conceded += opp_goals

            if team_goals > opp_goals:
                wins += 1
                momentum_values.append(1.0)
            elif team_goals == opp_goals:
                draws += 1
                momentum_values.append(0.0)
            else:
                losses += 1
                momentum_values.append(-1.0)

            last_5.append({
                "outcome": "win" if team_goals > opp_goals else "draw" if team_goals == opp_goals else "loss",
                "score": f"{team_goals}-{opp_goals}",
                "opponent": teams.get("away", {}).get("name") if is_home else teams.get("home", {}).get("name")
            })

        momentum_score = sum(momentum_values) / len(momentum_values) if momentum_values else 0.0

        return FormSummary(
            team_id=team_id,
            last_n_matches=len(matches),
            record={"win": wins, "draw": draws, "loss": losses},
            goals_scored=goals_scored,
            goals_conceded=goals_conceded,
            momentum_score=momentum_score,
            last_5=last_5
        )


class FetchH2HTool:
    """Tool for fetching head-to-head history from API-Football."""

    def __init__(self, api_key: str, base_url: str = "https://api-football-v1.p.rapidapi.com"):
        self.api_key = api_key
        self.base_url = base_url

    async def arun(self, team_a_id: str, team_b_id: str, last_n: int = 10) -> H2HSummary:
        """Fetch head-to-head data between two teams."""
        headers = {
            "x-rapidapi-key": self.api_key,
            "x-rapidapi-host": "api-football-v1.p.rapidapi.com"
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.base_url}/fixtures/headtohead",
                headers=headers,
                params={"h2h": f"{team_a_id}-{team_b_id}", "last": last_n}
            )
            response.raise_for_status()
            data = await response.json()

        matches = data.get("response", [])

        if not matches:
            return H2HSummary(
                team_a_id=team_a_id,
                team_b_id=team_b_id,
                team_a_wins=0,
                draws=0,
                team_b_wins=0,
                recent_meetings=[]
            )

        team_a_wins = draws = team_b_wins = 0
        recent_meetings = []

        for match in matches:
            teams = match.get("teams", {})
            goals = match.get("goals", {})

            home_team_id = teams.get("home", {}).get("id")
            home_goals = goals.get("home", 0)
            away_goals = goals.get("away", 0)

            if home_team_id == int(team_a_id):
                if home_goals > away_goals:
                    team_a_wins += 1
                    winner = team_a_id
                elif home_goals == away_goals:
                    draws += 1
                    winner = "draw"
                else:
                    team_b_wins += 1
                    winner = team_b_id
            else:
                if home_goals > away_goals:
                    team_b_wins += 1
                    winner = team_b_id
                elif home_goals == away_goals:
                    draws += 1
                    winner = "draw"
                else:
                    team_a_wins += 1
                    winner = team_a_id

            recent_meetings.append({
                "date": match.get("fixture", {}).get("date"),
                "score": f"{home_goals}-{away_goals}",
                "winner": winner
            })

        return H2HSummary(
            team_a_id=team_a_id,
            team_b_id=team_b_id,
            team_a_wins=team_a_wins,
            draws=draws,
            team_b_wins=team_b_wins,
            recent_meetings=recent_meetings
        )


def create_api_football_tools(api_key: str) -> list:
    """Create tools for API-Football integration."""
    form_tool = FetchTeamFormTool(api_key)
    h2h_tool = FetchH2HTool(api_key)
    return [form_tool, h2h_tool]