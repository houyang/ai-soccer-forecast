import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from soccer_agent.tools.api_football import FetchTeamFormTool


@pytest.mark.asyncio
async def test_fetch_team_form_success():
    tool = FetchTeamFormTool(api_key="test_key")

    mock_response_data = {
        "response": [
            {"teams": {"home": {"id": 1, "name": "Team A"}, "away": {"id": 2, "name": "Team B"}},
             "goals": {"home": 2, "away": 1}},
            {"teams": {"home": {"id": 2, "name": "Team B"}, "away": {"id": 1, "name": "Team A"}},
             "goals": {"home": 0, "away": 2}},
            {"teams": {"home": {"id": 1, "name": "Team A"}, "away": {"id": 3, "name": "Team C"}},
             "goals": {"home": 1, "away": 1}},
        ]
    }

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value=mock_response_data)

    async def mock_get(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await tool.arun(team_id="1", last_n_matches=3, context_mode="standard")

    assert result.team_id == "1"
    assert result.record["win"] == 2
    assert result.goals_scored == 5
    assert result.goals_conceded == 2


@pytest.mark.asyncio
async def test_fetch_team_form_empty_response():
    tool = FetchTeamFormTool(api_key="test_key")

    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value={"response": []})

    async def mock_get(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await tool.arun(team_id="1", last_n_matches=3, context_mode="standard")

    assert result.team_id == "1"
    assert result.record == {"win": 0, "draw": 0, "loss": 0}