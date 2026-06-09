import pytest
from unittest.mock import AsyncMock, patch
from soccer_agent.tools.odds import FetchOddsTool


@pytest.mark.asyncio
async def test_fetch_odds_success():
    tool = FetchOddsTool(api_key="test_key")

    mock_response_data = [
        {
            "id": "match_1",
            "bookmakers": [
                {
                    "key": "bet365",
                    "title": "Bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home", "price": 2.10},
                                {"name": "Draw", "price": 3.40},
                                {"name": "Away", "price": 3.20}
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value=mock_response_data)

    async def mock_get(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await tool.arun(match_id="match_1")

    assert result.match_id == "match_1"
    assert result.home_win_odds["bet365"] == 2.10
    assert result.implied_prob_home == pytest.approx(0.45, abs=0.05)


@pytest.mark.asyncio
async def test_fetch_odds_value_detection():
    tool = FetchOddsTool(api_key="test_key")

    mock_response_data = [
        {
            "id": "match_1",
            "bookmakers": [
                {
                    "key": "bet365",
                    "markets": [
                        {
                            "key": "h2h",
                            "outcomes": [
                                {"name": "Home", "price": 1.80},
                                {"name": "Draw", "price": 3.50},
                                {"name": "Away", "price": 4.50}
                            ]
                        }
                    ]
                }
            ]
        }
    ]

    mock_response = AsyncMock()
    mock_response.json = AsyncMock(return_value=mock_response_data)

    async def mock_get(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await tool.arun(match_id="match_1", our_predicted_implied_prob=0.40)

    assert result.value_detected is False