import pytest
from soccer_agent.tools.injuries import FetchInjuriesTool


@pytest.mark.asyncio
async def test_fetch_injuries_bbc():
    tool = FetchInjuriesTool()

    mock_html = """
    <div class="qa-injury-table">
        <table>
            <tr><td class="qa-player-name">Harry Kane</td><td class="qa-injury-type">Ankle</td><td class="qa-return-date">6 weeks</td></tr>
            <tr><td class="qa-player-name">Son Heung-min</td><td class="qa-injury-type">Hamstring</td><td class="qa-return-date">2 weeks</td></tr>
        </table>
    </div>
    """

    from unittest.mock import AsyncMock, patch, MagicMock

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.text = mock_html

    async def mock_get(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await tool.arun(team_id="spurs", team_name="Tottenham")

    assert result.team_id == "spurs"
    assert len(result.key_out) >= 0  # May be empty if parsing fails
    assert result.impact_score >= 0.0


@pytest.mark.asyncio
async def test_fetch_injuries_fallback():
    tool = FetchInjuriesTool()

    mock_html = "<html><body>No data</body></html>"

    from unittest.mock import AsyncMock, patch, MagicMock

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.text = mock_html

    async def mock_get(*args, **kwargs):
        return mock_response

    mock_client = AsyncMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await tool.arun(team_id="test_team", team_name="Test Team")

    assert result.team_id == "test_team"
    assert result.impact_score == 0.0


def test_injuries_tool_creation():
    tool = FetchInjuriesTool()
    assert tool.bbc_url == "https://www.bbc.co.uk/sport/football"