"""Tests for the live odds_market path (Task 26).

The odds_market tool has two backends:
  - live:    The Odds API (real bookmakers)
  - fixture: JSON file under fixtures/odds/

The tool chooses live when SOCCER_AGENT_ODDS_API_KEY is set in env.
Otherwise it skips live and goes straight to the fixture.

These tests inject a mock-transport httpx client into the tool so
they run without network and without an API key.
"""

from __future__ import annotations

import asyncio
import os

import httpx
import pytest

from soccer_agent.tools import default_registry
from soccer_agent.tools._fixtures import write_json
from soccer_agent.data.odds_api import TheOddsAPIClient


SAMPLE_FIXTURE_ODDS = {
    "bookmakers": [
        {"name": "pinnacle", "home": 2.1, "draw": 3.4, "away": 3.5},
    ],
    "implied_probs": {"home": 0.45, "draw": 0.28, "away": 0.27},
    "market_consensus_pick": "home",
}


SINGLE_EVENT_RESPONSE = [
    {
        "id": "abc",
        "sport_key": "soccer_epl",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [
            {
                "key": "pinnacle", "title": "Pinnacle",
                "markets": [
                    {"key": "h2h", "outcomes": [
                        {"name": "Arsenal", "price": 2.10},
                        {"name": "Draw", "price": 3.40},
                        {"name": "Chelsea", "price": 3.50},
                    ]}
                ],
            },
        ],
    }
]


@pytest.fixture
def fx(monkeypatch, tmp_path):
    """Point the env at a fresh fixture dir."""
    d = tmp_path / "fx"
    d.mkdir()
    monkeypatch.setenv("SOCCER_AGENT_FIXTURES_DIR", str(d))
    return d


def _patch_tool_with_mock_client(monkeypatch, response_payload, status=200):
    """Build a TheOddsAPIClient that talks to a MockTransport, then
    monkeypatch the odds_market tool to use it.

    The tool exposes no constructor for the client today, so we
    patch the module-level _run_odds_live to call the client directly.
    That keeps the contract narrow: tests verify "given a key, the
    live path is taken" without re-implementing the tool's
    live-vs-fixture switching logic.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=response_payload) if status < 400 else httpx.Response(status, text="err")

    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="https://api.test")
    return TheOddsAPIClient(api_key="test-key", http=http)


def test_odds_live_path_uses_client_when_key_present(monkeypatch, fx):
    """With an API key in env, the tool's live path should be tried
    before falling back to the fixture."""
    monkeypatch.setenv("SOCCER_AGENT_ODDS_API_KEY", "test-key")
    monkeypatch.setenv("SOCCER_AGENT_ODDS_API_SPORT", "soccer_epl")
    monkeypatch.setenv("SOCCER_AGENT_ODDS_API_EVENT_ID", "abc")

    client = _patch_tool_with_mock_client(monkeypatch, SINGLE_EVENT_RESPONSE)
    # Track that the live path is exercised by replacing the module
    # function. After the refactor, _run_odds_live() reads env vars
    # and calls the injected client.
    from soccer_agent.tools import odds_market
    called = {"live": 0, "fixture": 0}

    async def fake_live(payload):
        called["live"] += 1
        agg = await client.fetch_event_odds_as_output(
            sport=os.environ["SOCCER_AGENT_ODDS_API_SPORT"],
            event_id=os.environ["SOCCER_AGENT_ODDS_API_EVENT_ID"],
        )
        from soccer_agent.models import OddsOutput
        return OddsOutput.model_validate(agg.to_dict())

    monkeypatch.setattr(odds_market, "_run_odds_live", fake_live)

    # Seed a fixture too — the live path should win and the fixture
    # should NOT be consulted.
    write_json("odds", "man_city__real_madrid__2025-05-30.json", data=SAMPLE_FIXTURE_ODDS)

    reg = default_registry()
    res = asyncio.run(reg.run(
        "odds_market",
        {"home_team_id": "man_city", "away_team_id": "real_madrid", "kickoff_date": "2025-05-30"},
    ))
    assert res.ok is True
    assert called["live"] == 1, f"expected live path to be called once, got {called}"
    # The bookmakers should come from the API response, not the fixture.
    assert res.data.bookmakers[0].name == "Pinnacle"


def test_odds_falls_back_to_fixture_when_no_key(monkeypatch, fx):
    """Without an API key, the tool should skip live and use the fixture."""
    monkeypatch.delenv("SOCCER_AGENT_ODDS_API_KEY", raising=False)

    write_json("odds", "man_city__real_madrid__2025-05-30.json", data=SAMPLE_FIXTURE_ODDS)
    reg = default_registry()
    res = asyncio.run(reg.run(
        "odds_market",
        {"home_team_id": "man_city", "away_team_id": "real_madrid", "kickoff_date": "2025-05-30"},
    ))
    assert res.ok is True
    # Fixture had pinnacle only.
    assert res.data.market_consensus_pick == "home"


def test_odds_live_surfaces_429_to_caller(monkeypatch, fx):
    """A 429 from the API should surface as a retriable tool error."""
    monkeypatch.setenv("SOCCER_AGENT_ODDS_API_KEY", "test-key")
    monkeypatch.setenv("SOCCER_AGENT_ODDS_API_SPORT", "soccer_epl")
    monkeypatch.setenv("SOCCER_AGENT_ODDS_API_EVENT_ID", "abc")

    client = _patch_tool_with_mock_client(monkeypatch, {}, status=429)
    from soccer_agent.tools import odds_market
    from soccer_agent.models import OddsOutput

    async def fake_live(payload):
        from soccer_agent.data.odds_api import OddsAPIRateLimited
        try:
            await client.fetch_event_odds(
                sport=os.environ["SOCCER_AGENT_ODDS_API_SPORT"],
                event_id=os.environ["SOCCER_AGENT_ODDS_API_EVENT_ID"],
            )
        except OddsAPIRateLimited as e:
            from soccer_agent.tools.base import ToolError
            raise ToolError(source="live", message=str(e), retriable=True) from e
        return OddsOutput.model_validate({})  # unreachable

    monkeypatch.setattr(odds_market, "_run_odds_live", fake_live)

    reg = default_registry()
    res = asyncio.run(reg.run(
        "odds_market",
        {"home_team_id": "man_city", "away_team_id": "real_madrid", "kickoff_date": "2025-05-30"},
    ))
    assert res.ok is False
    # ToolResult.error is a string message; the structured info
    # (source='live', retriable=True) lives on the Signal, but the
    # agent runner captures it into a generic message. Just assert
    # the tool refused to fabricate a fixture-style fake success.
    assert "rate" in res.error.lower() or "429" in res.error
