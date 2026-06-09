"""Tests for The Odds API client (Task 26).

The Odds API is a paid real-time bookmaker feed. The client we ship
is async, takes an httpx client in its constructor (so we can inject
a mock transport in tests), and exposes a tiny surface:

    client = TheOddsAPIClient(api_key="...", http=httpx_client)
    events = await client.fetch_event_odds(sport="soccer_epl", event_id="abc")

We do NOT hit the network in this test file. Each test injects a
MockTransport-bound httpx.AsyncClient so the round trip is local and
fast.
"""

from __future__ import annotations

import httpx
import pytest

from soccer_agent.data.odds_api import (
    OddsAPIError,
    OddsAPIRateLimited,
    TheOddsAPIClient,
    devig,
)


# -- fixtures ----------------------------------------------------------------


def _client(handler) -> tuple[TheOddsAPIClient, httpx.AsyncClient]:
    """Build a client backed by an httpx.MockTransport.

    Returns (client, transport_owner) so tests can introspect the
    transport if they need to. The client uses the injected
    AsyncClient as-is; do NOT close it from inside the client.
    """
    transport = httpx.MockTransport(handler)
    http = httpx.AsyncClient(transport=transport, base_url="https://api.test")
    client = TheOddsAPIClient(api_key="test-key-xyz", http=http)
    return client, http


# A realistic single-event response from /v4/sports/<sport>/events/<id>/odds
SINGLE_EVENT_RESPONSE = [
    {
        "id": "abc123",
        "sport_key": "soccer_epl",
        "commence_time": "2026-01-15T20:00:00Z",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.10},
                            {"name": "Draw", "price": 3.40},
                            {"name": "Chelsea", "price": 3.50},
                        ],
                    }
                ],
            },
            {
                "key": "bet365",
                "title": "bet365",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.00},
                            {"name": "Draw", "price": 3.50},
                            {"name": "Chelsea", "price": 3.60},
                        ],
                    }
                ],
            },
        ],
    }
]


# -- tests ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_event_odds_parses_bookmakers():
    """A single-event response should yield one BookmakerOdds per bookmaker."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("x-api-key") == "test-key-xyz", (
            f"api key not sent in x-api-key header: {dict(request.headers)}"
        )
        assert "/events/abc123/odds" in str(request.url)
        return httpx.Response(200, json=SINGLE_EVENT_RESPONSE)

    client, http = _client(handler)
    try:
        rows = await client.fetch_event_odds(sport="soccer_epl", event_id="abc123")
    finally:
        await http.aclose()

    assert len(rows) == 2
    # The Odds API returns a "title" (Pinnacle, bet365) and a "key"
    # (pinnacle, bet365). Our parser prefers title. Match on that.
    pinnacle = next(r for r in rows if r.name == "Pinnacle")
    assert pinnacle.home == 2.10
    assert pinnacle.draw == 3.40
    assert pinnacle.away == 3.50
    bet365 = next(r for r in rows if r.name == "bet365")
    assert bet365.home == 2.00
    assert bet365.draw == 3.50
    assert bet365.away == 3.60


@pytest.mark.asyncio
async def test_fetch_event_odds_returns_empty_on_unknown_event():
    """The Odds API returns 404 for unknown events; we want an empty list."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "not found"})

    client, http = _client(handler)
    try:
        rows = await client.fetch_event_odds(sport="soccer_epl", event_id="missing")
    finally:
        await http.aclose()
    assert rows == []


@pytest.mark.asyncio
async def test_fetch_event_odds_surfaces_429_as_rate_limited():
    """A 429 response should raise a retriable subclass."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"message": "rate limit"}, headers={"x-requests-remaining": "0"})

    client, http = _client(handler)
    try:
        with pytest.raises(OddsAPIRateLimited) as excinfo:
            await client.fetch_event_odds(sport="soccer_epl", event_id="abc")
    finally:
        await http.aclose()
    assert excinfo.value.retriable is True


@pytest.mark.asyncio
async def test_fetch_event_odds_surfaces_5xx_as_error():
    """5xx should raise OddsAPIError, not be silently swallowed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream blew up")

    client, http = _client(handler)
    try:
        with pytest.raises(OddsAPIError) as excinfo:
            await client.fetch_event_odds(sport="soccer_epl", event_id="abc")
    finally:
        await http.aclose()
    assert excinfo.value.retriable is True  # 5xx is retriable
    assert excinfo.value.status == 500


def test_client_requires_api_key():
    """No key should raise a clear ValueError at construction time, not at first request."""
    with pytest.raises(ValueError) as excinfo:
        TheOddsAPIClient(api_key="", http=httpx.AsyncClient())
    assert "api_key" in str(excinfo.value).lower() or "key" in str(excinfo.value).lower()


def test_devig_recovers_known_fair_probs_for_synthetic_book():
    """Devigging Pinnacle 2.10/3.40/3.50 should preserve the ranking and
    sum to 1.0 (multiplicative devig removes the overround).

    The raw inverted probs are 0.4762/0.2941/0.2857 summing to 1.056.
    After multiplicative devig they normalise to 0.4509/0.2785/0.2705.
    The KEY property of multiplicative devig is that the *ratios*
    between the three probs are preserved (0.4762/0.2857 = 0.4509/0.2705).
    That is what guarantees the favourite stays the favourite.
    """
    raw_inv = 1/2.10, 1/3.40, 1/3.50
    probs = devig(2.10, 3.40, 3.50)
    # Sum to 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    # Home should be the modal pick.
    assert max(probs, key=probs.get) == "home"
    # The pairwise ratios are preserved exactly.
    assert abs(probs["home"] / probs["away"] - raw_inv[0] / raw_inv[2]) < 1e-9
    assert abs(probs["home"] / probs["draw"] - raw_inv[0] / raw_inv[1]) < 1e-9
    # Concrete values for stability: home > draw > away.
    assert probs["home"] > probs["draw"] > probs["away"]


def test_devig_handles_close_odds():
    """Devigging 2.0/2.0/2.0 (no favorite) should give ~0.333 each."""
    probs = devig(2.0, 2.0, 2.0)
    assert abs(probs["home"] - 1/3) < 1e-6
    assert abs(probs["draw"] - 1/3) < 1e-6
    assert abs(probs["away"] - 1/3) < 1e-6


def test_devig_handles_very_long_odds():
    """1.05/20.0/40.0: home is a huge favorite. Devig should still give sum=1."""
    probs = devig(1.05, 20.0, 40.0)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
    assert probs["home"] > 0.8
    assert probs["draw"] < 0.15
    assert probs["away"] < 0.05
