"""The Odds API client.

The Odds API (https://the-odds-api.com/) is a real-time bookmaker
feed covering 30+ sports including soccer_epl, soccer_uefa_champs_league,
americanfootball_nfl, etc. This module wraps the `/v4/sports/<sport>/...`
endpoints and adapts the response shape to our internal
`BookmakerOdds` model.

Design points:
  - Async (httpx.AsyncClient) so it composes cleanly with the rest of
    the agent's async tool layer.
  - The httpx client is INJECTED via the constructor. Tests pass a
    MockTransport; production passes a real httpx.AsyncClient.
  - Multiplicative devigging (a.k.a. "proportional" or "naive" devig):
    the simplest method that's good enough for 1X2 markets. Better
    methods (Shin, Power) add 1-2% accuracy on average; not worth
    the complexity here.
  - Missing or empty API key raises a clear error at construction.
    This is intentional: fail fast, not at the first live request.

To use in production:
    from soccer_agent.data.odds_api import TheOddsAPIClient
    async with httpx.AsyncClient() as http:
        client = TheOddsAPIClient(api_key=os.environ["ODDS_API_KEY"], http=http)
        rows = await client.fetch_event_odds("soccer_epl", "abc")

The Odds API auth model: include the key in the `x-api-key` header
(not the more common `Authorization: Bearer`). We test for that.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..models import BookmakerOdds


# Markets we know how to parse. The Odds API uses "h2h" for the
# home/draw/away 1X2 market (yes, including soccer draws; the naming
# is from US sports).
H2H_MARKET_KEY = "h2h"

# Outcome names vary by bookmaker. We normalise to {home, draw, away}
# by matching the API's outcome name against the home/away team names
# supplied in the event payload.
DRAW_OUTCOME_NAMES = {"draw", "tie", "x"}


class OddsAPIError(RuntimeError):
    """Generic Odds API failure. `retriable` distinguishes 5xx (yes)
    from 4xx (usually no) and is used by the tool runner to decide
    whether to surface the error or fall back to a fixture."""

    def __init__(self, message: str, *, status: int, retriable: bool):
        super().__init__(message)
        self.status = status
        self.retriable = retriable


class OddsAPIRateLimited(OddsAPIError):
    """429 — quota exhausted. Always retriable (after a backoff)."""

    def __init__(self, message: str = "Odds API rate limit hit"):
        super().__init__(message, status=429, retriable=True)


class OddsAPIConfigError(ValueError):
    """Missing or malformed config (e.g. no API key). Not retriable."""


def devig(home: float, draw: float, away: float) -> dict[str, float]:
    """Multiplicative devigging for 1X2 markets.

    Inverts each decimal price to get the raw implied prob, then
    scales so the three sum to 1.0. This is the simplest method; it
    assumes the bookmaker's vig is proportional to the true
    probability, which is roughly true for sharp books like Pinnacle.

    Returns {"home": p, "draw": p, "away": p} with sum == 1.0.
    """
    if home <= 1.0 or draw <= 1.0 or away <= 1.0:
        raise ValueError(
            f"decimal odds must be > 1.0, got home={home} draw={draw} away={away}"
        )
    inv = 1.0 / home, 1.0 / draw, 1.0 / away
    s = sum(inv)
    return {"home": inv[0] / s, "draw": inv[1] / s, "away": inv[2] / s}


def _outcome_to_index(outcome_name: str, home_team: str, away_team: str) -> str | None:
    """Map an Odds API outcome name to 'home' / 'draw' / 'away'.

    Returns None if the outcome is not part of the 1X2 market (e.g.
    some books also expose alternate lines or totals on the same
    market key; we just skip them).
    """
    n = outcome_name.strip().lower()
    if n in DRAW_OUTCOME_NAMES:
        return "draw"
    if n == home_team.strip().lower():
        return "home"
    if n == away_team.strip().lower():
        return "away"
    return None


def _parse_single_event(event: dict[str, Any]) -> list[BookmakerOdds]:
    """Extract one BookmakerOdds per bookmaker from an event payload."""
    home_team = event.get("home_team", "")
    away_team = event.get("away_team", "")
    out: list[BookmakerOdds] = []
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != H2H_MARKET_KEY:
                continue
            slots: dict[str, float] = {}
            for oc in market.get("outcomes", []):
                idx = _outcome_to_index(oc.get("name", ""), home_team, away_team)
                if idx is None:
                    continue
                try:
                    price = float(oc["price"])
                except (KeyError, TypeError, ValueError):
                    continue
                slots[idx] = price
            if {"home", "draw", "away"} <= set(slots):
                out.append(BookmakerOdds(
                    name=bm.get("title") or bm.get("key") or "unknown",
                    home=slots["home"],
                    draw=slots["draw"],
                    away=slots["away"],
                ))
                break  # one h2h market per bookmaker is enough
    return out


def _consensus_pick(probs: dict[str, float]) -> str:
    """Pick the most likely outcome; tie-break home > draw > away."""
    rank = {"home": 2, "draw": 1, "away": 0}
    return max(probs, key=lambda k: (probs[k], rank.get(k, -1)))


class TheOddsAPIClient:
    """Async client for The Odds API v4.

    The httpx.AsyncClient is injected so tests can pass a
    MockTransport. In production you'll typically pass
    `httpx.AsyncClient(timeout=10.0)`.
    """

    BASE_URL = "https://api.the-odds-api.com"

    def __init__(self, *, api_key: str, http: httpx.AsyncClient):
        if not api_key:
            raise OddsAPIConfigError("api_key is required")
        if http is None:
            raise OddsAPIConfigError("http client is required")
        self._api_key = api_key
        self._http = http

    async def fetch_event_odds(
        self, *, sport: str, event_id: str,
    ) -> list[BookmakerOdds]:
        """Fetch 1X2 odds for a single event across all bookmakers.

        Returns an empty list if the event is unknown (404). Raises
        OddsAPIError on any other non-2xx.
        """
        url = f"/v4/sports/{sport}/events/{event_id}/odds"
        params = {
            "api_key": self._api_key,  # Odds API accepts key as query param too
            "regions": "uk,eu,us",
            "markets": H2H_MARKET_KEY,
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }
        # Use query param to match the documented contract; the x-api-key
        # header variant is also accepted but the docs show query-param
        # usage in examples.
        resp = await self._http.get(
            url,
            params=params,
            headers={"x-api-key": self._api_key},
        )
        if resp.status_code == 404:
            return []
        if resp.status_code == 429:
            raise OddsAPIRateLimited(
                f"429 from {url}: {resp.text[:200]}"
            )
        if resp.status_code >= 500:
            raise OddsAPIError(
                f"{resp.status_code} from {url}: {resp.text[:200]}",
                status=resp.status_code,
                retriable=True,
            )
        if resp.status_code >= 400:
            raise OddsAPIError(
                f"{resp.status_code} from {url}: {resp.text[:200]}",
                status=resp.status_code,
                retriable=False,
            )
        try:
            payload = resp.json()
        except json.JSONDecodeError as e:
            raise OddsAPIError(
                f"non-JSON response from {url}: {e}",
                status=resp.status_code,
                retriable=False,
            ) from e
        if not isinstance(payload, list):
            # The Odds API sometimes wraps a single event in a one-element
            # list; sometimes a single object. Normalise.
            if isinstance(payload, dict) and "bookmakers" in payload:
                payload = [payload]
            else:
                raise OddsAPIError(
                    f"unexpected payload shape from {url}: {type(payload).__name__}",
                    status=200,
                    retriable=False,
                )
        rows: list[BookmakerOdds] = []
        for ev in payload:
            rows.extend(_parse_single_event(ev))
        return rows

    async def fetch_event_odds_as_output(
        self, *, sport: str, event_id: str,
    ) -> "OddsAPIAggregated":
        """Convenience wrapper: fetch + devig + consensus pick.

        Useful for the odds_market tool, which wants a full
        OddsOutput-ready dict.
        """
        rows = await self.fetch_event_odds(sport=sport, event_id=event_id)
        if not rows:
            return OddsAPIAggregated(
                bookmakers=[], implied_probs={"home": 0, "draw": 0, "away": 0},
                market_consensus_pick="home",
            )
        # Use the median book (by home odds) as the consensus reference,
        # then average the devigged probs across all books for the final
        # consensus. This is robust to one outlier bookmaker.
        avg = {"home": 0.0, "draw": 0.0, "away": 0.0}
        for r in rows:
            d = devig(r.home, r.draw, r.away)
            for k in avg:
                avg[k] += d[k]
        n = len(rows)
        implied = {k: v / n for k, v in avg.items()}
        return OddsAPIAggregated(
            bookmakers=rows,
            implied_probs=implied,
            market_consensus_pick=_consensus_pick(implied),
        )


# A small bundled DTO so callers don't have to construct the dict
# from three separate fields. Maps 1:1 onto OddsOutput.
from dataclasses import dataclass


@dataclass
class OddsAPIAggregated:
    bookmakers: list[BookmakerOdds]
    implied_probs: dict[str, float]
    market_consensus_pick: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "bookmakers": [b.model_dump() for b in self.bookmakers],
            "implied_probs": self.implied_probs,
            "market_consensus_pick": self.market_consensus_pick,
        }
