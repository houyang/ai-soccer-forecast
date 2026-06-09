"""odds_market tool.

Returns 1X2 odds across multiple bookmakers, plus the devigged
implied probabilities and the consensus pick.

Backends (in priority order):
  1. **Live**: The Odds API. Used when SOCCER_AGENT_ODDS_API_KEY is
     set in the environment. The sport + event_id are read from
     SOCCER_AGENT_ODDS_API_SPORT and SOCCER_AGENT_ODDS_API_EVENT_ID
     respectively. Mapping real fixtures to Odds API event_ids is
     outside the scope of this tool; callers (the agent runner or
     a separate "match_id resolver" tool) handle that. For our two
     target events (UCL 25/26 final, WC 26 final), the caller can
     hard-code the IDs or look them up once and cache.
  2. **Fixture**: JSON file under fixtures/odds/. Always used as a
     fallback when live is not configured, and after a retriable
     live error (so the user gets a stale-but-present answer
     instead of a 503).

Fixture: `fixtures/odds/<home>__<away>__<kickoff_date>.json`
"""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel

from ..data.odds_api import TheOddsAPIClient
from ..models import OddsOutput
from . import ToolError
from ._fixtures import load_json


class OddsInput(BaseModel):
    home_team_id: str
    away_team_id: str
    kickoff_date: str


async def _run_odds_live(payload: OddsInput) -> OddsOutput:
    """Live backend: call The Odds API.

    Reads:
      SOCCER_AGENT_ODDS_API_KEY       required to even attempt
      SOCCER_AGENT_ODDS_API_SPORT     e.g. "soccer_uefa_champs_league"
      SOCCER_AGENT_ODDS_API_EVENT_ID  the event id within that sport

    Raises ToolError(source="live", ...) on any API failure. The
    tool's run() method decides whether to fall through to the
    fixture based on the retriable flag.
    """
    api_key = os.environ.get("SOCCER_AGENT_ODDS_API_KEY", "").strip()
    if not api_key:
        raise ToolError(
            source="live",
            message="SOCCER_AGENT_ODDS_API_KEY not set",
            retriable=False,
        )
    sport = os.environ.get("SOCCER_AGENT_ODDS_API_SPORT", "").strip()
    event_id = os.environ.get("SOCCER_AGENT_ODDS_API_EVENT_ID", "").strip()
    if not sport or not event_id:
        raise ToolError(
            source="live",
            message="SOCCER_AGENT_ODDS_API_SPORT and SOCCER_AGENT_ODDS_API_EVENT_ID must be set",
            retriable=False,
        )
    async with httpx.AsyncClient(timeout=10.0) as http:
        client = TheOddsAPIClient(api_key=api_key, http=http)
        try:
            agg = await client.fetch_event_odds_as_output(
                sport=sport, event_id=event_id,
            )
        except Exception as e:
            # The Odds API client raises OddsAPIError / OddsAPIRateLimited;
            # httpx may raise connection errors. Convert all to ToolError.
            # Anything from the network or a 5xx is retriable.
            from ..data.odds_api import OddsAPIError
            if isinstance(e, OddsAPIError):
                raise ToolError(
                    source="live",
                    message=str(e),
                    retriable=e.retriable,
                ) from e
            # httpx transport / connection / timeout: retriable.
            raise ToolError(
                source="live",
                message=f"odds api transport error: {e!r}",
                retriable=True,
            ) from e
    return OddsOutput.model_validate(agg.to_dict())


class OddsMarketTool:
    name = "odds_market"
    description = "Bookmaker 1X2 odds with devigged implied probabilities"
    input_model = OddsInput
    output_model = OddsOutput

    async def run(self, payload: OddsInput) -> OddsOutput:  # type: ignore[override]
        # Always try live first (no-op if no key, raises immediately).
        # Fall through to fixture only when the live error is
        # non-retriable (config / 4xx) — a retriable live error
        # (network blip, 5xx, 429) bubbles up so the agent runner
        # can retry the whole call.
        try:
            return await _run_odds_live(payload)
        except ToolError as e:
            if e.retriable:
                raise
        data = load_json(
            "odds",
            f"{payload.home_team_id}__{payload.away_team_id}__{payload.kickoff_date}.json",
        )
        if data is None:
            raise ToolError(
                source="fixture",
                message=f"no odds fixture for {payload.home_team_id}__"
                        f"{payload.away_team_id}__{payload.kickoff_date}",
                retriable=False,
            )
        return OddsOutput.model_validate(data)
