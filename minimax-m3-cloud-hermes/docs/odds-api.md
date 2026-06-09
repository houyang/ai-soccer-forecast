# The Odds API integration

The agent's `odds_market` tool has two backends:

  - **Fixture** — `fixtures/odds/<home>__<away>__<kickoff_date>.json`.
    Always present in the repo, used for backtests and unit tests.
  - **Live** — calls [The Odds API](https://the-odds-api.com/) v4.
    Used when the env vars below are set.

The tool prefers live when configured. On a non-retriable live
error (config / 4xx), it falls through to the fixture. On a
retriable live error (network / 5xx / 429), it bubbles up so the
agent runner can retry the whole call.

## Env vars

| Var                              | Required | Notes                                                |
| -------------------------------- | -------- | ---------------------------------------------------- |
| `SOCCER_AGENT_ODDS_API_KEY`      | yes      | [REDACTED in this doc; supply your own at runtime]   |
| `SOCCER_AGENT_ODDS_API_SPORT`    | yes      | e.g. `soccer_epl`, `soccer_uefa_champs_league`       |
| `SOCCER_AGENT_ODDS_API_EVENT_ID` | yes      | Discover via `/v4/sports/<sport>/events`             |

## Discover event ids

```bash
curl "https://api.the-odds-api.com/v4/sports/soccer_uefa_champs_league/events?api_key=$SOCCER_AGENT_ODDS_API_KEY" | jq '.[].id'
```

For our two target events, you only need to do this once. Cache
the result (a JSON file, a cron'd env var, or a hard-coded constant
in a small "match registry" — the latter is fine for finals).

## Endpoint contract

We hit:

```
GET /v4/sports/{sport}/events/{event_id}/odds
    ?regions=uk,eu,us
    &markets=h2h
    &oddsFormat=decimal
    &dateFormat=iso
    &api_key=...
```

Headers: `x-api-key: ...` (set in addition to the query param —
Odds API accepts both, but the header survives in httpx logs).

## Response shape

A list of events (usually length 1, since we asked for a specific
event id). Each event has:

```json
{
  "id": "abc123",
  "sport_key": "soccer_uefa_champs_league",
  "home_team": "Manchester City",
  "away_team": "Inter Milan",
  "bookmakers": [
    {
      "key": "pinnacle",
      "title": "Pinnacle",
      "markets": [
        {
          "key": "h2h",
          "outcomes": [
            {"name": "Manchester City", "price": 1.91},
            {"name": "Draw", "price": 3.60},
            {"name": "Inter Milan", "price": 4.20}
          ]
        }
      ]
    }
  ]
}
```

Our parser (`_parse_single_event` in `data/odds_api.py`) maps
`home_team`/`away_team`/Draw into the 1X2 `home`/`away`/`draw`
slots, then emits one `BookmakerOdds` per bookmaker.

## Error model

The Odds API client raises:

  - `OddsAPIRateLimited` — HTTP 429. Always retriable (after a backoff).
  - `OddsAPIError(retriable=True)` — 5xx. Retriable.
  - `OddsAPIError(retriable=False)` — 4xx other than 429. Not retriable;
    usually a config issue (bad key, unknown sport, ...).
  - `OddsAPIConfigError` — raised at client construction when
    `api_key` is empty. Not retriable.

The tool layer maps these onto `ToolError(source="live", retriable=...)`.
The agent runner uses `retriable` to decide whether to fall through
to the fixture or bubble up to its own retry policy.

## Devigging

We multiplicative-devig across all returned bookmakers and average
the resulting fair probs. This is robust to one outlier bookmaker
(e.g. a soft book that overweights the favourite) and produces a
single consensus implied probability vector.

```python
from soccer_agent.data.odds_api import devig
devig(2.10, 3.40, 3.50)
# -> {'home': 0.4509, 'draw': 0.2785, 'away': 0.2706}
```

## Rate limits

The free tier is 500 requests/month. The agent should not poll
the live feed continuously — it should fetch once per match per day
at most, and only for the matches it actually plans to predict. A
good rule: fetch on agent initialisation, cache the result, and
re-fetch on user request (the API also supports cached responses
in its `/scores` endpoint, which returns historical odds at a
fraction of the credit cost).
