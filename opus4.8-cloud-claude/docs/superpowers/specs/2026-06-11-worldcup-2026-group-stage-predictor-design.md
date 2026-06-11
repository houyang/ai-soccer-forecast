# FIFA 2026 World Cup — Group-Stage Scoreline Predictor (Design)

Date: 2026-06-11
Status: Approved

## Goal

Predict the result and final score of every first-round (group-stage) match of the
FIFA 2026 World Cup (72 matches across 12 groups of 4), driven by real data pulled from
the API-Football (API-Sports v3) service and a layered set of 0–100 rankings for leagues,
clubs, players, coaches, and national teams.

## Data source

API-Sports v3 (`https://v3.football.api-sports.io`), header auth `x-apisports-key`. The
World Cup is league `id=1`; the 2026 edition is `season=2026`. Confirmed available:

| Data | Endpoint |
|------|----------|
| 48 national teams | `/teams?league=1&season=2026` |
| Group assignments (A–L) | `/standings?league=1&season=2026` |
| 72 group-stage fixtures | `/fixtures?league=1&season=2026` |
| 26-man squads (id, age, position) | `/players/squads?team=<id>` |
| Coach (age, career history) | `/coachs?team=<id>` |
| Per-player club stats (club, league, apps, rating, goals) | `/players?id=<id>&season=2025` |
| Club details (country, venue) | `/teams?id=<id>` |
| League details (teams, attendance) | `/leagues?id=<id>&season=2025` + `/standings` |

The key is read from the environment (`SOCCER_API_FOOTBALL_KEY`); only the variable name
appears in `.env.example`. Real key values and the `data/` cache are git-ignored.

## Architecture

New self-contained subpackage `src/soccer/worldcup/`. The existing 1X2 match-prediction
agent is untouched; this reuses its conventions (frozen dataclasses, injected I/O behind
`Protocol`, local-cache-first, strict typing, no network in tests).

```
src/soccer/worldcup/
  entities.py     # frozen dataclasses + (de)serialization
  apifootball.py  # ApiFootballClient: GET + paging, injected HTTP transport
  cache.py        # JsonCache: response cache under data dir ("stored locally first")
  ingest.py       # orchestrates layered fetch -> normalized local dataset
  ranking.py      # pure 0-100 ranking functions (league/club/player/coach/team)
  predict.py      # Poisson scoreline model over the 72 group matches
  cli.py          # `soccer wc {fetch,rank,predict}` subcommands
```

### 1. API client + cache

`ApiFootballClient` performs GETs with the key header, follows API-Sports paging, and
raises a specific `ApiFootballError` on HTTP/quota/parse failure. The HTTP transport is an
injected callable `HttpGet = Callable[[str, Mapping[str, str]], tuple[int, str]]`, so tests
supply a fake and never touch the network; the production default wraps stdlib
`urllib.request` (no new dependency).

`JsonCache` stores each endpoint response as a JSON file keyed by endpoint+params under
`<data_dir>/api/`. The client checks the cache first, so the full ingest runs once and
re-runs are free. The cache directory also serves as the offline dataset that `rank` and
`predict` read — they require no network at all.

### 2. Ingestion

`ingest.py` pulls in dependency order and writes one normalized dataset
(`<data_dir>/worldcup-2026.json`):

1. teams (48) + group assignment (standings)
2. fixtures (72 group matches)
3. squads (26 players/team) + coach/team
4. per-player club stats for season 2025 → derive each player's primary club, league,
   goals, rating
5. derive clubs (country, league, last-season W/L from standings) and leagues
   (team count, matches, average attendance)

Network failures on optional per-player detail degrade gracefully (the player keeps its
squad-level fields; ranking falls back to neutral values), mirroring the existing
dossier "missing" pattern.

### 3. Entities

Frozen dataclasses with explicit `to_dict`/`from_dict`:

- `League(id, name, country, n_teams, matches_played, avg_attendance)`
- `Club(id, name, country, league_id, wins, draws, losses, titles)`
- `Player(id, name, age, position, club_id, goals, rating, wc_team_id)`
- `Coach(id, name, age, wins, draws, losses, titles, team_id)`
- `NationalTeam(id, name, group, player_ids, coach_id, is_host, confederation,
  recent_w, recent_d, recent_l)`
- `Group(name, team_ids)`
- `WcMatch(fixture_id, matchday, group, home_id, away_id, kickoff, venue, home_goals,
  away_goals)` (goals present only if already played)
- `WorldCup(teams, players, coaches, clubs, leagues, groups, matches)` — the dataset root.

### 4. Rankings (0–100, deterministic, documented)

Computed in dependency order; each is a pure function of the dataset + lower-tier ranks.

- **League** = weighted blend of WC-player count, average attendance, and the country's
  soccer strength (a static confederation/heritage table), normalized to 0–100.
- **Club** = league rank, number of WC players, last-season win rate, and major titles.
- **Player** = club rank + league rank, goals (position-adjusted), and average rating.
- **Coach** = current team strength, league strength, and career win/loss record.
- **National team** = country soccer history + strength of the domestic leagues/clubs its
  players come from + squad player quality + coach quality + recent results (last year)
  + host/home-continent and travel/weather adjustment.

Each weight is a named module constant with a one-line rationale; all are unit-tested with
hand-checked fixtures.

### 5. Prediction (Poisson scorelines)

For each group match: map the two national-team ratings to expected goals via an
attack/defense split around a tournament baseline (~2.6 goals/match), apply a host/
home-continent boost and a travel+weather penalty derived from confederation distance and
the venue, then build the independent-Poisson scoreline matrix (0–N goals each). Output:

- most-likely exact scoreline,
- expected goals each side,
- P(home win) / P(draw) / P(away win),
- a one-line rationale (rating gap + adjustments).

Predictions are written to `<data_dir>/worldcup-2026-predictions.json` and printed grouped
by group. `predict` reads only the local dataset — fully offline and deterministic.

### 6. CLI

`soccer wc fetch` runs the live ingest (requires `SOCCER_API_FOOTBALL_KEY`). `soccer wc
rank` prints the five ranking tables. `soccer wc predict` prints the 72 predicted
scorelines and writes them to the data dir. Wired as a `wc` subcommand group alongside the
existing `predict/settle/eval/report` commands.

## Testing

Offline and deterministic:

- ranking functions: hand-checked fixtures asserting ordering and boundary behavior;
- Poisson model: known expected-goals inputs assert scoreline matrix sums to 1, symmetry,
  and monotonicity in the rating gap;
- entity (de)serialization round-trips;
- `JsonCache` read/write/hit using `tmp_path`;
- `ApiFootballClient` paging/error handling with an injected fake transport (no network);
- `ingest` against a tiny committed JSON fixture (2 groups) with a fake client.

`make check` (ruff format + lint, mypy strict, pytest+coverage) must stay green. The live
fetch is the only networked path and is never exercised by the test suite.

## Out of scope (for now)

- Knockout-round bracket simulation.
- A `wc settle` accuracy harness comparing predictions to actual results (easy follow-up,
  since results arrive during the tournament).
