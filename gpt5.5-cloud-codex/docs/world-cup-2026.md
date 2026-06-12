# FIFA 2026 World Cup Modeling

The World Cup 2026 workflow has two phases:

1. Fetch raw API-Football data into a local snapshot directory.
2. Load those snapshots into normalized profiles, rankings, and group-stage score predictions.

This keeps credentials and network access out of the model path. Once the snapshot exists,
predictions are deterministic and can be tested offline.

## Data Fetch

```bash
export API_FOOTBALL_KEY="..."
soccer-forecast fetch-world-cup-data \
  --data-dir data/api-football/world-cup-2026 \
  --world-cup-league-id 1 \
  --world-cup-season 2026 \
  --club-season 2025 \
  --request-delay-seconds 0.5
```

The fetch command is resumable. If the provider returns a rate-limit response, wait for
the limit window to clear and run the same command again; existing JSON files are reused.

To create a human-friendly group-by-group result table:

```bash
soccer-forecast predict-world-cup-group-stage --output markdown
```

The fetcher stores raw JSON files for:

- World Cup fixtures and national teams.
- World Cup standings, used to map each team and match to its group.
- Each national-team squad and coaching staff.
- Each player profile from squad data and club-season player statistics.
- Each coach trophy history.
- Clubs represented by those players, including last-season team statistics.
- Leagues represented by those clubs, including standings and fixture counts.

The default World Cup league id is `1`, which is the common API-Football id for the
FIFA World Cup. Override it if your API account or provider response uses a different id.

## External Factors

Some requested inputs are not reliably available from API-Football for every league or
club. Add `external_factors.json` in the snapshot directory to supply those values:

```json
{
  "country_history": {
    "Brazil": 100,
    "France": 96
  },
  "country_strength": {
    "England": 92,
    "Spain": 91
  },
  "league_average_attendance": {
    "39": 38500,
    "Premier League": 38500
  },
  "club_major_titles": {
    "50": 35,
    "Manchester City": 35
  }
}
```

Keys can be API ids or display names. Missing values fall back to conservative defaults.

## Ranking Inputs

All rankings are clamped to `0-100`.

- League ranking: World Cup player count, average attendance, country strength, league
  team count, and matches played.
- Club ranking: league ranking, World Cup player count, last-season win/loss record,
  and major titles.
- Player ranking: club ranking, league ranking, goals, and average match rating.
- Coach ranking: current national team context, recent team win/loss record, and titles.
- National-team ranking: country history, domestic league and club strength, player
  quality, coach quality, and recent results.

Match score predictions then apply venue adjustments for host-country advantage, travel
proxy effects, and hot-weather city context before converting adjusted team ratings into
expected goals and final scores.
