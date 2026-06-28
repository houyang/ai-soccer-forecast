# FIFA 2026 World Cup Modeling

The World Cup 2026 workflow has two phases:

1. Fetch raw API-Football data into a local snapshot directory.
2. Load those snapshots into normalized profiles, rankings, and score predictions.

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

During the tournament, refresh match results and tactical inputs separately:

```bash
soccer-forecast fetch-world-cup-match-updates \
  --data-dir data/api-football/world-cup-2026 \
  --completed-round-limit 1 \
  --request-delay-seconds 0.5
```

This overwrites the mutable fixture and standings snapshots, then stores raw lineup,
event, and match-statistics payloads for completed group-stage fixtures through the
selected round. Use the same round limit when predicting if you want a reproducible
"after round N" model run:

```bash
soccer-forecast predict-world-cup-group-stage \
  --completed-round-limit 1 \
  --remaining-only \
  --output markdown
```

After all group fixtures are complete, omit `--completed-round-limit` so the snapshot
contains every completed tournament fixture and tactical payload available from the
provider:

```bash
soccer-forecast fetch-world-cup-match-updates \
  --data-dir data/api-football/world-cup-2026 \
  --request-delay-seconds 0.5
```

Then forecast the known knockout fixtures:

```bash
soccer-forecast predict-world-cup-elimination-stage \
  --data-dir data/api-football/world-cup-2026 \
  --project-bracket \
  --output markdown
```

Elimination predictions use the same roster, coach, club, and league base ratings as the
group-stage model, then increase the weight of completed tournament form, group-stage
lineup quality, substitution quality, defensive record, rest days, host/travel context,
and knockout-experience history. Because knockout matches cannot end without a team
advancing, each prediction includes an advance pick, decision method, and home-team
advancement probability in JSON output. Without `--project-bracket`, the command outputs
only provider-confirmed knockout fixtures. With `--project-bracket`, it maps those
fixtures to official match numbers 73-88 and advances predicted winners through the
Round of 16, quarterfinals, semifinals, third-place match, and final.

To create a human-friendly group-by-group result table:

```bash
soccer-forecast predict-world-cup-group-stage --output markdown
```

To create a single-match PDF preview and structured JSON result before kickoff:

```bash
soccer-forecast predict-world-cup-match-preview 1489416 \
  --data-dir data/api-football/world-cup-2026 \
  --output predictions/wc-2026-1489416-preview.pdf \
  --json-output predictions/wc-2026-1489416-preview.json \
  --request-delay-seconds 0.5
```

The preview command refreshes the mutable fixture and standings snapshots, fetches
tactical snapshots for completed fixtures before the target kickoff, and fetches the
target fixture's lineup/event/statistics payloads. If the provider has announced the
starting XIs, the report uses those coaches, formations, starters, and substitutes. If
not, it falls back to the latest tournament lineup for each team, then to squad ranking
projections. Use `--json-output` when a machine-readable copy of the same preview and
prediction is needed.

The fetcher stores raw JSON files for:

- World Cup fixtures and national teams.
- World Cup standings, used to map each team and match to its group.
- Each national-team squad and coaching staff.
- Each player profile from squad data and club-season player statistics.
- Each coach trophy history.
- Clubs represented by those players, including last-season team statistics.
- Leagues represented by those clubs, including standings and fixture counts.
- Completed World Cup results plus tactical match snapshots when
  `fetch-world-cup-match-updates` is run.
- Target-match tactical snapshots when `predict-world-cup-match-preview` refreshes a
  single-match report.

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
proxy effects, and hot-weather city context. When match updates are available, the model
also applies bounded tournament adjustments from completed-match points, goal difference,
the most-used formation, selected starting XI quality, and substitution participants
before converting adjusted team ratings into expected goals and final scores. Knockout
forecasts use a lower-scoring elimination goal model and force an advancing side through
regular-time, extra-time, or penalty decision logic.
Single-match PDF/JSON previews use the same ratings, then add coach, formation, starter,
and possible-substitute sections for both teams.
