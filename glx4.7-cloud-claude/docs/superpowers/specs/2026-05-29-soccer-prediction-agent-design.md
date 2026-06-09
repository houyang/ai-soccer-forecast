# Soccer Prediction Agent - Design Specification

**Date:** 2026-05-29
**Status:** Approved
**Version:** 1.0

---

## Overview

An autonomous multi-tool agent that predicts soccer match outcomes by gathering form data, injuries, head-to-head history, weather, venue information, and odds. The agent outputs predictions with rationale and confidence scores, logs them, and autonomously evaluates accuracy via scheduled jobs.

**Scope:** Initial focus on Premier League, UEFA Champions League 2025/26, and FIFA World Cup 2026 Final.

---

## Requirements Summary

| Requirement | Priority | Notes |
|-------------|----------|-------|
| Prediction accuracy > 55% (vs 33% random) | P0 | Baseline for success |
| Support PL, CL 25/26, WC 2026 final | P0 | Tournament-aware context |
| Autonomous evaluation with self-reflection | P0 | Daily cron job |
| Polished React dashboard | P1 | Focus 2 deliverable |
| Production-grade architecture | P0 | Built with LangGraph |

---

## Section 1: Overall Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Scheduler (Cron)                             │
│                        ┌─────────────┐                                │
│                        │ Event-based │   ← Triggers on:               │
│                        │ + Daily     │     • Fixtures announced       │
│                        └─────────────┘     • Match kickoff (t-minus 2h)│
└─────────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
        ┌───────────────────┐           ┌─────────────────────┐
        │  Prediction Agent │           │  Evaluation Agent   │
        │    (LangGraph)    │           │    (LangGraph)      │
        └───────────────────┘           └─────────────────────┘
                    │                             │
    ┌───────────────┼───────────────┐             │
    ▼               ▼               ▼             ▼
┌────────┐   ┌─────────────┐  ┌──────────┐  ┌──────────────┐
│ Match  │   │ Odds/Injury │  │ Context  │  │ Result      │
│ Fetcher│   │ Scraper     │  │ Aware    │  │ Fetcher      │
│        │   │             │  │ Analyzer │  │              │
└────────┘   └─────────────┘  └──────────┘  └──────────────┘
    │               │               │               │
    └───────────────┴───────────────┴───────────────┘
                                    │
                            ┌───────▼────────┐
                            │  PostgreSQL   │
                            │  Schema:       │
                            │  - matches     │
                            │  - predictions │
                            │  - results     │
                            │  - evaluations │
                            └───────────────┘
                                    │
                            ┌───────▼────────┐
                            │ React Dashboard│
                            │  - Competition │
                            │    selector    │
                            │  - Tournament  │
                            │    bracket     │
                            │  - History     │
                            └────────────────┘
```

### Key Design Decisions

1. **Separate LangGraph workflows** for prediction and evaluation - clear separation of concerns
2. **Tournament-aware tools** - behavior adapts based on competition type and stage
3. **PostgreSQL as single source of truth** - audit trail, easy queries, dashboard integration
4. **React + tRPC** - type-safe end-to-end communication

---

## Section 2: Tool Design

### Core Tools

| Tool | Purpose | Input | Output | Data Source |
|------|---------|-------|--------|-------------|
| `FetchUpcomingFixtures` | Get matches to predict | `competition_id`, `days_ahead` | List of `Match` objects | API-Football |
| `FetchTeamForm` | Recent results, goals | `team_id`, `last_n_matches`, `context_mode` | `FormSummary` | API-Football |
| `FetchH2HHistory` | Head-to-head record | `team_a_id`, `team_b_id`, `last_n` | `H2HSummary` | API-Football |
| `FetchInjuries` | Key injuries/suspensions | `team_id` | `InjuryReport` | Scraper (BBC/ESPN) |
| `FetchOdds` | Bookmaker odds comparison | `match_id` | `OddsSummary` | Odds API |
| `FetchWeather` | Match day conditions | `venue_id`, `match_time_utc` | `WeatherForecast` | OpenWeatherMap |
| `FetchVenue` | Pitch size, surface, crowd | `venue_id` | `VenueInfo` | API-Football |
| `FetchResult` | Actual match outcome | `match_id` | `MatchResult` | API-Football |

### Tool Output Schemas

```python
@dataclass
class FormSummary:
    team_id: str
    last_n_matches: int
    record: dict[str, int]  # {"win": 3, "draw": 1, "loss": 1}
    goals_scored: int
    goals_conceded: int
    momentum_score: float  # -1.0 to 1.0, weighted recent
    last_5: list[dict]

@dataclass
class H2HSummary:
    team_a_id: str
    team_b_id: str
    team_a_wins: int
    draws: int
    team_b_wins: int
    recent_meetings: list[dict]

@dataclass
class InjuryReport:
    team_id: str
    key_out: list[dict]  # {player, position, severity, return_date}
    doubtful: list[dict]
    impact_score: float  # 0-1

@dataclass
class OddsSummary:
    match_id: str
    home_win_odds: dict[str, float]
    draw_odds: dict[str, float]
    away_win_odds: dict[str, float]
    implied_prob_home: float
    value_detected: bool

@dataclass
class WeatherForecast:
    venue_id: str
    temperature_celsius: float
    condition: str  # 'clear', 'rain', 'cloudy', etc.
    wind_speed_kmh: float

@dataclass
class VenueInfo:
    id: str
    name: str
    capacity: int
    surface: str  # 'grass', 'hybrid'
    city: str
```

### Tournament-Aware Tool Behavior

The `FetchTeamForm` tool accepts `context_mode`:
- `"standard"`: Last N matches (league default)
- `"group_stage"`: Only group matches (CL)
- `"knockout"`: Last N knockout matches (CL/WC)

`FetchInjuries` prioritizes differently:
- CL final: Key player status is critical
- WC final: Long-term injuries + rest days analysis

---

## Section 3: LangGraph Workflow Design

### Prediction State Schema

```python
@dataclass
class PredictionState:
    match_id: str
    competition_id: str
    stage: str  # "group", "knockout", "final"

    # Tool outputs
    team_a_form: FormSummary | None
    team_b_form: FormSummary | None
    h2h_history: H2HSummary | None
    injuries_a: InjuryReport | None
    injuries_b: InjuryReport | None
    odds: OddsSummary | None
    weather: WeatherForecast | None
    venue: VenueInfo | None

    # Reasoning
    context_analysis: str | None
    synthesized_rationale: str | None

    # Output
    predicted_outcome: str  # "home", "draw", "away"
    confidence_score: float  # 0-100
    timestamp: str
```

### Prediction Workflow Graph

```
                   Start (match_id)
                          │
                    Fetch Match Info
                    (venue, kickoff)
                          │
         ┌────────────────┼────────────────┐
         │                │                │
    Fetch Form A    Fetch Form B    Fetch H2H
         │                │                │
         └────────────────┼────────────────┘
                          │
         ┌────────────────┼────────────────┐
         │                │                │
  Fetch Injury A   Fetch Odds      Fetch Weather
         │                │                │
         └────────────────┼────────────────┘
                          │
                    Analyze Context
                  (tournament aware)
                          │
                 Synthesize & Reason
                    (LLM reasoning)
                          │
                  Calculate Confidence
                    (formula-based)
                          │
                  Log to PostgreSQL
                          │
                         End
```

### Node Definitions

```python
async def fetch_form_a(state: PredictionState) -> PredictionState:
    state.team_a_form = await fetch_team_form_tool(
        team_id=state.match.home_team_id,
        context_mode=state.stage
    )
    return state

async def analyze_context(state: PredictionState) -> PredictionState:
    if state.stage == "knockout":
        state.context_analysis = analyze_knockout_context(state)
    elif state.stage == "final":
        state.context_analysis = analyze_final_context(state)
    return state

async def synthesize_reasoning(state: PredictionState) -> PredictionState:
    prompt = build_reasoning_prompt(state)
    response = await llm.ainvoke(prompt)
    state.predicted_outcome = response.outcome
    state.synthesized_rationale = response.rationale
    return state
```

### Evaluation Workflow

```
        Find Pending Predictions
                  │
            Fetch Results
          (API-Football)
                  │
            Compare & Score
                  │
            Self-Reflect
          (LLM analysis)
                  │
            Update DB
                  │
                 End
```

---

## Section 4: Database Schema (PostgreSQL)

### Core Tables

```sql
-- Competitions
CREATE TABLE competitions (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    type VARCHAR(20) NOT NULL,  -- 'league', 'tournament'
    api_source VARCHAR(50) NOT NULL,
    current_season VARCHAR(10)
);

-- Teams
CREATE TABLE teams (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    logo_url VARCHAR(255),
    api_source VARCHAR(50) NOT NULL
);

-- Venues
CREATE TABLE venues (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    city VARCHAR(100),
    country VARCHAR(100),
    capacity INTEGER,
    surface VARCHAR(50),
    latitude FLOAT,
    longitude FLOAT
);

-- Matches
CREATE TABLE matches (
    id VARCHAR(50) PRIMARY KEY,
    competition_id VARCHAR(50) REFERENCES competitions(id),
    stage VARCHAR(50),
    home_team_id VARCHAR(50) REFERENCES teams(id),
    away_team_id VARCHAR(50) REFERENCES teams(id),
    venue_id VARCHAR(50) REFERENCES venues(id),
    kickoff_utc TIMESTAMP NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    winner VARCHAR(20),
    status VARCHAR(20) DEFAULT 'upcoming',
    temperature_celsius FLOAT,
    weather_condition VARCHAR(50),
    wind_speed_kmh FLOAT,
    INDEX idx_kickoff (kickoff_utc),
    INDEX idx_competition (competition_id)
);

-- Predictions
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    match_id VARCHAR(50) REFERENCES matches(id) UNIQUE,
    predicted_outcome VARCHAR(20) NOT NULL,
    confidence_score FLOAT NOT NULL CHECK (confidence_score BETWEEN 0 AND 100),
    rationale TEXT NOT NULL,
    reasoning_json JSONB NOT NULL,
    timestamp_utc TIMESTAMP NOT NULL DEFAULT NOW(),
    tools_used JSONB NOT NULL,
    model_version VARCHAR(50),
    INDEX idx_match (match_id),
    INDEX idx_timestamp (timestamp_utc)
);

-- Evaluations
CREATE TABLE evaluations (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id) UNIQUE,
    actual_outcome VARCHAR(20) NOT NULL,
    correct BOOLEAN NOT NULL,
    confidence_score FLOAT NOT NULL,
    calibrated_confidence FLOAT,
    self_reflection TEXT,
    reflection_timestamp TIMESTAMP NOT NULL,
    tools_used_correctly JSONB,
    missed_factors JSONB,
    INDEX idx_prediction (prediction_id),
    INDEX idx_correct (correct)
);

-- Metrics (aggregated)
CREATE TABLE metrics (
    id SERIAL PRIMARY KEY,
    competition_id VARCHAR(50),
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    total_predictions INTEGER NOT NULL,
    correct_predictions INTEGER NOT NULL,
    accuracy_rate FLOAT NOT NULL,
    avg_confidence FLOAT,
    avg_confidence_when_correct FLOAT,
    avg_confidence_when_wrong FLOAT,
    home_accuracy FLOAT,
    draw_accuracy FLOAT,
    away_accuracy FLOAT,
    INDEX idx_competition_period (competition_id, period_start, period_end)
);

-- Tool errors
CREATE TABLE tool_errors (
    id SERIAL PRIMARY KEY,
    tool_name VARCHAR(50) NOT NULL,
    match_id VARCHAR(50),
    error_message TEXT NOT NULL,
    timestamp_utc TIMESTAMP NOT NULL DEFAULT NOW(),
    resolved BOOLEAN DEFAULT FALSE
);
```

---

## Section 5: Evaluation Harness Design

### Confidence Calibration

```python
def evaluate_calibration(predictions_window: list[Evaluation]) -> dict:
    buckets = [(0,50), (50,60), (60,70), (70,80), (80,90), (90,100)]
    calibration = {}

    for low, high in buckets:
        bucket_preds = [p for p in predictions_window
                       if low <= p.confidence_score < high]
        actual_accuracy = sum(1 for p in bucket_preds if p.correct) / len(bucket_preds)
        calibration[f"{low}-{high}"] = {
            "expected": (low + high) / 2,
            "actual": actual_accuracy * 100,
            "bias": actual_accuracy * 100 - (low + high) / 2
        }

    return calibration
```

### Self-Reflection Prompt

```
You predicted {outcome} with {confidence}% confidence.
Actual result: {actual_result}.

Context:
- Team A form: {team_a_form}
- Team B form: {team_b_form}
- H2H history: {h2h}
- Key injuries: {injuries}
- Weather: {weather}
- Odds: {odds}

Your reasoning: {original_rationale}

Analyze this prediction:
1. Was your reasoning sound? Why/why not?
2. What factors did you overlook?
3. Which tools were most/least helpful?
4. What would you do differently next time?

Output: JSON with {analysis, key_insights, improvement_suggestion}
```

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Overall Accuracy | % of correct predictions | >55% |
| Calibration Error | |R²| between predicted and actual | <0.1 |
| Home Win Accuracy | Accuracy on home win predictions | >60% |
| High Confidence | Accuracy on 80%+ predictions | >75% |
| Tournament Special | Accuracy on knockout/final matches | >50% |

### Scheduler Triggers

```python
PREDICTION_TRIGGER = "0 */6 * * *"  # Every 6 hours
EVALUATION_JOB = "0 8 * * *"        # Daily at 8 AM
METRICS_UPDATE = "0 9 * * 1"         # Weekly on Monday
```

---

## Section 6: React Dashboard Design

### Layout Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  Soccer Predictor                                      [User ▼] [⚙️] │
├──────────────────────────────────────────────────────────────────────┤
│  [Competitions ▼]  [Time Range ▼]  [Export]                          │
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │   Overall        │  │   This Week      │  │   CL 25/26       │  │
│  │   Accuracy: 58%  │  │   Accuracy: 62%  │  │   Accuracy: 55%  │  │
│  │   +3% vs last    │  │                  │  │                  │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Accuracy Trend                                              │   │
│  │  Line chart showing accuracy over time per competition       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Upcoming Predictions                          [Load More]   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Recent Evaluations                            [View All]    │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| `AccuracyCards` | Small metric cards with sparkline trends |
| `AccuracyChart` | Line chart showing accuracy over time |
| `PredictionTable` | Upcoming predictions with drill-down |
| `EvaluationLog` | Recent predictions with actual results |
| `ConfidenceCalibrationChart` | Scatter plot: predicted vs actual confidence |
| `TournamentBracket` | CL knockout bracket |
| `SettingsPanel` | API keys, schedule, manual trigger |

### Page Routes

| Route | Content |
|-------|---------|
| `/` | Dashboard overview |
| `/predictions` | All predictions with filters |
| `/prediction/:id` | Detailed prediction view |
| `/evaluations` | Evaluation history |
| `/competitions` | Competition management |
| `/settings` | Configuration |

### Technology Stack

- React 18 + TypeScript
- TanStack Query
- Recharts
- Tailwind CSS
- tRPC (end-to-end type safety)

---

## Section 7: Error Handling & Observability

### Error Handling Strategy

| Level | Scenario | Action |
|-------|----------|--------|
| Tool-level | API rate limit, timeout | Retry 3x with backoff, log, continue degraded |
| Node-level | LLM timeout, parsing error | Fail gracefully, flag for review |
| Workflow-level | Multiple tool failures | Skip prediction, notify, retry next cycle |
| System-level | Database connection lost | Queue predictions, alert |

### Observability Stack

- **Metrics (Prometheus):** prediction_count, latency, tool_errors, accuracy_rate
- **Logging:** Structured JSON with workflow context
- **Tracing (OpenTelemetry):** Full workflow trace visualization
- **Alerts:** Accuracy < 40% for 7 days, tool error rate > 10%, no predictions in 24h

### Circuit Breaker

Tools use circuit breaker pattern: after N failures, tool is temporarily disabled (with timeout).

---

## Section 8: Testing Strategy

### Test Pyramid

```
              ┌─────────────────┐
              │   E2E Tests     │  ← Weekly
              │   10%           │
              └─────────────────┘
           ┌─────────────────────────┐
           │    Integration Tests    │  ← CI, mock APIs
           │    30%                  │
           └─────────────────────────┘
        ┌─────────────────────────────────┐
        │         Unit Tests              │  ← CI, fast
        │         60%                     │
        └─────────────────────────────────┘
```

### Test Categories

| Type | What | Tool |
|------|------|------|
| Unit | Individual tool functions, confidence calc | pytest |
| Integration | LangGraph workflows, DB operations | pytest + fixtures |
| E2E | Full prediction cycle, dashboard | Playwright |
| LLM Eval | Reasoning quality, confidence calibration | Custom harness |

---

## Implementation Order

1. Database schema and migrations
2. Tool implementations (with mocks)
3. LangGraph prediction workflow
4. PostgreSQL integration and logging
5. Evaluation workflow
6. Scheduler setup
7. React dashboard (basic)
8. Enhanced dashboard with charts
9. Observability integration

---

## Dependencies

| Category | Library | Purpose |
|----------|---------|---------|
| Orchestration | langgraph | Workflow orchestration |
| LLM | langchain-anthropic | Claude integration |
| Database | sqlalchemy, asyncpg | PostgreSQL |
| API | httpx, aiohttp | HTTP clients |
| Dashboard | react, trpc | Frontend |
| Testing | pytest, playwright | Test suite |
| Observability | prometheus-client, opentelemetry | Metrics/tracing |

---

## Configuration

- **API Keys:** Stored in environment variables (`.env` file with `.env.example` template)
- **Database URL:** Environment variable `DATABASE_URL`
- **LLM API Key:** Environment variable `ANTHROPIC_API_KEY`

---

## Success Criteria

- [ ] Overall prediction accuracy > 55%
- [ ] Support PL, CL 25/26, WC 2026 final
- [ ] Autonomous daily evaluation running
- [ ] Dashboard shows real-time accuracy
- [ ] All tests passing (unit, integration, e2e)
- [ ] Error rate < 5% on tool calls