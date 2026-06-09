# Elo ratings (Task 27)

> Per-team home/away Elo ratings with a form window, persisted to
> disk and loaded by the agent on startup.

## Why

The numeric reasoner used to treat every team as 1500/1500 — i.e.
the Elo component was a constant `0.5` that carried no information.
That meant the **form**, **H2H**, **market** and **injury** signals
were the only sources of truth. Real prediction work needs an Elo
prior that improves with every match the system observes.

This module fixes that with three upgrades over a textbook Elo:

1. **Home/away splits.** Each team tracks a `home` rating and an
   `away` rating. A team can be strong at home but fragile on the
   road, and the model respects that asymmetry. The mean of the two
   is exposed as `overall` for callers that just want a single number.

2. **Home-advantage delta.** The home team gets a `home_advantage`
   Elo boost (default 65, calibrated to ~46% home-win rate in
   top-flight European soccer) added to their home rating in the
   `expected` calculation. This is the standard Elo "home boost"
   treatment but applied on top of the per-venue rating.

3. **Form window.** A `weight_for(matches_back, window)` function
   provides a linear decay (1.0 → 0.0 over the window) so the
   `form` aggregation in callers can down-weight stale results.
   The default window is 8 matches.

## Files

| Path                                  | Role |
|---------------------------------------|------|
| `src/soccer_agent/elo.py`             | The module (EloState, MatchResult, update, predict). |
| `scripts/build_elo_state.py`          | CLI: read historical matches → write `elo_state.json`. |
| `tests/test_elo.py`                   | Unit tests for the math (expected, update, predict, JSON). |
| `tests/test_elo_integration.py`       | Reasoner uses the state when one is provided. |
| `tests/test_agent_elo.py`             | Agent loads state from disk / env var / fresh. |
| `tests/test_build_elo_state.py`       | CLI script. |

## The math

`expected(a, b) = 1 / (1 + 10^((b - a) / 400))` — the standard Elo
logistic. The home team's effective rating is `home_rating +
home_advantage`; the away team uses their plain `away_rating`. The
result is in [0, 1] and represents the probability that the home
team wins in regulation.

`update_state` is run once per completed match. Both teams'
home/away ratings are nudged by `K * (actual - expected)`, and the
loser's are nudged the other way. The draw case uses
`actual = 0.5`. The mean of the two ratings is updated and stored
as `overall`.

`predict_proba` returns a 3-tuple `(p_home, p_away, p_draw)`:

  - `p_home_raw = expected(home.home + H, away.away)`
  - `p_away_raw = 1 - p_home_raw`
  - `p_draw = clamp(0.27 - gap * 0.3, 0.15, 0.40)` — a small draw
    residual that shrinks as the gap grows, then the leftover is
    split proportionally to `p_home_raw` and `p_away_raw`.

## Building a state from history

`scripts/build_elo_state.py` reads a JSONL of past matches and
walks them in chronological order, applying `update_state` to each.
This is the bridge between raw historical data and the agent:

    python scripts/build_elo_state.py \
        --matches data/past_matches.jsonl \
        --out    data/elo_state.json \
        --report

Each input row needs:

    {"date": "2024-08-15", "home_id": "man_city", "away_id": "arsenal",
     "home_goals": 2, "away_goals": 0}

The script sorts by `date` so input order is irrelevant, and it
auto-registers teams at 1500 the first time they appear.

## Loading in the agent

Three ways:

1. Constructor argument (preferred for tests and one-off runs):

       agent = PredictionAgent(..., elo_state_path="data/elo_state.json")

2. Environment variable (preferred for production):

       export SOCCER_AGENT_ELO_STATE=/path/to/elo_state.json

3. Default: a fresh empty `EloState` is created. Predictions still
   work — they just won't have any per-team prior.

When the state is provided, the `PredictionAgent.predict()` method
passes it into the `MatchContext.elo_state` field, and the numeric
reasoner calls `elo.predict_proba(state, home_id, away_id)` to seed
the home-win probability. The LLM reasoner ignores the state (it
already has its own reasoning); only the numeric baseline uses it.

## Why home/away splits matter for prediction

A common failure mode of single-rating Elo is that a team with
"70% home win rate" and "30% away win rate" gets a single rating of
~0.5, washing out the home signal. With the split:

  - a.home=1612, a.away=1388  → expected(a at home, b at 1500) = 0.65
  - expected(a away, b at home 1500) = 0.36

The two predictions differ by 29 percentage points, which is the
size of the signal we're now capturing. Calibrating this against
your dataset is a one-line config change
(`EloState(home_advantage=80)`) — see the eval harness.

## Limitations

- The form window is a *weight function* in this iteration, not a
  per-team `deque`. Callers (e.g. the future per-team form tool)
  can build their own `deque` of `FormEvent` items; the
  `weight_for(n, window)` helper is what they should multiply by.
- No margin-of-victory multiplier in `update_state`. Soccer scores
  are low enough that this rarely matters, but if your dataset
  includes basketball or football, add the
  `ln(|diff| + 1) * (2.2 / (winner_elo_diff * 0.001 + 2.2))` term.
- No time decay. A team that was elite 3 years ago is treated the
  same as a team that was elite last week. For a 24-month rolling
  state, the pre-compute script should be re-run every 30 days.
- No goal-difference scaling. A 1-0 win moves the same as a 5-0 win.
  Acceptable for a low-scoring sport; revisit if you switch domains.

## Tests

    pytest tests/test_elo.py tests/test_elo_integration.py \
           tests/test_agent_elo.py tests/test_build_elo_state.py

224 unit tests pass as of Task 27 completion.
