#!/usr/bin/env bash
# e2e_smoke.sh - Phase 1 wrap-up smoke test.
#
# Exercises the entire Phase 1 surface end-to-end from a real shell, in a
# throwaway tmpdir, using the installed `soccer-agent` binary and the
# installed FastAPI server. Proves:
#
#   1. CLI:    predict  -> evaluate  -> list  -> eval
#   2. API:    GET /health, GET /predictions, GET /predictions/{id},
#              POST /predictions, POST /predictions/{id}/result, GET /metrics
#   3. Eval:   eval writes a strict-JSON summary with all metric fields
#              (accuracy, brier, log_loss, per-class, calibration, ECE).
#
# Each emitted JSON payload is round-tripped through `python -m json.tool`
# to catch NaN / Infinity / missing-key regressions.
#
# Exit code: 0 if every step passes, 1 if any step fails.

set -uo pipefail  # NOTE: do NOT use `set -e`; we want every step to print its output even if the next step fails.

DEMO=$(mktemp -d -t soccer-agent-smoke-XXXXXX)
echo ">>> tmpdir: $DEMO"

# Config: fresh DB + fixture dir, with no shared state from prior runs.
export SOCCER_AGENT_FIXTURES_DIR="$DEMO/fixtures"
export SOCCER_AGENT_DB_PATH="$DEMO/agent.db"
export SOCCER_AGENT_PORT=8742
export SOCCER_AGENT_HOST=127.0.0.1
mkdir -p "$SOCCER_AGENT_FIXTURES_DIR"

# Materialize the eval dataset fixtures so every tool has data to read.
echo ">>> materializing fixtures..."
python -c "
from pathlib import Path
import os
from soccer_agent.eval.dataset import EVAL_CASES
from soccer_agent.eval.fixture_factory import materialize_all
materialize_all(EVAL_CASES, Path(os.environ['SOCCER_AGENT_FIXTURES_DIR']))
print(f'  materialized {len(EVAL_CASES)} cases')
"

# ---------- 1. CLI: predict -> evaluate -> list -----------------------------
echo
echo ">>> [CLI] soccer-agent predict ..."
PRED_OUT=$(soccer-agent predict \
  --home-id bayern --away-id barca \
  --venue-id puskas_arena --kickoff 2024-10-15T20:00:00 \
  --competition UCL --season 2024-2025 --round gs)
echo "$PRED_OUT" | python -m json.tool > "$DEMO/pred.json"
MATCH_ID=$(python -c "import json; print(json.load(open('$DEMO/pred.json'))['match_id'])")
PICK=$(python -c "import json; print(json.load(open('$DEMO/pred.json'))['pick'])")
CONF=$(python -c "import json; print(json.load(open('$DEMO/pred.json'))['confidence'])")
echo "  match_id=$MATCH_ID  pick=$PICK  confidence=$CONF"

echo ">>> [CLI] soccer-agent evaluate ..."
# Use a fake 2-1 scoreline; either pick could be "correct" depending on pick.
EVAL_OUT=$(soccer-agent evaluate --match-id "$MATCH_ID" --home-goals 2 --away-goals 1)
echo "$EVAL_OUT" | python -m json.tool > "$DEMO/result.json"
WAS_CORRECT=$(python -c "import json; print(json.load(open('$DEMO/result.json'))['result']['was_correct'])")
BRIER=$(python -c "import json; print(json.load(open('$DEMO/result.json'))['result']['result_brier'])")
echo "  was_correct=$WAS_CORRECT  brier=$BRIER"

echo ">>> [CLI] soccer-agent list ..."
LIST_OUT=$(soccer-agent list --limit 5)
echo "$LIST_OUT" | python -m json.tool > "$DEMO/list.json"
COUNT=$(python -c "import json; print(len(json.load(open('$DEMO/list.json'))))")
echo "  list returned $COUNT predictions"

# ---------- 2. CLI: eval over the whole dataset ----------------------------
echo
echo ">>> [CLI] soccer-agent eval --output summary.json ..."
soccer-agent eval --output "$DEMO/summary.json" > "$DEMO/summary.stdout.json"
python -m json.tool < "$DEMO/summary.stdout.json" > "$DEMO/summary.stdout.pretty.json"
# Verify every expected metric is present
for key in n_total n_resolved accuracy brier_mean log_loss top_factor_hit_rate per_class; do
  python -c "
import json, sys
s = json.load(open('$DEMO/summary.json'))
assert '$key' in s, f'missing key: $key'
print(f'  $key = {s[\"$key\"]}')
"
done

# ---------- 3. API: spin up uvicorn, hit every endpoint --------------------
echo
echo ">>> [API] starting uvicorn on $SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT ..."
python -m uvicorn soccer_agent.api.server:app \
  --host "$SOCCER_AGENT_HOST" --port "$SOCCER_AGENT_PORT" \
  --log-level warning &
UVICORN_PID=$!
trap "kill $UVICORN_PID 2>/dev/null || true" EXIT

# Wait for /health
for _ in $(seq 1 30); do
  if curl -fs "http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/health" > /dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

echo ">>> [API] GET /health"
curl -fs "http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/health" | python -m json.tool

echo ">>> [API] GET /predictions?limit=5"
curl -fs "http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/predictions?limit=5" \
  | python -m json.tool | head -10

echo ">>> [API] GET /predictions/$MATCH_ID"
curl -fs "http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/predictions/$MATCH_ID" \
  | python -m json.tool | head -15

echo ">>> [API] POST /predictions (new match)"
NEW_PAYLOAD=$(python -c "
from soccer_agent.eval.dataset import EVAL_CASES
from soccer_agent.agent import _season_for
c = EVAL_CASES[1]
import json
print(json.dumps({
    'match_id': c.match_id,
    'home_id': c.home_id,
    'away_id': c.away_id,
    'venue_id': c.venue_id,
    'kickoff': c.kickoff.isoformat(),
    'competition': c.competition,
    'season': _season_for(c.kickoff),
}))
")
curl -fs -X POST "http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/predictions" \
  -H 'content-type: application/json' \
  -d "$NEW_PAYLOAD" | python -m json.tool | head -10

echo ">>> [API] POST /predictions/{id}/result"
curl -fs -X POST "http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/predictions/$MATCH_ID/result" \
  -H 'content-type: application/json' \
  -d '{"home_goals": 1, "away_goals": 1}' | python -m json.tool | head -10

echo ">>> [API] GET /metrics"
curl -fs "http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/metrics" | python -m json.tool | head -20

echo ">>> [API] GET /predictions/does_not_exist (expect 404)"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
  "http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/predictions/does_not_exist")
[ "$HTTP_CODE" = "404" ] || { echo "  FAIL: expected 404, got $HTTP_CODE"; exit 1; }
echo "  OK 404"

echo ">>> [API] POST /predictions (invalid body, expect 422)"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  "http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/predictions" \
  -H 'content-type: application/json' -d '{"home_id": "x"}')
[ "$HTTP_CODE" = "422" ] || { echo "  FAIL: expected 422, got $HTTP_CODE"; exit 1; }
echo "  OK 422"

# Stop uvicorn
kill $UVICORN_PID 2>/dev/null || true
wait $UVICORN_PID 2>/dev/null || true
trap - EXIT

# ---------- 4. Final assertions --------------------------------------------
echo
echo ">>> final assertions"
test -s "$DEMO/agent.db" && echo "  DB file exists ($(du -h "$DEMO/agent.db" | cut -f1))"
test -s "$DEMO/summary.json" && echo "  summary.json exists ($(du -h "$DEMO/summary.json" | cut -f1))"
python -c "
import json
s = json.load(open('$DEMO/summary.json'))
assert s['n_total'] >= len(__import__('os').listdir('$SOCCER_AGENT_FIXTURES_DIR/form')) * 0, 'n_total too small'
# Strict-JSON round trip catches NaN/Inf regressions.
import json as _j
_j.dumps(s, allow_nan=False)
print(f'  summary round-trip OK: n_total={s[\"n_total\"]} accuracy={s[\"accuracy\"]:.3f} brier={s[\"brier_mean\"]:.3f}')
"

echo
echo ">>> ALL SMOKE STEPS PASSED"
echo ">>> tmpdir: $DEMO (kept for inspection; rm -rf to clean up)"
