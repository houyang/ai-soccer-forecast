#!/usr/bin/env bash
# E2E shell verification: start uvicorn, hit the API, stop uvicorn.
set -e
DEMO=$(mktemp -d)
cd "$DEMO"
export SOCCER_AGENT_FIXTURES_DIR="$DEMO/fixtures"
export SOCCER_AGENT_DB_PATH="$DEMO/agent.db"
export SOCCER_AGENT_PORT=8765
export SOCCER_AGENT_HOST=127.0.0.1

mkdir -p fixtures
python -c "
import os
from pathlib import Path
from soccer_agent.eval.dataset import EVAL_CASES
from soccer_agent.eval.fixture_factory import materialize_all
root = Path(os.environ['SOCCER_AGENT_FIXTURES_DIR'])
materialize_all(EVAL_CASES, root)
print('materialized', len(EVAL_CASES), 'cases')
"

echo '--- starting uvicorn ---'
python -m uvicorn soccer_agent.api.server:app --host $SOCCER_AGENT_HOST --port $SOCCER_AGENT_PORT --log-level warning &
PID=$!
trap "kill $PID 2>/dev/null" EXIT

# wait for ready
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -fs http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/health > /dev/null 2>&1; then
    break
  fi
  sleep 0.3
done

echo '--- GET /health ---'
curl -s http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/health | python -m json.tool

echo '--- POST /predictions (first case) ---'
PAYLOAD=$(python -c "
from soccer_agent.eval.dataset import EVAL_CASES
from soccer_agent.agent import _season_for
c = EVAL_CASES[0]
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
curl -s -X POST http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/predictions -H 'content-type: application/json' -d "$PAYLOAD" | python -m json.tool

echo '--- POST /predictions/<id>/result ---'
MATCH_ID=$(python -c "from soccer_agent.eval.dataset import EVAL_CASES; print(EVAL_CASES[0].match_id)")
curl -s -X POST "http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/predictions/$MATCH_ID/result" -H 'content-type: application/json' -d '{"home_goals": 2, "away_goals": 1}' | python -m json.tool

echo '--- GET /predictions?limit=5 ---'
curl -s "http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/predictions?limit=5" | python -m json.tool | head -40

echo '--- GET /metrics ---'
curl -s http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/metrics | python -m json.tool | head -25

echo '--- GET /predictions/<bogus> ---'
curl -s -o /dev/null -w "status=%{http_code}\n" http://$SOCCER_AGENT_HOST:$SOCCER_AGENT_PORT/predictions/does_not_exist

echo 'DONE'
