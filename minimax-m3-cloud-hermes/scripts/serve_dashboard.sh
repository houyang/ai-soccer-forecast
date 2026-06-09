#!/usr/bin/env bash
# scripts/serve_dashboard.sh — start the FastAPI server with the dashboard UI.
#
# Defaults: binds 127.0.0.1:8000 (localhost only, no auth — by design for
# a personal-use dashboard). Override with SOCCER_AGENT_HOST / SOCCER_AGENT_PORT.
#
# This is the single command to run the whole product. The page polls
# /api/dashboard (composed from /metrics, /predictions, and the calibration
# report) and the forms POST to /predictions and /predictions/:id/result.

set -euo pipefail

HOST="${SOCCER_AGENT_HOST:-127.0.0.1}"
PORT="${SOCCER_AGENT_PORT:-8000}"

# Pick a provider. Default to stub so the server can boot without
# any LLM installed. Override with SOCCER_AGENT_LLM_PROVIDER=ollama
# (or openai/openrouter) for live predictions.
export SOCCER_AGENT_LLM_PROVIDER="${SOCCER_AGENT_LLM_PROVIDER:-stub}"

cd "$(dirname "$0")/.."

echo "soccer-agent dashboard"
echo "  url:    http://${HOST}:${PORT}/"
echo "  api:    http://${HOST}:${PORT}/api/dashboard"
echo "  docs:   http://${HOST}:${PORT}/docs"
echo "  db:     ${SOCCER_AGENT_DB_PATH:-data/soccer_agent.db}"
echo "  llm:    ${SOCCER_AGENT_LLM_PROVIDER}"
echo

exec python -m uvicorn soccer_agent.api.server:app \
  --host "$HOST" --port "$PORT" --no-access-log
