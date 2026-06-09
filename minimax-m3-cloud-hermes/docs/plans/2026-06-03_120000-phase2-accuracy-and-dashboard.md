# Phase 2 — Accuracy + Dashboard

**Date:** 2026-06-03
**Branch:** master (linear, all in main, one commit per task)
**Working dir:** `/root/ai-agent-dev/`

## Goal

Get the agent from "shipping pipeline" (Phase 1) to "useful predictions on
real matches" (Phase 2). Two workstreams: **accuracy iteration** and
**dashboard**. User decision: accuracy first. LLM = local Ollama with
`qwen2.5:0.5b` (and `1.5b` available as upgrade). Stub LLM stays as the
default for unit tests. Dashboard deferred to a later task block in this
phase.

## Constraint check (the environment)

- 8 GB RAM, ARM64, no GPU, no Docker, no Ollama, but apt + internet OK
- `/root` has 323 GB free, `/tmp` is a 512M tmpfs → ollama models in `/root`
- Per `~/.hermes/memory` § "Stable prefs": TDD one-commit-per-task, Brier
  formula derived from docstring not implementation, E2E shell verification
  after every shipped artifact, plans under `docs/plans/`, credentials
  redacted in any saved state.

## Tasks

| # | Title                                                          | Status |
| - | -------------------------------------------------------------- | ------ |
| 21 | Install Ollama + pull `qwen2.5:0.5b` and `1.5b`                | next   |
| 22 | `OllamaClient` (OpenAI-compatible HTTP) + factory dispatch     | pending |
| 23 | Test OllamaClient against the live daemon (round-trip + 429)   | pending |
| 24 | Wire `LLMReasoner` → `OllamaClient` as the default real client | pending |
| 25 | Iteration harness: N prompt candidates × M eval cycles, keep best, log everything | pending |
| 26 | Bookmaker odds feed: real The Odds API client with fixture fallback | pending |
| 27 | Elo upgrade: home/away splits + form-window weighting          | pending |
| 28 | Calibration: `confidence` matches hit rate (ECE ≤ 0.05)        | pending |
| 29 | (dashboard block, deferred)                                    | —      |

## Task 21 detail

**What:** Download ollama linux/arm64 static binary, install to
`/usr/local/bin/ollama`, start the daemon on `:11434`, pull
`qwen2.5:0.5b` (~400MB) and `qwen2.5:1.5b` (~1GB), verify chat completion
end-to-end.

**Verification (in order):**

1. `ollama --version` → expected `ollama version 0.x.x`
2. `curl -sS http://localhost:11434/api/version` → 200 + version
3. `curl -sS http://localhost:11434/api/tags` → contains both `qwen2.5:0.5b` and `qwen2.5:1.5b`
4. `curl -sS -X POST http://localhost:11434/api/chat -d '{"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"Reply with the single word: OK"}],"stream":false}'` → `{"message":{"content":"OK"}}` (or similar, model is non-deterministic on the exact word, content must be non-empty and not an error)
5. Document install + model paths in `docs/ollama_setup.md`

**Pitfalls (do not hit these):**

- `set -e` in install script: every step must `|| true` and the script
  exits 0 if the daemon is up and both models are listed
- 8GB RAM, ARM64: don't try `qwen2.5:7b` or `14b` — they OOM-kill
- Don't bind the daemon to `0.0.0.0`: localhost only, no auth needed for
  this single-user sandbox
- Ollama stores models in `~/.ollama/models`; on this container that's
  `/root/.ollama/models`. Confirm before pulling (would be a 512M tmpfs
  problem if the home dir resolved to /tmp).
- First model pull is the slow one. Plan ~5 min for 0.5b, ~15 min for 1.5b
  on a typical 100Mbps sandbox connection.

**Done means:** a Python one-liner can do

```python
from soccer_agent.llm.client import get_client
c = get_client()  # or OllamaClient() explicitly
r = c.chat([{"role":"user","content":"Reply OK"}], model="qwen2.5:0.5b")
assert r.content.strip() != ""
```

and it works. After that, Task 22 builds the formal adapter.

## After 21

Build the `OllamaClient` against the real daemon (test #4 above proves the
HTTP shape). Wire it into `get_client()` as a third option after
`OpenAIClient` / `OpenRouterClient`. Run the existing 24/25 eval set
through it. Compare accuracy to the numeric baseline. Iterate.

## Deferred: dashboard

Static HTML/JS frontend served by the same FastAPI on `/dashboard`. Vanilla
JS, Plotly via CDN, no build step. Three pages: predictions, eval summary,
predict-this-match form. Localhost only. Estimated 5-7 tasks, separate
plan when the API is stable enough that frontend work doesn't churn it.

## Out of scope for Phase 2 round 1

- Real-time result polling beyond what ResultScout already does
- Multi-user / auth / cloud deploy
- WC 2026 final live prediction (the *target*, not the eval — needs the
  whole pipeline to be accurate enough to be worth running for real)
