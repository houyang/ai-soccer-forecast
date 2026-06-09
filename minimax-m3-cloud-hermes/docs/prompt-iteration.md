# Prompt Iteration

The agent's `LLMReasoner` ships with a default system prompt. To find
a better one, we run a **sweep**: the same eval dataset under N prompt
candidates, scored head-to-head, with the winner surfaced.

## How to run

```bash
# Default: candidates in docs/prompts/, ollama as the backend.
python scripts/prompt_iterate.py

# Stub LLM (fast, deterministic, runs in ~0.2s; useful for CI):
SOCCER_AGENT_LLM_PROVIDER=stub python scripts/prompt_iterate.py

# Use a different model:
SOCCER_AGENT_LLM_MODEL=qwen2.5:1.5b python scripts/prompt_iterate.py
```

Outputs land in `docs/sweep_results/<timestamp>/`:
  - `per_candidate.json` — full metric_summary per candidate
  - `leaderboard.md` — ranked table

## Authoring candidates

Each candidate is a single `.md` file in `docs/prompts/`. The file's
basename is the candidate name. The full text of the file is the
system prompt. Keep them focused — a 5-line prompt with clear
constraints usually beats a 30-line prompt with rules scattered
throughout.

Conventions for a good candidate:
  - Name like `v4-draw-aware.md` (version + theme).
  - Tell the model how to weight signals (form vs odds vs injuries).
  - Specify the JSON contract.
  - Tell it to cite specific numbers in the rationale.

## Cost & wall-clock

Each (candidate × eval case) pair is one LLM call. The pinned eval
dataset is 10 cases; with 3 candidates that's 30 LLM calls. On a
sandbox CPU the 0.5B model takes ~4 minutes per call. On a host with
a fast ollama (Apple Silicon, GPU, or 7B+ on a decent CPU) the full
sweep finishes in 1-5 minutes total. Use `-m "not slow"` to skip the
few test cases that hit live ollama for long contexts.

## What to do with the result

Read the leaderboard. If the winner is the same prompt you already
ship, no action. If a new prompt wins by a meaningful margin (≥5%
accuracy or ≥0.05 brier), promote it:
  1. Update `SYSTEM_PROMPT` in `src/soccer_agent/reasoners/llm.py` to
     the winning candidate's text.
  2. Update the file in `docs/prompts/` to be a reference copy of the
     winning prompt (so the sweep stays reproducible).
  3. Commit and re-run `pytest -m "not ollama"` to confirm nothing
     regressed.
