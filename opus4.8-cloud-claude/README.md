# Soccer Prediction Agent

A multi-tool agent that builds a typed dossier for an upcoming match (form, injuries,
head-to-head, weather, venue, bookmaker odds), reasons with a local Ollama model behind
a swappable interface, logs a 1X2 prediction with rationale and confidence, settles
results on demand, and scores itself against the bookmaker via an offline eval harness.

## Setup

    python -m venv .venv && source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e ".[dev]"

## Commands

    soccer eval --scenario ucl-2025-26 --reasoner fake
    soccer eval --scenario wc-2026-final --reasoner ollama
    soccer predict --match wc-final
    soccer settle
    soccer report

Configuration is read from environment variables (see `.env.example`): data directory,
Ollama host/model/timeout, provider mode (`fixture|http`), and reasoner (`fake|ollama`).
The `fake` reasoner and `fixture` providers are fully offline and require no network or
Ollama; they are the default and what the test suite and CI use.

## Quality gate

    make check   # ruff format check + ruff lint + mypy + pytest with coverage

## Architecture

See `docs/architecture.md` and the Phase 1 design at
`docs/superpowers/specs/2026-06-08-soccer-prediction-agent-phase1-design.md`.
