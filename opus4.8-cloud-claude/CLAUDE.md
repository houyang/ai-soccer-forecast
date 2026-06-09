# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current State

This repository is **empty** — no code, packaging, or tooling exists yet. It is one slot in a
multi-model comparison (sibling directories under `soccer-forcast/` are independent
implementations by other models). The task is to build the project from scratch per the spec.

The authoritative spec is `../AGENTS.md` (the project's operating rules). Read it before
starting work — the rules below summarize the parts that affect how you build, but `AGENTS.md`
governs and goes deeper on review, security, and PR expectations.

## What to Build

A reliable, well-tested **Python project template** intended as a company-standard starting
point for production services, libraries, and automation tools. The importable package is
`soccer`, using a `src/` layout.

Target layout (create these as you go):

```text
src/soccer/          # Importable package code (src layout)
tests/               # pytest tests; shared fixtures in tests/conftest.py only when reused
docs/                # Longer design notes
.github/workflows/   # CI: ruff format check, ruff lint, mypy, pytest+coverage
README.md            # Human-facing setup/usage
pyproject.toml       # Packaging + tool config; holds version metadata
Makefile             # Developer command shortcuts
.env.example         # Names/docs of env vars only — never real secrets
```

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

## Standard Commands

Prefer the `Makefile` targets; fall back to the raw commands if `make` is unavailable.

| Make target      | Equivalent command                                    |
|------------------|-------------------------------------------------------|
| `make format`    | `ruff format .`                                       |
| `make lint`      | `ruff check .`                                         |
| `make typecheck` | `mypy src tests`                                       |
| `make test`      | `pytest`                                               |
| `make coverage`  | `pytest --cov=soccer --cov-report=term-missing`       |
| `make check`     | all of the above (must pass before release)           |

Run a single test: `pytest tests/test_module.py::test_name`.

## Non-Obvious Conventions

These are the project's hard constraints — not generic advice:

- **Tooling is canonical, not advisory.** Ruff is the source of truth for formatting *and*
  linting; mypy for typing; pytest for tests. Don't fight the formatter or silence type errors
  without a narrow, specific `# type: ignore[code]`.
- **Target Python 3.11+.** Use the `src/` layout for all importable code.
- **Dependency injection for side effects.** I/O, time, randomness, and HTTP must be injected so
  they're testable — this drives much of the API shape. Tests must not touch the network,
  wall-clock time, machine-specific paths, or rely on test order; use tmp dirs + monkeypatch.
- **No import-time side effects.** Read config and env vars at application boundaries, not at
  import. Library modules must not configure global logging — use the stdlib `logging` module.
- **Repository boundary is strict.** Work only inside this directory. Do not reference, read, or
  write paths outside it, and do not invent credentials, endpoints, or deployment environments.
  (`../AGENTS.md` is the one external file to consult as the spec.)
- **Tests ship with behavior changes.** Every behavior change adds/updates tests and exercises
  important branches and error paths, including regression tests for bug fixes.

## Definition of Done

Before considering work complete: format, lint, typecheck, and test all pass (or any skipped
check is explicitly reported); docs updated when behavior/commands/config changed; final summary
names the important changed files and the validation commands run.
