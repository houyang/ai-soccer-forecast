# Soccer Forecast Agent Evaluation

This repository is an aggregate workspace containing multiple generated Python projects for a FIFA 2026 World Cup soccer prediction agent. Each subfolder represents a different LLM model and coding-agent run.

Local LLM suitability is assessed against a MacBook Pro M5 Pro with 24 GB RAM, 15 CPU cores, and 16 GPU cores.

## Project Runs

| Subfolder | LLM model | Coding agent | Source files | Test files | Test functions |
| --- | --- | --- | ---: | ---: | ---: |
| `opus4.8-cloud-claude` | Opus 4.8 | Claude Code | 52 | 19 | 73 |
| `minimax-m3-cloud-hermes` | MiniMax M3 | Hermes | 71 | 28 | 294 |
| `gpt5.5-cloud-codex` | GPT-5.5 | Codex | 16 | 6 | 11 |
| `glx4.7-cloud-claude` | GLM/GLX 4.7 | Claude Code | 27 | 11 | 18 |
| `gemma4-12bmlx-local-codex` | Gemma 4 12B MLX local | Codex | 16 | 0 pytest-discoverable files | 1 |

Counts exclude `.venv` and `__pycache__` directories. `gemma4-12bmlx-local-codex/tests/evaluation/backtest.py` contains a test-like function, but the file name is not pytest-discoverable by the normal `test*.py` convention.

## Test Suite Summary

### `opus4.8-cloud-claude`

Opus 4.8 with Claude Code produced the most template-compliant test suite. Tests are grouped around agent orchestration, CLI behavior, configuration, evaluation harness behavior, scenario settlement, persistent store behavior, model lifecycle/probability validation, dossier construction, provider fixtures, HTTP stubs, and reasoning backends. The suite also includes fake, prompt, and Ollama reasoner tests, which makes local-LLM integration testable without requiring live network access by default.

Reported validation from existing evaluations: 80 passing tests, 96% coverage, `ruff` passing, `mypy` passing, and CI present.

### `minimax-m3-cloud-hermes`

MiniMax M3 with Hermes produced the broadest and most domain-heavy test suite. Its tests cover core agent behavior, Elo integration, calibration, calibration storage, fixture factories, form tools, odds API/live odds, LLM and numeric reasoners, metrics, evaluation harnesses, prompt sweeps, dashboard/API/CLI surfaces, database behavior, registry behavior, and self-evaluation.

Reported validation from existing evaluations is mixed: one report measured 307 passing tests with `PYTHONPATH=src`, while another emphasized that the project had stronger domain evaluation than template hygiene. Known concerns include lint failures, committed runtime artifacts, and heavier operational surface area.

### `gpt5.5-cloud-codex`

GPT-5.5 with Codex produced a smaller but clean test suite. Tests focus on the prediction agent, competition catalog behavior, live World Cup loading, storage, and evaluation. This project is less ambitious than Opus or MiniMax/Hermes, but its tests align closely with the implemented behavior.

Reported validation from existing evaluations: 11 passing tests, `ruff` passing, and `mypy` passing.

### `glx4.7-cloud-claude`

GLM/GLX 4.7 with Claude Code produced structural tests for weather, sessions, API football integration, state, odds, evaluation, configuration, models, schemas, injuries, and the prediction workflow. The test list suggests an ambitious architecture, but the existing evaluations agree that the implementation is incomplete or difficult to run.

Reported validation from existing evaluations: tests fail during collection or were not runnable in the evaluated environment, and the project lacks several expected template files.

### `gemma4-12bmlx-local-codex`

Gemma 4 12B MLX local with Codex produced only a minimal `tests/evaluation/backtest.py` file and no pytest-discoverable `test*.py` files. Existing evaluations describe the implementation as skeletal, with syntax and packaging issues.

Reported validation from existing evaluations: no discoverable pytest tests, and source compilation failures were reported.

## Existing Evaluation Scores

The root evaluation reports are:

| Report | Scoring format |
| --- | --- |
| `evaluation-codex.txt` | Numeric 10-point ratings |
| `evaluation-claude.txt` | Numeric 10-point ratings plus letter grades |
| `evaluation-grok.txt` | Numeric 10-point ratings |
| `evaluation-gemini.txt` | Letter grades converted to 10-point ratings |

Gemini letter grades are converted with this mapping: `A+` = 9.5, `A` = 9.0, `A-` = 8.7, `B+` = 8.5, `C` = 7.0, and `D` = 6.0. The aggregate numeric rating uses a simple average across all four reports.

| Project | Codex score | Claude score | Grok score | Gemini grade | Gemini score | Simple average |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `minimax-m3-cloud-hermes` | 8.0 | 7.0 | 9.0 | A+ | 9.5 | 8.38 |
| `opus4.8-cloud-claude` | 9.0 | 9.0 | 8.5 | C | 7.0 | 8.38 |
| `gpt5.5-cloud-codex` | 7.5 | 7.5 | 7.5 | B+ | 8.5 | 7.75 |
| `glx4.7-cloud-claude` | 3.0 | 4.5 | 4.0 | D | 6.0 | 4.38 |
| `gemma4-12bmlx-local-codex` | 1.0 | 2.5 | 1.5 | C | 7.0 | 3.00 |

Simple average formula:

```text
(Codex score + Claude score + Grok score + converted Gemini score) / 4
```

## Aggregate Ranking

1. `minimax-m3-cloud-hermes` - 8.38/10. Most capable domain prototype: strong evaluation, calibration, Elo, dashboard/API work, but weaker repository hygiene.
2. `opus4.8-cloud-claude` - 8.38/10. Best overall template-quality implementation: clean layout, broad tests, strict tooling, and strong maintainability.
3. `gpt5.5-cloud-codex` - 7.75/10. Clean, conservative baseline with fewer features and fewer tests.
4. `glx4.7-cloud-claude` - 4.38/10. Ambitious structure, but incomplete and not reliably runnable.
5. `gemma4-12bmlx-local-codex` - 3.00/10. Minimal, non-runnable or barely runnable skeleton with no meaningful pytest coverage.

## Key Takeaways

The strongest template implementation is `opus4.8-cloud-claude`. The strongest domain experiment is `minimax-m3-cloud-hermes`. The most reliable smaller baseline is `gpt5.5-cloud-codex`.

The evaluation reports disagree most sharply on `minimax-m3-cloud-hermes`, `opus4.8-cloud-claude`, and `gemma4-12bmlx-local-codex`. That disagreement comes from weighting different criteria: Gemini favored feature breadth and documentation, while the numeric reports weighted runnable quality gates, source hygiene, and template adherence more heavily.
