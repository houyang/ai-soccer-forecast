# AGENTS.md

This file defines the operating rules for AI coding agents and human contributors working in this project. It is intentionally self-contained and applies to this repository only.

## Mission

Build and maintain a reliable, readable, secure, and well-tested Python project template that can be used as a company-standard starting point for production services, libraries, and automation tools.

## Working Principles

- Prefer simple, explicit Python over clever abstractions.
- Preserve existing behavior unless the requested change explicitly requires otherwise.
- Keep changes focused on the task at hand.
- Treat tests, typing, linting, and formatting as part of the implementation, not as cleanup.
- Make decisions that a future maintainer can understand from the code and nearby documentation.
- Do not introduce external services, network calls, telemetry, or background processes without a clear requirement.
- Do not reference files, paths, or configuration outside this repository.

## Repository Boundaries

- Work only inside this repository.
- Do not assume access to anything outside this repository or to private credentials.
- Do not modify generated files unless the project explicitly tracks them.
- Do not remove or overwrite user changes. If unexpected changes appear, inspect them and work around them.
- Keep secrets out of source control. Use `.env.example` for names and documentation only.

## Expected Project Layout

```text
.
├── src/soccer/              # Importable package code
├── tests/                   # Automated tests
├── docs/                    # Project documentation
├── .github/workflows/       # Continuous integration
├── AGENTS.md                # Agent and contributor instructions
├── README.md                # Human-facing project overview
├── pyproject.toml           # Packaging and tool configuration
├── Makefile                 # Local developer commands
└── .gitignore               # Ignored local/build artifacts
```

## Environment Setup

Use an isolated virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pre-commit install
```

When a dependency is added, update `pyproject.toml` and verify the project from a clean environment when practical.

## Standard Commands

```bash
make format
make lint
make typecheck
make test
make coverage
make check
```

If `make` is unavailable, use the equivalent commands from `pyproject.toml`:

```bash
ruff format .
ruff check .
mypy src tests
pytest
pytest --cov=soccer --cov-report=term-missing
```

## Coding Standards

- Target Python 3.11 or newer unless project requirements say otherwise.
- Use the `src/` layout for importable code.
- Use meaningful module, class, function, and variable names.
- Keep public interfaces small and documented.
- Use type annotations for public APIs and non-obvious internal code.
- Prefer `pathlib.Path` over string path manipulation.
- Prefer dataclasses or typed structures for structured data.
- Prefer dependency injection for side effects such as I/O, time, randomness, and HTTP calls.
- Raise specific exceptions with actionable messages.
- Avoid broad `except Exception` blocks unless the error is re-raised or deliberately translated.
- Avoid mutable default arguments.
- Avoid hidden global state.
- Keep functions short enough that their behavior is easy to test directly.
- Add comments only when they clarify intent, constraints, or non-obvious tradeoffs.

## Formatting And Linting

Ruff is the source of truth for formatting and linting.

- Run `ruff format .` before completing code changes.
- Run `ruff check .` and fix reported issues.
- Do not manually fight the formatter.
- Keep line length within the configured limit.
- Prefer automated fixes only after reviewing their impact.

## Typing

mypy is the source of truth for static typing.

- Run `mypy src tests`.
- Do not silence type errors without a narrow explanation.
- Prefer precise types over `Any`.
- Use `typing.Protocol` when behavior matters more than concrete implementation.
- Keep `# type: ignore[...]` comments specific and rare.

## Testing Standards

pytest is the source of truth for tests.

- Add or update tests for every behavior change.
- Keep unit tests fast, deterministic, and independent.
- Put shared fixtures in `tests/conftest.py` only when they are used by multiple test modules.
- Prefer clear Arrange-Act-Assert test structure.
- Test public behavior rather than private implementation details.
- Include regression tests for bug fixes.
- Avoid tests that depend on wall-clock time, network access, machine-specific paths, or test order.
- Use temporary directories and monkeypatching for filesystem and environment interactions.

## Coverage Expectations

- New business logic should include meaningful test coverage.
- Coverage should prove important branches and error paths, not just import modules.
- Do not reduce coverage without documenting why the missing coverage is acceptable.

## Dependency Policy

- Add dependencies only when they provide clear value over standard library code.
- Prefer mature, well-maintained packages with stable APIs.
- Keep runtime dependencies separate from development dependencies.
- Avoid adding dependencies for trivial helpers.
- Pin only when reproducibility or compatibility requires it.
- Re-run tests and quality checks after dependency changes.

## Security Standards

- Never commit secrets, tokens, private keys, passwords, or real credentials.
- Do not log secrets or sensitive personal data.
- Validate and normalize untrusted input at boundaries.
- Use safe parsers and encoders for structured data.
- Avoid shell execution. When shell execution is necessary, avoid `shell=True` and pass arguments as lists.
- Use least-privilege file permissions and narrow filesystem access.
- Treat deserialization, archive extraction, path traversal, and subprocess usage as high-risk areas.

## Documentation Standards

- Keep `README.md` accurate for setup, usage, and common workflows.
- Document public APIs when behavior is not obvious from names and types.
- Put longer design notes in `docs/`.
- Update documentation in the same change that alters user-facing behavior.
- Prefer examples that can be copied and run from the repository root.

## Pull Request Expectations

Before considering work complete:

1. Format code.
2. Run linting.
3. Run type checking.
4. Run tests.
5. Update documentation when behavior, commands, or configuration changed.
6. Summarize changed files and validation results.

## CI Expectations

Continuous integration should run:

- Ruff format check
- Ruff lint
- mypy
- pytest with coverage

CI failures should be treated as blocking unless explicitly waived by project maintainers.

## Review Guidance

When reviewing code, prioritize:

1. Correctness and behavior regressions.
2. Security and data safety.
3. Test coverage for changed behavior.
4. API clarity and maintainability.
5. Simplicity and consistency with the existing codebase.
6. Formatting and style issues after substantive concerns.

Review comments should be specific, actionable, and tied to files or behavior.

## Agent Workflow

When an AI agent works on this project:

1. Inspect relevant files before editing.
2. State the intended change briefly when the task is non-trivial.
3. Make the smallest coherent change that satisfies the request.
4. Add or update tests with the implementation.
5. Run the narrowest useful validation first, then broader checks when practical.
6. Report what changed and which checks passed or failed.

## Prohibited Agent Behavior

Agents must not:

- Modify files outside this repository.
- Reference locations outside this repository.
- Invent credentials, endpoints, or deployment environments.
- Delete user work to make a task easier.
- Use destructive Git commands unless explicitly requested.
- Hide failing checks.
- Leave temporary debugging output in committed code.
- Add large generated artifacts without a clear reason.
- Replace established tooling without explicit approval.

## Error Handling Policy

- Fail fast for programmer errors.
- Return clear, actionable messages for user-facing errors.
- Preserve exception context when translating errors.
- Test expected failure modes.
- Avoid swallowing errors silently.

## Logging Policy

- Use the standard `logging` module for library code.
- Do not configure global logging in importable modules.
- Keep log messages useful for diagnosis.
- Do not log secrets, tokens, credentials, or sensitive input.

## Configuration Policy

- Read configuration at application boundaries.
- Keep defaults explicit and documented.
- Use environment variables for deployment-specific settings.
- Document expected environment variables in `.env.example`.
- Avoid import-time environment reads unless there is a specific reason.

## Release Policy

- Keep version metadata in `pyproject.toml`.
- Use semantic versioning when publishing packages.
- Document notable changes in release notes or a changelog when the project starts releasing versions.
- Ensure `make check` passes before release.

## Definition Of Done

A change is done when:

- The requested behavior is implemented.
- Tests cover the new or changed behavior.
- Formatting, linting, typing, and tests pass, or any skipped validation is clearly reported.
- Documentation is updated where relevant.
- The final summary names the important changed files and validation commands.
