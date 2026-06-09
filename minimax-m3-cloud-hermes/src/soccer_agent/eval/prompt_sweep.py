"""Prompt-iteration sweep harness (Task 25).

Given N prompt candidates and an eval dataset, run the same N eval
cases under each candidate, score them, and report which prompt
scored best. Used by `scripts/prompt_iterate.py` to find a prompt
template that beats the default before shipping it.

Design notes:
  - Per-candidate isolation: each candidate gets its own DB file
    (`<db_path>.stem__<candidate_name>.db`) so predictions for
    different prompts don't clobber each other.
  - Per-candidate client factory: the caller passes a `client_factory`
    so we can use the stub LLM in unit tests and a live ollama
    client in the real script.
  - No blending: the sweep measures the LLM-with-this-prompt alone
    (no secondary numeric reasoner). This is the right axis to
    optimize on; blending is a downstream concern.

Usage:
    from soccer_agent.eval.prompt_sweep import run_prompt_sweep
    result = run_prompt_sweep(
        candidates=[PromptCandidate("a", "..."), PromptCandidate("b", "...")],
        eval_cases=EVAL_CASES,
        client_factory=lambda: OpenAICompatClient(...),
        db_path=Path("agent.db"),
    )
    print(result.best.name)
"""

from __future__ import annotations

import asyncio
import json as jsonlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..agent import PredictionAgent
from ..db import Database, init_db
from ..llm import LLMClient
from ..models import Match, Team
from ..reasoners import LLMReasoner
from ..tools import default_registry
from .dataset import EvalCase
from .fixture_factory import materialize_case
from .metrics import metric_summary, row_from_db


# Default tool set mirrors EvalHarness so the sweep is apples-to-apples
# with the regular eval. The user can pass a different set if they want
# to measure the effect of fewer/more signals.
DEFAULT_SWEEP_TOOLS = (
    "form_recent",
    "injury_news",
    "h2h_history",
    "weather_venue",
    "odds_market",
    "venue_info",
)


@dataclass(frozen=True)
class PromptCandidate:
    """A single system-prompt variant to evaluate.

    `name` is used as the DB suffix and as the key in SweepResult.
    `system_prompt` is what gets passed to the LLMReasoner.
    """

    name: str
    system_prompt: str


@dataclass
class SweepResult:
    """Aggregate output of `run_prompt_sweep`."""

    per_candidate: dict[str, dict[str, Any]] = field(default_factory=dict)
    best: PromptCandidate | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_candidate": self.per_candidate,
            "best": self.best.name if self.best else None,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _candidate_db_path(base: Path, candidate: PromptCandidate) -> Path:
    """`<base>.stem__<sanitized_name>.db` so each candidate is isolated.

    Sanitize: replace anything not [A-Za-z0-9_] with '_'.
    """
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in candidate.name)
    return base.with_name(f"{base.stem}__{safe}{base.suffix or '.db'}")


def _match_from_case(case: EvalCase) -> Match:
    return Match(
        match_id=case.match_id,
        competition=case.competition,
        kickoff=case.kickoff,
        home=Team(id=case.home_id, name=case.home_id),
        away=Team(id=case.away_id, name=case.away_id),
        venue_id=case.venue_id,
    )


def _scrub_nans(o: Any) -> Any:
    """Recursively replace NaN/inf floats with None so the result is
    strict JSON (no NaN literal)."""
    if isinstance(o, float):
        if o != o or o in (float("inf"), float("-inf")):
            return None
    if isinstance(o, dict):
        return {k: _scrub_nans(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_scrub_nans(v) for v in o]
    return o


def _make_reasoner(candidate: PromptCandidate, client: LLMClient) -> LLMReasoner:
    """Build an LLMReasoner wired to a specific candidate's prompt."""
    return LLMReasoner(client=client, system_prompt=candidate.system_prompt)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def run_prompt_sweep(
    candidates: list[PromptCandidate],
    eval_cases: list[EvalCase],
    client_factory: Callable[[], LLMClient],
    db_path: Path,
    *,
    tool_names: tuple[str, ...] = DEFAULT_SWEEP_TOOLS,
    fixtures_dir: Path | None = None,
    output: Path | None = None,
) -> SweepResult:
    """Run the eval dataset under each prompt candidate and return a
    SweepResult with per-candidate metrics and a `.best` field.

    Side effects: writes one DB file per candidate under `db_path.parent`
    with the sanitized candidate name. If `output` is given, writes a
    JSON summary there.

    Tests use a stub client_factory (returns a _ConstantClient). The
    real script in `scripts/prompt_iterate.py` passes a factory that
    builds an ollama/openai client.
    """
    if not candidates:
        return SweepResult(per_candidate={}, best=None)

    db_path = Path(db_path)
    if fixtures_dir is None:
        # Use a temp dir per candidate under tmp_path style; but for
        # reproducibility with the rest of the codebase, default to
        # db_path.parent / "fixtures_<candidate>".
        fixtures_root = db_path.parent / "fixtures_sweep"
    else:
        fixtures_root = Path(fixtures_dir)

    fixtures_root.mkdir(parents=True, exist_ok=True)
    result = SweepResult()

    for candidate in candidates:
        cand_db = _candidate_db_path(db_path, candidate)
        cand_fx = fixtures_root / candidate.name
        cand_fx.mkdir(parents=True, exist_ok=True)

        # Materialize fixtures for this candidate (deterministic per case)
        for case in eval_cases:
            materialize_case(case, cand_fx)

        # Record the result for each case (idempotent: PRIMARY KEY is match_id)
        init_db(cand_db)
        for case in eval_cases:
            with Database(cand_db) as db:
                db.insert_result({
                    "match_id": case.match_id,
                    "home_goals": case.home_goals,
                    "away_goals": case.away_goals,
                    "decided_at": case.kickoff.isoformat(),
                })

        # Build the agent
        client = client_factory()
        reasoner = _make_reasoner(candidate, client)
        agent = PredictionAgent(
            registry=default_registry(),
            reasoner=reasoner,
            secondary_reasoner=None,  # measure the prompt alone
            db_path=cand_db,
        )

        # Run the eval
        import os
        old_fx = os.environ.get("SOCCER_AGENT_FIXTURES_DIR")
        os.environ["SOCCER_AGENT_FIXTURES_DIR"] = str(cand_fx)
        try:
            for case in eval_cases:
                match = _match_from_case(case)
                asyncio.run(agent.predict(match, tool_names=tool_names))
        finally:
            if old_fx is None:
                os.environ.pop("SOCCER_AGENT_FIXTURES_DIR", None)
            else:
                os.environ["SOCCER_AGENT_FIXTURES_DIR"] = old_fx

        # Score
        with Database(cand_db) as db:
            preds = db.list_predictions(limit=10_000)
        parsed = [row_from_db(dict(r)) for r in preds]
        summary = metric_summary(parsed)
        result.per_candidate[candidate.name] = summary

    # Pick the best by accuracy. Tie-break: lower brier_mean, then by
    # name (alphabetical) for determinism.
    if result.per_candidate:
        ranked = sorted(
            result.per_candidate.items(),
            key=lambda kv: (
                -kv[1].get("accuracy", 0.0),
                kv[1].get("brier_mean", 9.9),
                kv[0],
            ),
        )
        best_name = ranked[0][0]
        best_candidate = next(c for c in candidates if c.name == best_name)
        result.best = best_candidate

    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            jsonlib.dumps(
                _scrub_nans(result.to_dict()),
                indent=2,
                default=str,
                allow_nan=False,
            )
        )

    return result
