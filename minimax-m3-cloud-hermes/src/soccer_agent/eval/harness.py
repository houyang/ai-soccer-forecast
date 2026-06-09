"""Eval harness (Task 18).

A single function/object that runs the agent over `EVAL_CASES`,
records the actual results (which are baked into the dataset), pulls
the joined rows back, computes the metric suite, and persists a
summary to the `eval_runs` table.

Idempotency: predictions are keyed by `prediction_id` (UUID), not
match_id. Re-running the harness will INSERT OR REPLACE those rows
on the same `prediction_id` if the agent emits the same UUID, but
the agent generates a fresh UUID per `predict()` call. To stay
truly idempotent on (case, fixtures, reasoner) the harness looks up
the existing prediction by match_id first and reuses it when present.

Usage:
    from soccer_agent.eval.harness import EvalHarness
    h = EvalHarness(fixtures_dir=Path("fixtures"), db_path=Path("agent.db"))
    summary = h.run()
    print(summary["accuracy"], summary["brier_mean"])

Or via the module-level helper:
    from soccer_agent.eval.harness import run_eval
    run_eval(fixtures_dir=..., db_path=..., output=Path("summary.json"))
"""
from __future__ import annotations

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..agent import PredictionAgent
from ..db import Database
from ..models import Match, Team
from ..reasoners import LLMReasoner, NumericReasoner
from ..tools import default_registry
from .dataset import EVAL_CASES, EvalCase
from .fixture_factory import materialize_case
from .metrics import metric_summary, row_from_db


ReasonerName = Literal["numeric", "llm"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _match_from_case(case: EvalCase) -> Match:
    return Match(
        match_id=case.match_id,
        competition=case.competition,
        kickoff=case.kickoff,
        home=Team(id=case.home_id, name=case.home_id),
        away=Team(id=case.away_id, name=case.away_id),
        venue_id=case.venue_id,
    )


def _score_pair(case: EvalCase) -> tuple[int, int]:
    """Build (home_goals, away_goals) that yield `case.actual_winner`."""
    if case.actual_winner == "home":
        return (2, 1)
    if case.actual_winner == "away":
        return (1, 2)
    return (1, 1)  # draw


@dataclass
class EvalHarness:
    """Run the agent over the eval dataset and score the predictions."""

    fixtures_dir: Path
    db_path: Path
    tool_names: list[str] = field(
        default_factory=lambda: [
            "form_recent", "injury_news", "h2h_history",
            "weather_venue", "odds_market", "venue_info",
        ]
    )
    reasoner: ReasonerName = "numeric"
    # If True, materialize fixtures even if a directory already exists.
    force_rematerialize: bool = False
    # Per-case probability (0..1) of flipping each of (form, h2h, odds)
    # so it disagrees with the actual result. `0.0` (default) keeps
    # the original "fair fixture" contract. Calibration runs use
    # `noise=0.4` to get realistic, measurable miscalibration.
    fixture_noise: float = 0.0
    # Seed for the per-case RNG inside the fixture factory. Same
    # (case, fixture_noise, fixture_seed) → same bytes. `None` means
    # "deterministic but not reproducible across runs" — i.e. process-
    # startup entropy. Pin a value when comparing Brier numbers.
    fixture_seed: int | None = None
    # Calibration (Task 31/35): the agent will fit/load a calibrator
    # from this directory at predict-time. `calibrator_key` is the
    # filename inside that dir (e.g. "isotonic", "isotonic_EPL",
    # "isotonic_UCL"). `None` disables calibration entirely.
    calibrator_root: str | os.PathLike | None = None
    calibrator_key: str | None = "isotonic"

    def __post_init__(self) -> None:
        self.fixtures_dir = Path(self.fixtures_dir)
        self.db_path = Path(self.db_path)
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)

    # -- low-level helpers --------------------------------------------------

    def _existing_prediction_for(self, match_id: str) -> dict[str, Any] | None:
        """Idempotency: if a prediction for match_id already exists, return it.
        We search by match_id, not prediction_id, because the agent emits
        a fresh UUID on every `predict()` call.
        """
        with Database(self.db_path) as db:
            rows = db.execute(
                "SELECT * FROM predictions WHERE match_id = ? ORDER BY created_at DESC LIMIT 1",
                (match_id,),
            )
        return rows[0] if rows else None

    def _record_result_if_missing(self, case: EvalCase) -> None:
        """`results.match_id` is PRIMARY KEY — INSERT OR REPLACE keeps this
        idempotent."""
        home, away = _score_pair(case)
        with Database(self.db_path) as db:
            db.insert_result({
                "match_id": case.match_id,
                "home_goals": home,
                "away_goals": away,
                "decided_at": case.kickoff.isoformat(),
            })

    def _materialize_fixtures(self) -> None:
        for case in EVAL_CASES:
            materialize_case(
                case, self.fixtures_dir,
                noise=self.fixture_noise,
                seed=self.fixture_seed,
            )

    def _make_agent(self) -> PredictionAgent:
        reasoner = (
            LLMReasoner() if self.reasoner == "llm" else NumericReasoner()
        )
        return PredictionAgent(
            registry=default_registry(),
            reasoner=reasoner,
            db_path=str(self.db_path),
            calibrator_root=self.calibrator_root,
            calibrator_key=self.calibrator_key,
        )

    # -- main entry point ---------------------------------------------------

    def run(self, output: Path | None = None) -> dict[str, Any]:
        """Run the full eval. Returns the metric summary dict. If `output`
        is provided, also writes a pretty JSON copy there."""
        self._materialize_fixtures()

        agent = self._make_agent()
        env_overrides = {
            "SOCCER_AGENT_FIXTURES_DIR": str(self.fixtures_dir),
            "SOCCER_AGENT_DB_PATH": str(self.db_path),
        }
        # Make the agent and DB see our paths.
        import os
        old_env = {k: os.environ.get(k) for k in env_overrides}
        os.environ.update(env_overrides)
        try:
            for case in EVAL_CASES:
                self._record_result_if_missing(case)
                if self._existing_prediction_for(case.match_id) is None:
                    match = _match_from_case(case)
                    asyncio.run(agent.predict(match, tool_names=self.tool_names))
                # else: idempotent re-run, re-score the existing row
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        # Pull joined rows and score.
        summary = self._score()

        # Persist the run.
        with Database(self.db_path) as db:
            db.insert_eval_run({
                "eval_id": str(uuid.uuid4()),
                "started_at": _now_iso(),
                "finished_at": _now_iso(),
                "dataset_path": "soccer_agent.eval.dataset:EVAL_CASES",
                "n_matches": summary["n_total"],
                "n_with_results": summary["n_resolved"],
                "metrics": {
                    "accuracy": summary["accuracy"],
                    "brier_mean": summary["brier_mean"],
                    "log_loss": summary["log_loss"],
                    "top_factor_hit_rate": summary["top_factor_hit_rate"],
                    "calibration_ece": summary["calibration_ece"],
                },
                "judge_score": None,
                "config": {
                    "reasoner": self.reasoner,
                    "tool_names": self.tool_names,
                },
            })

        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(_scrub_nans(summary), indent=2, default=str, allow_nan=False)
            )

        return summary

    # -- scoring ------------------------------------------------------------

    def _score(self) -> dict[str, Any]:
        """Pull all joined prediction+result rows and compute the metrics."""
        with Database(self.db_path) as db:
            preds = db.list_predictions(limit=10_000)
        # list_predictions already joins results (per Task 14: returns
        # result_brier, home_goals, away_goals, etc.)
        parsed = []
        for row in preds:
            r = dict(row)
            # the joined row gives us home_goals/away_goals if present
            parsed.append(row_from_db(r))
        s = metric_summary(parsed)
        s["reasoner"] = self.reasoner
        return s


# -- module-level convenience -------------------------------------------------

def run_eval(
    fixtures_dir: Path,
    db_path: Path,
    output: Path | None = None,
    reasoner: ReasonerName = "numeric",
) -> dict[str, Any]:
    """Shortcut: `EvalHarness(...).run(output=...)`."""
    return EvalHarness(
        fixtures_dir=fixtures_dir,
        db_path=db_path,
        reasoner=reasoner,
    ).run(output=output)
