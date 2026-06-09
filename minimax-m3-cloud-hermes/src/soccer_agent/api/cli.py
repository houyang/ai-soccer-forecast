"""Command-line interface for soccer-agent.

Three subcommands, all JSON to stdout:

  soccer-agent predict  --home-id H --away-id A --venue-id V --kickoff ISO
                        [--competition UCL] [--season 2025/26] [--round final]
                        [--tools form_recent,injury_news,...]

  soccer-agent evaluate --match-id M --home-goals N --away-goals N

  soccer-agent list    [--limit 10]   # most recent predictions

The CLI is the canonical "make a prediction and log it" entry point; the
HTTP API (Task 19) reuses the same predict/evaluate functions.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from typing import Iterable, Optional

import click

from soccer_agent.agent import PredictionAgent
from soccer_agent.db import Database, get_db
from soccer_agent.models import Match, Team
from soccer_agent.reasoners import NumericReasoner
from soccer_agent.tools import default_registry


# ---------- helpers -----------------------------------------------------------

DEFAULT_TOOL_LIST = "form_recent,injury_news,h2h_history,weather_venue,odds_market,venue_info"


def _default_db_path() -> str:
    """Resolve the DB path the agent should open (from env or default)."""
    return os.environ.get("SOCCER_AGENT_DB_PATH") or "data/soccer_agent.db"


def _build_agent() -> PredictionAgent:
    """Construct a PredictionAgent with the default tool registry and numeric reasoner.

    For the CLI we deliberately do NOT pass a secondary reasoner: the
    NumericReasoner is a deterministic, single-source scorer, and blending
    it with a second numeric instance would just concatenate the same
    rationale twice (the agent's `is not self.reasoner` check only
    compares Python identity, not class). LLM-backed blending will come
    in Phase 2 when we have a meaningfully different second opinion.
    """
    registry = default_registry()
    reasoner = NumericReasoner()
    return PredictionAgent(
        registry=registry,
        reasoner=reasoner,
        secondary_reasoner=None,
        db_path=_default_db_path(),
    )


def _emit(payload: dict | list) -> None:
    """Single line of JSON to stdout. Errors go to stderr + exit 1.
    NaN / inf are emitted as null (JSON has no NaN literal; downstream
    consumers (jq, dashboards) handle null better than NaN)."""
    from soccer_agent.eval.harness import _scrub_nans
    click.echo(json.dumps(_scrub_nans(payload), default=str, sort_keys=False, allow_nan=False))


# ---------- command group -----------------------------------------------------

@click.group()
@click.version_option()
def main() -> None:
    """soccer-agent: multi-tool football match prediction agent."""


# ---------- predict -----------------------------------------------------------

@main.command("predict")
@click.option("--home-id", required=True, help="Home team id (e.g. man_city).")
@click.option("--away-id", required=True, help="Away team id (e.g. real_madrid).")
@click.option("--venue-id", required=True, help="Venue id (e.g. puskas_arena).")
@click.option("--kickoff", required=True, help="ISO datetime, e.g. 2025-05-30T20:00:00.")
@click.option("--competition", default="UCL", show_default=True)
@click.option("--season", default="2025/26", show_default=True)
@click.option("--round", default=None, help="Optional tournament round (final, sf, etc.).")
@click.option("--match-id", default=None,
              help="Optional explicit match id; default = home_vs_away__<date>.")
@click.option("--tools", "tool_names", default=DEFAULT_TOOL_LIST, show_default=True,
              help="Comma-separated tool names to invoke.")
def predict_cmd(
    home_id: str,
    away_id: str,
    venue_id: str,
    kickoff: str,
    competition: str,
    season: str,
    round: Optional[str],
    match_id: Optional[str],
    tool_names: str,
) -> None:
    """Run the agent, log the prediction, and print a JSON Prediction row."""
    match_id = match_id or _synth_match_id(home_id, away_id, kickoff)
    match = Match(
        match_id=match_id,
        competition=competition,
        round=round,
        kickoff=datetime.fromisoformat(kickoff),
        home=Team(id=home_id, name=home_id.replace("_", " ").title()),
        away=Team(id=away_id, name=away_id.replace("_", " ").title()),
        venue_id=venue_id,
    )
    tools = [t.strip() for t in tool_names.split(",") if t.strip()]
    agent = _build_agent()
    try:
        pred = asyncio.run(agent.predict(match, tool_names=tools))
    except Exception as e:  # surface the failure cleanly
        click.echo(f"error: predict failed: {e}", err=True)
        sys.exit(1)
    _emit(_pred_to_dict(pred))


# ---------- evaluate ----------------------------------------------------------

@main.command("evaluate")
@click.option("--match-id", required=True)
@click.option("--home-goals", required=True, type=int)
@click.option("--away-goals", required=True, type=int)
def evaluate_cmd(match_id: str, home_goals: int, away_goals: int) -> None:
    """Record a result, self-evaluate the most recent prediction for this match."""
    agent = _build_agent()
    try:
        # Persist the result first (agent.evaluate() expects it to exist).
        agent.db.insert_result({
            "match_id": match_id,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "was_correct": None,    # agent.evaluate() fills these
            "brier": None,
            "top_factor_hit": None,
        })
        pred = asyncio.run(agent.evaluate(match_id))
    except KeyError as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(2)
    except RuntimeError as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(3)
    except Exception as e:
        click.echo(f"error: evaluate failed: {e}", err=True)
        sys.exit(1)
    _emit(_pred_to_dict(pred, with_result=True))


# ---------- list --------------------------------------------------------------

@main.command("list")
@click.option("--limit", default=10, show_default=True, type=int)
def list_cmd(limit: int) -> None:
    """Print the most recent N predictions (JSON array)."""
    db = Database(_default_db_path())
    rows = db.list_predictions(limit=limit)
    _emit(rows)


# ---------- eval --------------------------------------------------------------

@main.command("eval")
@click.option("--output", "output", default=None, type=click.Path(),
              help="Optional path to write the JSON summary to.")
@click.option("--reasoner", default="numeric", show_default=True,
              type=click.Choice(["numeric", "llm"]),
              help="Reasoner to use. 'llm' falls back to numeric if no API key is set.")
def eval_cmd(output: str | None, reasoner: str) -> None:
    """Run the agent over the eval dataset and print a JSON summary.

    The eval is idempotent: re-running inserts a new eval_runs row but
    does NOT re-predict cases that already have a prediction in the DB.
    """
    from pathlib import Path as _P
    from soccer_agent.eval.harness import EvalHarness
    h = EvalHarness(
        fixtures_dir=_P(os.environ.get("SOCCER_AGENT_FIXTURES_DIR", "data/fixtures")),
        db_path=_P(_default_db_path()),
        reasoner=reasoner,  # type: ignore[arg-type]
    )
    summary = h.run(output=_P(output) if output else None)
    _emit(summary)


# ---------- helpers (private) -------------------------------------------------

def _synth_match_id(home_id: str, away_id: str, kickoff: str) -> str:
    """Derive a stable match id from the participants + date (not time)."""
    date_part = kickoff.split("T", 1)[0]
    return f"{home_id}_vs_{away_id}__{date_part}"


def _pred_to_dict(pred, with_result: bool = False) -> dict:
    """Serialize a Prediction to a JSON-friendly dict.

    Exposes the public aliases (`pick`/`confidence`/`rationale`) so the
    CLI output matches the FastAPI contract; the internal `final_*`
    column names are an implementation detail of the SQLite layer and
    should not leak to consumers.

    `reasoner_outputs` is a `list[ReasonerOutput]` after Pydantic
    round-trip — we have to dump each model explicitly. A bare
    `json.dumps(..., default=str)` would otherwise emit the Python
    `repr()` of each model (the bug caught by the Phase-1 smoke run).

    `with_result=True` includes actual/correct/brier; used by evaluate.
    """
    return {
        "prediction_id": pred.prediction_id,
        "match_id": pred.match_id,
        "created_at": pred.created_at,
        "pick": pred.pick,
        "probs": pred.final_probs,
        "confidence": pred.confidence,
        "rationale": pred.rationale,
        "reasoner_outputs": [
            ro.model_dump(mode="json")
            if hasattr(ro, "model_dump")
            else ro
            for ro in (pred.reasoner_outputs or [])
        ],
        "signals": {k: v.model_dump(mode="json") for k, v in pred.signals.items()},
        "warnings": pred.warnings,
        "model_versions": pred.model_versions,
        # Keep the result block under a `result` key for parity with the API.
        **(
            {
                "result": {
                    "actual": pred.actual,
                    "was_correct": pred.correct,
                    "result_brier": pred.brier,
                }
            }
            if with_result
            else {}
        ),
    }


if __name__ == "__main__":
    main()
