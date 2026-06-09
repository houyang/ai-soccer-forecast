"""FastAPI surface (Task 19).

A thin HTTP wrapper around the existing agent, DB, and harness.
No business logic here — all of it lives in the agent/harness/metrics
modules and is re-used as-is. This module is responsible for:

  - request/response shapes (Pydantic models)
  - env-var driven config (DB path, fixtures dir)
  - JSON serialization of Pydantic rows (with NaN-scrubbing)

End-points:
  GET  /health
  GET  /predictions?limit=N
  GET  /predictions/{match_id}
  GET  /metrics
  POST /predictions
  POST /predictions/{match_id}/result
"""
from __future__ import annotations

import json as jsonlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from soccer_agent.agent import PredictionAgent
from soccer_agent.db import Database
from soccer_agent.eval.harness import EvalHarness, _scrub_nans
from soccer_agent.models import Match, Team
from soccer_agent.reasoners import NumericReasoner
from soccer_agent.tools import default_registry


# ---------- request/response schemas ------------------------------------------

class CreatePredictionRequest(BaseModel):
    home_id: str
    away_id: str
    venue_id: str
    kickoff: str  # ISO datetime string
    competition: str = "UCL"
    season: str = "2025/26"
    round: Optional[str] = None
    match_id: Optional[str] = None
    tool_names: Optional[str] = None  # comma-separated; default = all


class RecordResultRequest(BaseModel):
    home_goals: int = Field(..., ge=0)
    away_goals: int = Field(..., ge=0)


# ---------- helpers -----------------------------------------------------------

def _db_path() -> Path:
    return Path(os.environ.get("SOCCER_AGENT_DB_PATH", "data/soccer_agent.db"))


def _fixtures_dir() -> Path:
    return Path(os.environ.get("SOCCER_AGENT_FIXTURES_DIR", "data/fixtures"))


def _static_dir() -> Path:
    """Path to the dashboard static assets (index.html, app.js, style.css).

    Sibling of server.py, inside src/soccer_agent/api/static/. Overridable
    via SOCCER_AGENT_STATIC_DIR for testability.
    """
    override = os.environ.get("SOCCER_AGENT_STATIC_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent / "static"


def _build_agent() -> PredictionAgent:
    return PredictionAgent(
        registry=default_registry(),
        reasoner=NumericReasoner(),
        db_path=_db_path(),
    )


def _synth_match_id(payload: CreatePredictionRequest) -> str:
    """Same convention as the CLI: home_vs_away__YYYY-MM-DD."""
    if payload.match_id:
        return payload.match_id
    date_part = payload.kickoff.split("T", 1)[0]
    return f"{payload.home_id}_vs_{payload.away_id}__{date_part}"


def _prediction_row_to_dict(row: Any) -> dict:
    """Convert a sqlite3.Row from list_predictions to a JSON-friendly dict.
    Renames the internal DB columns (`final_pick`, `final_confidence`,
    `final_rationale`) to the public API contract (`pick`, `confidence`,
    `rationale`). Result fields are nullable; preserve that.
    Also nest the flat result fields into a `result` object so the
    response shape is clearer.
    """
    out = dict(row)
    out["pick"] = out.pop("final_pick", None)
    out["confidence"] = out.pop("final_confidence", None)
    out["rationale"] = out.pop("final_rationale", None)
    # Pull result fields into a nested object, renaming the DB
    # columns to the public API contract.
    result_renames = {
        "home_goals": "home_goals",
        "away_goals": "away_goals",
        "decided_at": "decided_at",
        "was_correct": "was_correct",
        "result_brier": "brier",  # DB column is result_brier
        "top_factor_hit": "top_factor_hit",
    }
    result = {}
    for db_key, public_key in result_renames.items():
        if db_key in out:
            result[public_key] = out.pop(db_key)
    out["result"] = result if result else None
    # NaN -> null (per the strict-JSON contract)
    return _scrub_nans(out)


# ---------- factory -----------------------------------------------------------

def create_app() -> FastAPI:
    """Build a fresh FastAPI app. Called per-TestClient to pick up env vars."""
    app = FastAPI(
        title="soccer-agent",
        version="0.1.0",
        description="Multi-tool football match prediction agent — Phase 1 surface.",
    )

    # ---- health ----
    @app.get("/health")
    def health() -> dict:
        # check DB connectivity
        db_status = "ok"
        try:
            db = Database(_db_path())
            db.list_predictions(limit=1)
        except Exception as e:  # noqa: BLE001
            db_status = f"error: {type(e).__name__}: {e}"
        return {"status": "ok" if db_status == "ok" else "degraded",
                "db": db_status,
                "fixtures_dir": str(_fixtures_dir()),
                "db_path": str(_db_path())}

    # ---- list predictions ----
    @app.get("/predictions")
    def list_predictions(limit: int = 50) -> list[dict]:
        db = Database(_db_path())
        return [_prediction_row_to_dict(r) for r in db.list_predictions(limit=limit)]

    # ---- get one ----
    @app.get("/predictions/{match_id}")
    def get_prediction(match_id: str) -> dict:
        db = Database(_db_path())
        # SQLite stores match_ids; do an O(N) scan capped by the limit.
        for r in db.list_predictions(limit=10_000):
            if r["match_id"] == match_id:
                return _prediction_row_to_dict(r)
        raise HTTPException(status_code=404, detail=f"no prediction for match_id={match_id!r}")

    # ---- create prediction ----
    @app.post("/predictions", status_code=201)
    def create_prediction(req: CreatePredictionRequest) -> dict:
        match_id = _synth_match_id(req)
        # Match requires nested Team objects; build them from the IDs
        # using the same convention as the CLI (title-case the id).
        match = Match(
            match_id=match_id,
            home=Team(id=req.home_id, name=req.home_id.replace("_", " ").title()),
            away=Team(id=req.away_id, name=req.away_id.replace("_", " ").title()),
            kickoff=datetime.fromisoformat(req.kickoff),
            venue_id=req.venue_id,
            competition=req.competition,
            season=req.season,
            round=req.round,
        )
        agent = _build_agent()
        try:
            asyncio_run(agent.predict(match))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"predict failed: {type(e).__name__}: {e}")
        return _prediction_row_to_dict(_load_row(agent, match_id))

    # ---- Task 33: calibration monitor ---------------------------------
    @app.get("/calibration/status")
    def calibration_status() -> dict:
        """Aggregate stats over predictions with a recorded raw vs
        final confidence. Powers the dashboard's calibration tile.
        """
        import json as jsonlib
        from soccer_agent.db import Database
        db = Database(_db_path())
        rows = db.execute(
            "SELECT raw_confidence, final_confidence, model_versions "
            "FROM predictions WHERE raw_confidence IS NOT NULL"
        )
        n = len(rows)
        if n == 0:
            return {
                "n": 0, "n_calibrated": 0,
                "mean_delta": None, "abs_mean_delta": None,
                "calibrators": {},
            }
        deltas: list[float] = []
        cal_counts: dict[str, int] = {}
        n_calibrated = 0
        for r in rows:
            raw = r["raw_confidence"]
            final = r["final_confidence"]
            if raw is None or final is None:
                continue
            deltas.append(final - raw)
            try:
                mv = jsonlib.loads(r["model_versions"]) if r["model_versions"] else {}
            except (TypeError, ValueError):
                mv = {}
            cal = mv.get("calibrator")
            if cal:
                n_calibrated += 1
                cal_counts[cal] = cal_counts.get(cal, 0) + 1
        mean_d = sum(deltas) / len(deltas) if deltas else None
        abs_mean = (
            sum(abs(d) for d in deltas) / len(deltas) if deltas else None
        )
        return {
            "n": n,
            "n_calibrated": n_calibrated,
            "mean_delta": mean_d,
            "abs_mean_delta": abs_mean,
            "calibrators": cal_counts,
        }

    # ---- record result ----
    @app.post("/predictions/{match_id}/result")
    def record_result(match_id: str, req: RecordResultRequest) -> dict:
        agent = _build_agent()
        # Verify the prediction exists; agent.evaluate() raises KeyError
        # if it doesn't, but we want a clean 404 instead of a 500.
        existing = None
        for r in agent.db.list_predictions(limit=10_000):
            if r["match_id"] == match_id:
                existing = r
                break
        if existing is None:
            raise HTTPException(status_code=404, detail=f"no prediction for match_id={match_id!r}")
        # Persist the result first (agent.evaluate() expects it to exist).
        agent.db.insert_result({
            "match_id": match_id,
            "home_goals": req.home_goals,
            "away_goals": req.away_goals,
            "decided_at": datetime.now(timezone.utc).isoformat(),
            "was_correct": None,   # agent.evaluate() fills these in
            "brier": None,
            "top_factor_hit": None,
        })
        try:
            pred = asyncio_run(agent.evaluate(match_id))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"evaluate failed: {type(e).__name__}: {e}")
        # Re-fetch via the list path so the shape matches GET /predictions/{id}.
        for r in agent.db.list_predictions(limit=10_000):
            if r["match_id"] == match_id:
                return _prediction_row_to_dict(r)
        # shouldn't reach here
        return _scrub_nans(pred.model_dump(mode="json"))

    # ---- metrics ----
    @app.get("/metrics")
    def metrics() -> dict:
        h = EvalHarness(
            fixtures_dir=_fixtures_dir(),
            db_path=_db_path(),
        )
        # `run()` will be idempotent: re-uses existing predictions.
        return h.run()

    # ---- dashboard (Task 29) ----
    @app.get("/api/dashboard")
    def dashboard(limit: int = 50) -> dict:
        """Single-shot payload for the static dashboard page.

        Composes three existing sources:
          - EvalHarness.run()     -> summary metrics
          - db.list_predictions() -> recent predictions (with results)
          - run_calibration_report() -> reliability data

        This is the *only* endpoint the page polls. Caching / static
        hosting can sit in front of it later without changing the
        page.
        """
        from datetime import datetime, timezone
        from soccer_agent.eval.calibration import run_calibration_report

        db = Database(_db_path())
        # Predictions (descending by created_at, capped).
        rows = db.list_predictions(limit=limit)
        predictions = [_prediction_row_to_dict(r) for r in rows]

        # Summary metrics. Reuse the existing /metrics path so we
        # stay consistent with what users see in CLI/eval output.
        # If the DB is empty, /metrics will return zeros — that's
        # the right shape for an empty dashboard.
        try:
            h = EvalHarness(
                fixtures_dir=_fixtures_dir(),
                db_path=_db_path(),
            )
            metrics = h.run()
        except Exception:  # noqa: BLE001
            # If the harness can't run (no fixtures, etc.) fall back
            # to a minimal summary so the page still renders.
            metrics = {
                "n": 0, "n_resolved": 0,
                "accuracy": None, "brier": None, "log_loss": None,
            }
        # The harness returns `n_total`, `brier_mean`, etc. — map
        # those into the public dashboard shape.
        summary = {
            "n_predictions": metrics.get("n_total", 0),
            "n_resolved": metrics.get("n_resolved", 0),
            "accuracy": metrics.get("accuracy"),
            "brier": metrics.get("brier_mean"),
            "log_loss": metrics.get("log_loss"),
            "calibration_ece": metrics.get("calibration_ece"),
        }

        # Calibration. Reuses the same code path as the CLI
        # `python -m soccer_agent.eval.calibration`. Empty DB is
        # handled inside run_calibration_report (returns n=0).
        calib = run_calibration_report(_db_path()).to_dict()

        # Task 33: include the live calibration monitor (mean delta
        # since raw→final was added in Task 31). Computed inline
        # here so the page only fetches /api/dashboard.
        raw_rows = db.execute(
            "SELECT raw_confidence, final_confidence, model_versions "
            "FROM predictions WHERE raw_confidence IS NOT NULL"
        )
        n_raw = len(raw_rows)
        n_cal = 0
        cal_counts: dict[str, int] = {}
        deltas: list[float] = []
        for r in raw_rows:
            if r["raw_confidence"] is None or r["final_confidence"] is None:
                continue
            deltas.append(r["final_confidence"] - r["raw_confidence"])
            try:
                mv = jsonlib.loads(r["model_versions"]) if r["model_versions"] else {}
            except (TypeError, ValueError):
                mv = {}
            cal = mv.get("calibrator")
            if cal:
                n_cal += 1
                cal_counts[cal] = cal_counts.get(cal, 0) + 1
        cal_monitor = {
            "n_with_raw": n_raw,
            "n_calibrated": n_cal,
            "mean_delta": (sum(deltas) / len(deltas)) if deltas else None,
            "abs_mean_delta": (
                sum(abs(d) for d in deltas) / len(deltas) if deltas else None
            ),
            "calibrators": cal_counts,
        }

        return {
            "summary": summary,
            "predictions": predictions,
            "calibration": calib,
            "calibration_monitor": cal_monitor,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ---- static dashboard assets (Task 29) ----
    # Mounted last so the API routes above take precedence.
    static_path = _static_dir()
    if static_path.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(static_path), html=True),
            name="dashboard",
        )
    else:
        # The dir is missing — log loudly so a fresh checkout doesn't
        # silently serve a 404 from the root. The API still works.
        import sys
        print(
            f"[soccer_agent.api] dashboard static dir not found at "
            f"{static_path}; GET / will return 404. Run "
            f"`scripts/serve_dashboard.sh` from a fresh checkout once "
            f"the static assets are restored.",
            file=sys.stderr,
        )

    return app


# ---------- sync-async bridge + row reload -----------------------------------

def asyncio_run(coro):
    """Run an async coroutine synchronously.

    The agent's predict() is async (it gathers tool signals concurrently).
    FastAPI's request handlers are sync by default; we bridge here.
    """
    import asyncio
    return asyncio.run(coro)


# ---------- entry point (uvicorn) --------------------------------------------

# A module-level `app` so `uvicorn soccer_agent.api.server:app` works.
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    host = os.environ.get("SOCCER_AGENT_HOST", "127.0.0.1")
    port = int(os.environ.get("SOCCER_AGENT_PORT", "8000"))
    uvicorn.run("soccer_agent.api.server:app", host=host, port=port, reload=False)


def _load_row(agent: PredictionAgent, match_id: str) -> dict:
    """Re-read the just-saved prediction as a sqlite3.Row for serialization."""
    # We use the agent's own Database handle (already opened to the right
    # path) rather than re-opening, to keep the response shape consistent
    # with list_predictions.
    for r in agent.db.list_predictions(limit=10_000):
        if r["match_id"] == match_id:
            return dict(r)
    # fall back to model dump (shouldn't happen if insert succeeded)
    raise RuntimeError(f"prediction {match_id!r} not found in DB after predict()")
