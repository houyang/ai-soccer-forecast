"""PredictionAgent — the orchestrator.

Responsibilities:
  1. Given a Match, fan out to all registered tools in parallel.
  2. Assemble a MatchContext (Match + Signals + optional Venue).
  3. Run the primary reasoner (LLM by default, with NumericReasoner as
     fallback) and persist the ReasonerOutput.
  4. Blend reasoner outputs (if multiple are provided) and write a
     single Prediction row to the DB.
  5. After a match is played, evaluate the stored Prediction against
     the stored Result.

The agent never raises on tool failure — a failed tool is logged,
surfaced in warnings, and skipped. The agent does raise on:
  - Unknown match (no such prediction row to update)
  - No result yet (can't evaluate what hasn't happened)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from .db import Database
from .models import (
    Match,
    MatchContext,
    Prediction,
    ReasonerOutput,
    Signal,
    Venue,
)
from .reasoners import LLMReasoner, NumericReasoner
from .reasoners.base import Reasoner, normalize_probs
from .tools import ToolRegistry
from .eval.scoring import brier as brier_score, top_factor_hit
import os
from pathlib import Path as _Path
from typing import Any, Callable
from datetime import datetime as _datetime


def _season_for(kickoff: _datetime) -> str:
    """Render a season key in the form 'YYYY-YYYY'.

    The form tool's Pydantic default is '2024-2025', so the agent
    must use the same format to find fixtures. Returns a 4-digit
    next year for unambiguous lookup (e.g. '2024-2025' not '2024-25').
    Cross-year fixtures (e.g. Jan 2025) still belong to the
    2024-2025 season by football convention.
    """
    y = kickoff.year
    if kickoff.month >= 7:  # Jul-Dec → start year
        return f"{y}-{y + 1}"
    return f"{y - 1}-{y}"


def _top_factor_hit_for_prediction(row: dict[str, Any], actual: str) -> int | None:
    """Return 1/0/None for the best top_factor_hit across a prediction's reasoners.

    - 1 = at least one reasoner hit
    - 0 = all reasoners with factors missed
    - None = no reasoner emitted factors (we don't know)
    Stored as int (SQLite has no native bool) so the eval harness can SUM/AVG it.
    """
    raw_outputs = json.loads(row["reasoner_outputs"]) or []
    any_factors = False
    any_hit = False
    for ro in raw_outputs:
        try:
            out = ReasonerOutput.model_validate(ro)
        except Exception:  # noqa: BLE001
            continue
        if not out.factors:
            continue
        any_factors = True
        hit = top_factor_hit(out, actual)
        if hit is True:
            any_hit = True
            break
        if hit is None:
            # treat None per-reasoner as "unknown" — don't change any_factors
            pass
    if not any_factors:
        return None
    return 1 if any_hit else 0


# Tools to run for every prediction. Add more by passing a custom
# registry to the agent constructor.
DEFAULT_TOOLS = (
    "form_recent",
    "injury_news",
    "h2h_history",
    "weather_venue",
    "odds_market",
    "venue_info",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _signal_from_result(name: str, result) -> Signal:
    """Convert a ToolResult into a Signal stored on the MatchContext."""
    if result.ok and result.data is not None:
        data = result.data
        if hasattr(data, "model_dump"):
            data = data.model_dump(mode="json")
        return Signal(
            tool=name,
            ok=True,
            data=data,
            source=result.source,
        )
    return Signal(
        tool=name,
        ok=False,
        data={},
        source=result.source,
        error=result.error,
        warnings=[result.error] if result.error else [],
    )


def _blend(reasoner_outputs: list[ReasonerOutput], weights: list[float] | None = None) -> ReasonerOutput:
    """Weighted average of multiple ReasonerOutputs.

    The blended output's `reasoner` is "blend"; its pick is the argmax
    of the blended probs; confidence is the top-1 - top-2 margin.
    """
    if not reasoner_outputs:
        raise ValueError("cannot blend zero reasoner outputs")
    if len(reasoner_outputs) == 1:
        return reasoner_outputs[0]
    if weights is None:
        weights = [1.0] * len(reasoner_outputs)
    s = sum(weights)
    weights = [w / s for w in weights]

    blended = {"home": 0.0, "draw": 0.0, "away": 0.0}
    for ro, w in zip(reasoner_outputs, weights):
        for k in blended:
            blended[k] += w * ro.probs.get(k, 0.0)
    blended = normalize_probs(blended)
    pick = max(blended, key=blended.get)  # type: ignore[arg-type]
    sorted_p = sorted(blended.values(), reverse=True)
    confidence = max(0.0, min(1.0, sorted_p[0] - sorted_p[1]))

    rationales = [f"[{ro.reasoner}] {ro.rationale}" for ro in reasoner_outputs]
    return ReasonerOutput(
        reasoner="blend",
        pick=pick,  # type: ignore[arg-type]
        probs=blended,
        confidence=confidence,
        rationale="Blended view: " + " ".join(rationales),
        warnings=[w for ro in reasoner_outputs for w in ro.warnings],
    )


class PredictionAgent:
    def __init__(
        self,
        registry: ToolRegistry,
        reasoner: Reasoner | None = None,
        *,
        secondary_reasoner: Reasoner | None = None,
        blend_weights: tuple[float, float] = (0.5, 0.5),
        db_path: str | os.PathLike | None = None,
        tool_timeout_s: float = 10.0,
        elo_state_path: str | os.PathLike | None = None,
        # Task 31: calibrator wiring. If calibrator_key is None OR
        # no file exists for the competition, predict() runs in
        # uncalibrated mode (raw_confidence is stored but
        # final_confidence == raw_confidence).
        calibrator_root: str | os.PathLike | None = None,
        calibrator_key: str | None = "isotonic",
    ):
        self.registry = registry
        # Default: LLM (which falls back to numeric on error) with
        # numeric as a secondary for blending.
        self.reasoner = reasoner or LLMReasoner()
        self.secondary = secondary_reasoner or NumericReasoner()
        self.blend_weights = blend_weights
        self.db = Database(str(db_path)) if db_path is not None else Database()
        self.tool_timeout_s = tool_timeout_s
        # Load the Elo state from disk (or use a fresh in-memory one).
        # Falls back to env var SOCCER_AGENT_ELO_STATE for the path.
        from .elo import EloState  # local import to avoid cycles
        if elo_state_path is None:
            elo_state_path = os.environ.get("SOCCER_AGENT_ELO_STATE")
        if elo_state_path is not None and os.path.exists(elo_state_path):
            self.elo_state: Any = EloState.from_json(elo_state_path)
        else:
            self.elo_state = EloState()  # fresh, empty state
        # Calibrator store. Lazily populated per-competition in
        # predict(); the key (e.g. "isotonic") is fixed for the
        # lifetime of the agent, the root can be None to disable.
        from .calibration_store import load_calibrator
        self._calibrator_load = load_calibrator
        self._calibrator_root: _Path | None = (
            _Path(calibrator_root) if calibrator_root is not None else None
        )
        self._calibrator_key = calibrator_key
        # Cache: competition -> calibrator (or None if missing).
        # Avoids hitting the disk on every predict().
        self._calibrator_cache: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    async def predict(
        self,
        match: Match,
        *,
        tool_names: Iterable[str] = DEFAULT_TOOLS,
    ) -> Prediction:
        # 1. Fan out tools in parallel. return_exceptions=True so a single
        # tool bug doesn't break the whole gather; we surface all errors
        # below and convert each to a Signal.
        signals: dict[str, Signal] = {}
        tasks = []
        names = list(tool_names)
        for name in names:
            if name not in self.registry.names:
                # Unknown tool: skip payload build entirely and record a
                # warning on the Signal so the caller can see what was dropped.
                signals[name] = Signal(
                    tool=name, ok=False, data={}, source="fixture",
                    warnings=[f"tool {name!r} not registered"],
                )
                continue
            payload = self._build_tool_payload(name, match)
            tasks.append(self._run_tool(name, payload, signals))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for name, res in zip(names, results):
            if isinstance(res, Exception) and name not in signals:
                signals[name] = Signal(
                    tool=name, ok=False, data={}, source="fixture",
                    warnings=[f"unexpected error: {res!r}"],
                )

        # 2. Pull venue from the venue_info signal if present.
        venue = None
        v = signals.get("venue_info")
        if v and v.ok and isinstance(v.data, dict):
            try:
                venue = Venue.model_validate(v.data)
            except Exception:
                venue = None

        # 3. Assemble context. The agent's pre-loaded EloState
        # (if any) is passed through so the numeric reasoner can
        # use real per-team home/away ratings.
        ctx = MatchContext(
            match=match,
            signals=signals,
            venue=venue,
            elo_state=self.elo_state,
        )

        # 4. Run reasoners.
        ro_primary = self.reasoner.run(ctx)
        reasoner_outputs: list[ReasonerOutput] = [ro_primary]

        if self.secondary is not None and self.secondary is not self.reasoner:
            try:
                ro_secondary = self.secondary.run(ctx)
                reasoner_outputs.append(ro_secondary)
            except Exception as e:  # pragma: no cover - defensive
                ro_secondary = ReasonerOutput(
                    reasoner="secondary_error",
                    pick=ro_primary.pick,
                    probs=ro_primary.probs,
                    confidence=ro_primary.confidence,
                    rationale=f"secondary reasoner failed: {e}",
                    warnings=[f"secondary_error: {e}"],
                )
                reasoner_outputs.append(ro_secondary)

        # 5. Blend.
        if len(reasoner_outputs) == 1:
            blended = reasoner_outputs[0]
        else:
            blended = _blend(reasoner_outputs, list(self.blend_weights))

        # 6. Build the Prediction row.
        prediction_id = str(uuid.uuid4())
        created_at = _utc_now()
        # Surface tool-level warnings on the prediction record too.
        tool_warnings = [
            f"tool:{sig.tool}:{w}"
            for sig in signals.values()
            for w in (sig.warnings or [])
        ]
        all_warnings = (blended.warnings or []) + tool_warnings

        model_versions: dict[str, str] = {
            "reasoner": self.reasoner.name,  # type: ignore[attr-defined]
        }
        for n in names:
            if n in self.registry.names:
                model_versions[f"tool:{n}"] = getattr(
                    self.registry.get(n), "version", "0.0.0"
                )

        # Task 31: apply the calibrator (if one is fitted for this
        # competition) to the blended confidence. raw_confidence is
        # the un-calibrated top-1 margin; final_confidence is the
        # calibrated one. If no calibrator is available, both are
        # equal and the agent behaves as before.
        raw_confidence = blended.confidence
        final_confidence, calibrator_used = self._apply_calibrator(
            match.competition, raw_confidence,
        )
        if calibrator_used is not None:
            # Surface the calibrator in the model versions block so
            # the dashboard can show "calibrated by isotonic (EPL,
            # n=34, ECE=0.00)" next to the prediction row.
            # `calibrator_used` is already the right label —
            # either f"{key}@{competition}" for per-comp or
            # f"{key}@global" for the global fallback (Task 35).
            model_versions["calibrator"] = calibrator_used

        pred = Prediction(
            prediction_id=prediction_id,
            match_id=match.match_id,
            created_at=created_at,
            signals=signals,
            reasoner_outputs=[ro.model_dump() for ro in reasoner_outputs],
            final_pick=blended.pick,  # type: ignore[arg-type]
            final_probs=blended.probs,
            final_confidence=final_confidence,
            final_rationale=blended.rationale,
            warnings=all_warnings,
            model_versions=model_versions,
            raw_confidence=raw_confidence,
            calibrator=calibrator_used,
        )

        # 7. Persist.
        self.db.insert_prediction({
            "prediction_id": pred.prediction_id,
            "match_id": pred.match_id,
            "created_at": pred.created_at,
            "signals": {k: v.model_dump() for k, v in pred.signals.items()},
            "reasoner_outputs": [ro.model_dump() for ro in pred.reasoner_outputs],
            "final_pick": pred.final_pick,
            "final_probs": pred.final_probs,
            "final_confidence": pred.final_confidence,
            "raw_confidence": pred.raw_confidence,
            "final_rationale": pred.final_rationale,
            "warnings": pred.warnings,
            "model_versions": pred.model_versions,
        })
        return pred

    # ------------------------------------------------------------------
    # calibrator lookup
    # ------------------------------------------------------------------

    def _apply_calibrator(
        self, competition: str, raw_confidence: float,
    ) -> tuple[float, str | None]:
        """Return (calibrated_confidence, calibrator_label_or_None).

        Lookup order (Task 35):
            1. isotonic_<COMP>   — per-competition calibrator
            2. isotonic          — global fallback

        If neither file exists, return the (clamped) input unchanged
        with label None. The 0.85 cap from Task 32 is applied
        uniformly to keep the calibrator's input range sane.
        """
        if self._calibrator_root is None or self._calibrator_key is None:
            # 0.85 cap still applies so callers can't see >0.85
            # confidence even when calibration is off.
            return min(0.85, raw_confidence), None

        # 0.85 cap is applied before any calibration call.
        clamped = min(0.85, raw_confidence)

        cal, scope = self._load_calibrator_with_fallback(competition)
        if cal is None:
            return clamped, None

        out = cal.calibrate([clamped])
        calibrated = float(out[0]) if out else clamped
        # Isotonic can over/under-shoot at the edges of the fitted
        # range; clamp to [0, 1] defensively.
        calibrated = max(0.0, min(1.0, calibrated))
        if scope == "per_competition":
            label = f"{self._calibrator_key}@{competition}"
        else:
            label = f"{self._calibrator_key}@global"
        return calibrated, label

    def _load_calibrator_with_fallback(
        self, competition: str,
    ) -> tuple[Any, str | None]:
        """Cache-aware lookup: per-comp first, then global.

        Returns (calibrator_or_None, scope) where scope is one of
        "per_competition" | "global" | None. None means no file
        was found and the caller should pass the input through.
        """
        if competition in self._calibrator_cache:
            return self._calibrator_cache[competition]
        # Try per-competition first.
        per_comp = self._calibrator_load(
            key=f"{self._calibrator_key}_{competition}",
            root=self._calibrator_root,
        )
        if per_comp is not None:
            result = (per_comp, "per_competition")
        else:
            glob = self._calibrator_load(
                key=self._calibrator_key, root=self._calibrator_root,
            )
            result = (glob, "global" if glob is not None else None)
        # Cache the full (cal, scope) tuple keyed by competition.
        self._calibrator_cache[competition] = result
        return result

    # ------------------------------------------------------------------
    # evaluate (single match) + evaluate_all (loop)
    # ------------------------------------------------------------------

    async def evaluate(self, match_id: str) -> Prediction:
        """Look up the most recent prediction for match_id, attach the result."""
        with self.db._connect() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE match_id = ? ORDER BY created_at DESC LIMIT 1",
                (match_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"no prediction found for match_id={match_id!r}")
            result = conn.execute(
                "SELECT * FROM results WHERE match_id = ?", (match_id,),
            ).fetchone()
        if result is None:
            raise RuntimeError(f"no result for match_id={match_id!r} — can't evaluate yet")

        # Determine actual outcome.
        hg, ag = int(result["home_goals"]), int(result["away_goals"])
        if hg > ag:
            actual = "home"
        elif hg < ag:
            actual = "away"
        else:
            actual = "draw"
        was_correct = 1 if (row["final_pick"] == actual) else 0

        # Brier score on the 3-way simplex.
        probs = json.loads(row["final_probs"])
        actual_vec = {"home": 1.0, "draw": 1.0, "away": 1.0}
        actual_vec[actual] = 0.0  # one-hot
        brier = sum((probs.get(k, 0.0) - actual_vec[k]) ** 2 for k in ("home", "draw", "away")) / 2.0

        # Was the top factor the one that moved the outcome? Cheap proxy: any factor
        # with sign matching the actual margin direction. Best-effort only.
        top_factor_hit = None  # filled in by the eval harness in Task 18

        self.db.insert_result({
            "match_id": match_id,
            "home_goals": hg,
            "away_goals": ag,
            "decided_at": result["decided_at"],
            "was_correct": was_correct,
            "brier": brier,
            "top_factor_hit": top_factor_hit,
        })

        # Re-fetch the prediction row joined with result.
        return self._load_prediction_with_result(match_id)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _build_tool_payload(self, tool_name: str, match: Match) -> dict[str, Any]:
        """Build the per-tool input dict for a given match."""
        if tool_name == "form_recent":
            return {
                "home_team_id": match.home.id,
                "away_team_id": match.away.id,
                "season": _season_for(match.kickoff),
                "n_matches": 5,
            }
        if tool_name == "injury_news":
            return {
                "home_team_id": match.home.id,
                "away_team_id": match.away.id,
                "kickoff_date": match.kickoff.date().isoformat(),
            }
        if tool_name == "h2h_history":
            return {
                "home_team_id": match.home.id,
                "away_team_id": match.away.id,
                "max_meetings": 10,
            }
        if tool_name == "weather_venue":
            return {
                "venue_id": match.venue_id,
                "date": match.kickoff.date().isoformat(),
                "is_dome": False,
            }
        if tool_name == "odds_market":
            return {
                "home_team_id": match.home.id,
                "away_team_id": match.away.id,
                "kickoff_date": match.kickoff.date().isoformat(),
            }
        if tool_name == "venue_info":
            return {"venue_id": match.venue_id}
        raise ValueError(f"unknown tool: {tool_name}")

    async def _run_tool(self, name: str, payload: dict[str, Any], signals: dict[str, Signal]) -> None:
        if name not in self.registry.names:
            signals[name] = Signal(
                tool=name, ok=False, data={}, source="missing",
                warnings=[f"tool {name!r} not registered"],
            )
            return
        t0 = datetime.now(timezone.utc)
        try:
            result = await self.registry.run(name, payload, timeout=self.tool_timeout_s)
        except Exception as e:  # never let a tool crash the agent
            duration = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
            self.db.log_tool_call(tool=name, ok=False, duration_ms=duration, error=str(e))
            signals[name] = Signal(
                tool=name, ok=False, data={}, source="fixture",
                warnings=[f"unexpected error: {e!r}"],
            )
            return
        duration = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
        self.db.log_tool_call(
            tool=name,
            ok=result.ok,
            duration_ms=duration,
            source=result.source,
            error=result.error if result.error else None,
        )
        signals[name] = _signal_from_result(name, result)

    def evaluate_all(
        self,
        scout: Any,
        *,
        since: datetime | None = None,
    ) -> int:
        """Score every prediction that has a fresh result, write back to DB.

        `scout` is a `ResultScout` (duck-typed: needs `fetch_new_results(since)`).
        Returns the number of predictions that were newly scored. A
        second call with the same scout returns 0 — the loop is
        idempotent because `insert_result` is INSERT OR REPLACE and we
        skip rows that already have a was_correct set.
        """
        from .eval.scout import ResultScout  # local import — keep agent light
        if not isinstance(scout, ResultScout):
            raise TypeError(f"scout must be a ResultScout, got {type(scout).__name__}")
        if since is None:
            since = datetime(1970, 1, 1, tzinfo=timezone.utc)
        results = scout.fetch_new_results(since=since)
        scored = 0
        for r in results:
            row = self.db.execute_one(
                "SELECT * FROM predictions WHERE match_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (r.match_id,),
            )
            if row is None:
                # No prediction for this match — skip quietly.
                continue
            already = self.db.execute_one(
                "SELECT 1 FROM results WHERE match_id = ?",
                (r.match_id,),
            )
            if already is not None:
                continue
            probs = json.loads(row["final_probs"])
            b = brier_score(probs, r.winner)
            tfh = _top_factor_hit_for_prediction(row, r.winner)
            was_correct = 1 if (row["final_pick"] == r.winner) else 0
            self.db.insert_result({
                "match_id": r.match_id,
                "home_goals": r.home_goals,
                "away_goals": r.away_goals,
                "decided_at": r.decided_at,
                "was_correct": was_correct,
                "brier": b,
                "top_factor_hit": tfh,
            })
            scored += 1
        return scored

    def _load_prediction_with_result(self, match_id: str) -> Prediction:
        with self.db._connect() as conn:
            row = conn.execute(
                """
                SELECT p.*, r.home_goals, r.away_goals, r.decided_at,
                       r.was_correct, r.brier AS result_brier
                FROM predictions p
                LEFT JOIN results r ON r.match_id = p.match_id
                WHERE p.match_id = ?
                ORDER BY p.created_at DESC
                LIMIT 1
                """,
                (match_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"no prediction found for match_id={match_id!r}")
        signals = {}
        for k, v in (json.loads(row["signals"]) or {}).items():
            signals[k] = Signal.model_validate(v)
        return Prediction(
            prediction_id=row["prediction_id"],
            match_id=row["match_id"],
            created_at=row["created_at"],
            signals=signals,
            reasoner_outputs=json.loads(row["reasoner_outputs"]) or [],
            final_pick=row["final_pick"],
            final_probs=json.loads(row["final_probs"]),
            final_confidence=float(row["final_confidence"]),
            final_rationale=row["final_rationale"],
            warnings=json.loads(row["warnings"]) or [],
            model_versions=json.loads(row["model_versions"]) or {},
            actual=(
                "home" if row["home_goals"] > row["away_goals"]
                else "away" if row["home_goals"] < row["away_goals"]
                else "draw"
            ) if row["home_goals"] is not None else None,
            correct=bool(row["was_correct"]) if row["was_correct"] is not None else None,
            brier=float(row["result_brier"]) if row["result_brier"] is not None else None,
        )
