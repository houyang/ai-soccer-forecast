"""SQLite persistence layer.

Plain stdlib sqlite3. JSON columns for structured data. Idempotent
schema migrations via CREATE TABLE IF NOT EXISTS.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Any, Iterator

from .config import get_settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    match_id      TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    signals       TEXT NOT NULL DEFAULT '{}',
    reasoner_outputs TEXT NOT NULL DEFAULT '[]',
    final_pick    TEXT NOT NULL,
    final_probs   TEXT NOT NULL,
    final_confidence REAL NOT NULL,
    raw_confidence REAL,  -- added in Task 31: the un-calibrated top-1
                          -- margin. NULL for pre-migration rows.
    final_rationale TEXT NOT NULL,
    warnings      TEXT NOT NULL DEFAULT '[]',
    model_versions TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_predictions_match_id ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions(created_at);

CREATE TABLE IF NOT EXISTS results (
    match_id      TEXT PRIMARY KEY,
    home_goals    INTEGER NOT NULL,
    away_goals    INTEGER NOT NULL,
    decided_at    TEXT NOT NULL,
    was_correct   INTEGER,
    brier         REAL,
    top_factor_hit INTEGER
);

CREATE TABLE IF NOT EXISTS eval_runs (
    eval_id       TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    dataset_path  TEXT NOT NULL,
    n_matches     INTEGER NOT NULL,
    n_with_results INTEGER NOT NULL,
    metrics       TEXT NOT NULL,
    judge_score   REAL,
    config        TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT,
    tool          TEXT NOT NULL,
    called_at     TEXT NOT NULL,
    duration_ms   INTEGER,
    ok            INTEGER NOT NULL,
    source        TEXT,
    error         TEXT
);

CREATE INDEX IF NOT EXISTS idx_tool_calls_prediction_id ON tool_calls(prediction_id);
"""


class Database:
    """Thin wrapper over sqlite3 with JSON helpers and a context manager."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = get_settings().db_path
        self.db_path = db_path
        # ensure parent dir
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # initial migration
        self._connect().executescript(SCHEMA)
        self._connect().commit()
        # task 31: ensure raw_confidence column exists for DBs created
        # before that change. Idempotent — does nothing if the
        # column is already there.
        self._add_column_if_missing(
            "predictions", "raw_confidence", "REAL"
        )

    def _add_column_if_missing(
        self, table: str, column: str, col_type: str
    ) -> None:
        """Idempotent ALTER TABLE for new columns.

        SQLite has no ADD COLUMN IF NOT EXISTS, so we read
        PRAGMA table_info and only alter when the column is absent.
        Used by the Task 31 migration (raw_confidence).
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                f"PRAGMA table_info({table})"
            ).fetchall()
            names = {r["name"] for r in rows}
            if column not in names:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {col_type}"
                )
                conn.commit()
        finally:
            conn.close()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def execute(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._tx() as conn:
            cur = conn.execute(sql, params)
            return cur.fetchall()

    def execute_one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        rows = self.execute(sql, params)
        return rows[0] if rows else None

    # -- predictions --------------------------------------------------------

    def insert_prediction(self, payload: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO predictions
                  (prediction_id, match_id, created_at, signals, reasoner_outputs,
                   final_pick, final_probs, final_confidence, raw_confidence,
                   final_rationale, warnings, model_versions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["prediction_id"],
                    payload["match_id"],
                    payload["created_at"],
                    json.dumps(payload.get("signals", {})),
                    json.dumps(payload.get("reasoner_outputs", [])),
                    payload["final_pick"],
                    json.dumps(payload.get("final_probs", {})),
                    float(payload["final_confidence"]),
                    # raw_confidence is optional — None for pre-31 rows
                    # and for predictions made before a calibrator
                    # existed. We tolerate both.
                    (
                        float(payload["raw_confidence"])
                        if payload.get("raw_confidence") is not None
                        else None
                    ),
                    payload["final_rationale"],
                    json.dumps(payload.get("warnings", [])),
                    json.dumps(payload.get("model_versions", {})),
                ),
            )

    def get_prediction(self, prediction_id: str) -> dict[str, Any] | None:
        row = self.execute_one("SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,))
        return _row_to_prediction(row) if row else None

    def list_predictions(self, limit: int = 50) -> list[dict[str, Any]]:
        sql = """
        SELECT p.*, r.home_goals, r.away_goals, r.decided_at,
               r.was_correct, r.brier AS result_brier, r.top_factor_hit
        FROM predictions p
        LEFT JOIN results r ON r.match_id = p.match_id
        ORDER BY p.created_at DESC
        LIMIT ?
        """
        return [_row_to_prediction(r, with_result=True) for r in self.execute(sql, (limit,))]

    # -- results ------------------------------------------------------------

    def insert_result(self, payload: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO results
                  (match_id, home_goals, away_goals, decided_at, was_correct, brier, top_factor_hit)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["match_id"],
                    int(payload["home_goals"]),
                    int(payload["away_goals"]),
                    payload["decided_at"],
                    payload.get("was_correct"),
                    payload.get("brier"),
                    payload.get("top_factor_hit"),
                ),
            )

    # -- eval runs ----------------------------------------------------------

    def insert_eval_run(self, payload: dict[str, Any]) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO eval_runs
                  (eval_id, started_at, finished_at, dataset_path, n_matches,
                   n_with_results, metrics, judge_score, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["eval_id"],
                    payload["started_at"],
                    payload.get("finished_at"),
                    payload["dataset_path"],
                    int(payload["n_matches"]),
                    int(payload["n_with_results"]),
                    json.dumps(payload.get("metrics", {})),
                    payload.get("judge_score"),
                    json.dumps(payload.get("config", {})),
                ),
            )

    def list_eval_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.execute("SELECT * FROM eval_runs ORDER BY started_at DESC LIMIT ?", (limit,))
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["metrics"] = json.loads(d["metrics"])
            except (TypeError, ValueError):
                pass
            try:
                d["config"] = json.loads(d["config"])
            except (TypeError, ValueError):
                pass
            out.append(d)
        return out

    # -- tool call logging --------------------------------------------------

    def log_tool_call(
        self,
        *,
        tool: str,
        ok: bool,
        duration_ms: int,
        source: str | None = None,
        error: str | None = None,
        prediction_id: str | None = None,
        called_at: str | None = None,
    ) -> None:
        from datetime import datetime, timezone
        called_at = called_at or datetime.now(timezone.utc).isoformat()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO tool_calls
                  (prediction_id, tool, called_at, duration_ms, ok, source, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (prediction_id, tool, called_at, duration_ms, 1 if ok else 0, source, error),
            )

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        """No persistent connection; provided for symmetry / get_db factory cleanup."""
        return None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *_args) -> None:
        return None


def init_db(db_path: str | None = None) -> None:
    Database(db_path)


def get_db() -> Database:
    """Factory used by FastAPI and CLI."""
    return Database()


def _row_to_prediction(row: sqlite3.Row, with_result: bool = False) -> dict[str, Any]:
    d = dict(row)
    for key in ("signals", "reasoner_outputs", "final_probs", "warnings", "model_versions"):
        raw = d.get(key)
        if raw is None:
            continue
        if isinstance(raw, (dict, list)):
            continue
        try:
            d[key] = json.loads(raw)
        except (TypeError, ValueError):
            d[key] = {}
    if not with_result:
        for k in ("home_goals", "away_goals", "decided_at", "was_correct", "result_brier", "top_factor_hit"):
            d.pop(k, None)
    return d
