"""Tests for soccer_agent.db."""

import sqlite3

import pytest

from soccer_agent.db import Database, get_db, init_db


def test_init_db_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    init_db(str(db_path))
    init_db(str(db_path))  # second call should not raise
    with sqlite3.connect(str(db_path)) as conn:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"predictions", "results", "eval_runs", "tool_calls"} <= tables


def test_database_context_manager(tmp_path):
    db_path = tmp_path / "test.db"
    with Database(str(db_path)) as db:
        rows = db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        names = {r["name"] for r in rows}
        assert "predictions" in names


def test_insert_and_fetch_prediction(tmp_path):
    db_path = tmp_path / "test.db"
    payload = {
        "prediction_id": "p1",
        "match_id": "m1",
        "created_at": "2026-05-30T10:00:00Z",
        "signals": {"form_recent": {"tool": "form_recent", "data": {}, "source": "fixture"}},
        "reasoner_outputs": [],
        "final_pick": "home",
        "final_probs": {"home": 0.6, "draw": 0.2, "away": 0.2},
        "final_confidence": 0.6,
        "final_rationale": "home favourite",
        "warnings": [],
        "model_versions": {"numeric": "v0.1"},
    }
    with Database(str(db_path)) as db:
        db.insert_prediction(payload)
        row = db.get_prediction("p1")
    assert row is not None
    assert row["match_id"] == "m1"
    assert row["final_pick"] == "home"


def test_insert_result_is_idempotent_per_match(tmp_path):
    db_path = tmp_path / "test.db"
    result = {
        "match_id": "m1",
        "home_goals": 2,
        "away_goals": 1,
        "decided_at": "2026-05-30T22:00:00Z",
        "was_correct": 1,
        "brier": 0.32,
        "top_factor_hit": 1,
    }
    with Database(str(db_path)) as db:
        db.insert_result(result)
        db.insert_result(result)  # should replace, not duplicate
        rows = db.execute("SELECT * FROM results WHERE match_id = ?", ("m1",))
    assert len(rows) == 1


def test_list_predictions_joins_results(tmp_path):
    db_path = tmp_path / "test.db"
    with Database(str(db_path)) as db:
        db.insert_prediction({
            "prediction_id": "p1", "match_id": "m1",
            "created_at": "2026-05-30T10:00:00Z",
            "signals": {}, "reasoner_outputs": [],
            "final_pick": "home",
            "final_probs": {"home": 0.6, "draw": 0.2, "away": 0.2},
            "final_confidence": 0.6, "final_rationale": "x",
            "warnings": [], "model_versions": {},
        })
        db.insert_result({
            "match_id": "m1", "home_goals": 1, "away_goals": 1,
            "decided_at": "2026-05-30T22:00:00Z",
            "was_correct": 0, "brier": 0.4, "top_factor_hit": 0,
        })
        rows = db.list_predictions(limit=10)
    assert len(rows) == 1
    assert rows[0]["home_goals"] == 1


def test_get_db_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("SOCCER_AGENT_DB_PATH", str(tmp_path / "from_env.db"))
    db = get_db()
    try:
        assert db.db_path == str(tmp_path / "from_env.db")
    finally:
        db.close()


# --- Task 31: raw_confidence column --------------------------------------


def test_raw_confidence_round_trips(tmp_path):
    """raw_confidence is optional; when supplied, it's stored alongside
    final_confidence so we can show the user the pre/post-calibration
    pair in the dashboard."""
    db_path = tmp_path / "raw.db"
    payload = {
        "prediction_id": "p_raw",
        "match_id": "m_raw",
        "created_at": "2026-05-30T10:00:00Z",
        "signals": {},
        "reasoner_outputs": [],
        "final_pick": "home",
        "final_probs": {"home": 0.6, "draw": 0.2, "away": 0.2},
        "final_confidence": 0.5,  # calibrated
        "raw_confidence": 0.85,   # uncalibrated
        "final_rationale": "x",
        "warnings": [],
        "model_versions": {},
    }
    with Database(str(db_path)) as db:
        db.insert_prediction(payload)
        row = db.get_prediction("p_raw")
    assert row is not None
    # raw_confidence lives on the dict; the row helper passes it through.
    rc = row.get("raw_confidence")
    assert rc is not None and abs(rc - 0.85) < 1e-9, f"raw_confidence lost: {rc!r}"


def test_raw_confidence_optional(tmp_path):
    """Pre-31 rows (and predictions made before a calibrator exists)
    pass without raw_confidence. INSERT should succeed and the
    column should be NULL."""
    db_path = tmp_path / "opt.db"
    payload = {
        "prediction_id": "p_opt",
        "match_id": "m_opt",
        "created_at": "2026-05-30T10:00:00Z",
        "signals": {},
        "reasoner_outputs": [],
        "final_pick": "draw",
        "final_probs": {"home": 0.33, "draw": 0.34, "away": 0.33},
        "final_confidence": 0.4,
        # raw_confidence omitted on purpose
        "final_rationale": "x",
        "warnings": [],
        "model_versions": {},
    }
    with Database(str(db_path)) as db:
        db.insert_prediction(payload)
        row = db.get_prediction("p_opt")
    assert row is not None
    assert row.get("raw_confidence") is None


def test_add_column_if_missing_is_idempotent(tmp_path):
    """The migration runs on every Database() construction. It must
    be a no-op when the column already exists (no exception)."""
    db_path = tmp_path / "mig.db"
    with Database(str(db_path)) as db:
        pass  # first construction adds the column
    # Second construction must not raise.
    with Database(str(db_path)) as db:
        rows = db.execute(
            "PRAGMA table_info(predictions)"
        )
        names = {r["name"] for r in rows}
    assert "raw_confidence" in names
