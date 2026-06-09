"""Tests for Task 19: the FastAPI surface.

We use FastAPI's TestClient (which spins uvicorn in-process) so the
tests exercise the full request/response cycle. No live socket.

Endpoints:
  GET  /health
  GET  /predictions?limit=N
  GET  /predictions/{match_id}
  GET  /metrics
  POST /predictions
  POST /predictions/{match_id}/result
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from soccer_agent.eval.dataset import EVAL_CASES
from soccer_agent.eval.fixture_factory import materialize_all
from soccer_agent.agent import _season_for


# ---------- fixtures ----------------------------------------------------------

def _seed(fx: Path) -> None:
    from soccer_agent.eval.fixture_factory import materialize_all
    materialize_all(EVAL_CASES, fx)


@pytest.fixture
def env(monkeypatch, tmp_path):
    fx = tmp_path / "fx"
    fx.mkdir()
    db = tmp_path / "agent.db"
    monkeypatch.setenv("SOCCER_AGENT_FIXTURES_DIR", str(fx))
    monkeypatch.setenv("SOCCER_AGENT_DB_PATH", str(db))
    _seed(fx)
    return fx, db


@pytest.fixture
def client(env):
    # build the app lazily so it picks up our env vars
    from soccer_agent.api.server import create_app
    return TestClient(create_app())


# ---------- tests -------------------------------------------------------------

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "db" in body
    assert body["db"] == "ok"


def test_list_predictions_empty(client):
    r = client.get("/predictions")
    assert r.status_code == 200
    assert r.json() == []


def test_list_predictions_after_create(client):
    # create one
    case = EVAL_CASES[0]
    payload = {
        "match_id": case.match_id,
        "home_id": case.home_id,
        "away_id": case.away_id,
        "venue_id": case.venue_id,
        "kickoff": case.kickoff.isoformat(),
        "competition": case.competition,
        "season": _season_for(case.kickoff),
    }
    r = client.post("/predictions", json=payload)
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["pick"] in ("home", "draw", "away")
    assert 0.0 <= created["confidence"] <= 1.0
    assert "rationale" in created
    # now list
    r2 = client.get("/predictions")
    assert r2.status_code == 200
    rows = r2.json()
    assert len(rows) == 1
    assert rows[0]["match_id"] == case.match_id


def test_get_prediction_by_match_id(client):
    case = EVAL_CASES[0]
    payload = {
        "match_id": case.match_id,
        "home_id": case.home_id,
        "away_id": case.away_id,
        "venue_id": case.venue_id,
        "kickoff": case.kickoff.isoformat(),
        "competition": case.competition,
        "season": _season_for(case.kickoff),
    }
    client.post("/predictions", json=payload)
    r = client.get(f"/predictions/{case.match_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["match_id"] == case.match_id
    assert "pick" in body
    assert "result" in body  # may be null


def test_get_prediction_missing_returns_404(client):
    r = client.get("/predictions/does_not_exist__vs__anyone__2024-01-01")
    assert r.status_code == 404


def test_post_result_evaluates_prediction(client, env):
    _, db_path = env
    case = EVAL_CASES[0]
    payload = {
        "match_id": case.match_id,
        "home_id": case.home_id,
        "away_id": case.away_id,
        "venue_id": case.venue_id,
        "kickoff": case.kickoff.isoformat(),
        "competition": case.competition,
        "season": _season_for(case.kickoff),
    }
    r = client.post("/predictions", json=payload)
    assert r.status_code == 201
    # record a result (we just use 2-1 home; doesn't matter for this test)
    rr = client.post(
        f"/predictions/{case.match_id}/result",
        json={"home_goals": 2, "away_goals": 1},
    )
    assert rr.status_code == 200, rr.text
    body = rr.json()
    assert body["result"]["home_goals"] == 2
    assert body["result"]["away_goals"] == 1
    assert "was_correct" in body["result"]
    # row updated in DB
    con = sqlite3.connect(str(db_path))
    row = con.execute(
        "SELECT home_goals, away_goals FROM results WHERE match_id = ?",
        (case.match_id,),
    ).fetchone()
    con.close()
    assert row == (2, 1)


def test_post_result_missing_match_returns_404(client):
    r = client.post(
        "/predictions/nonexistent__vs__foo__2024-01-01/result",
        json={"home_goals": 1, "away_goals": 0},
    )
    assert r.status_code == 404


def test_get_metrics_runs_eval(client, env, monkeypatch):
    """GET /metrics should run the harness over EVAL_CASES and return a summary."""
    _, db_path = env
    r = client.get("/metrics")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_total"] == len(EVAL_CASES)
    for key in ("accuracy", "brier_mean", "log_loss", "per_class", "calibration_ece"):
        assert key in body


def test_metrics_strict_json_no_nan_literals(client):
    """NaN values in the summary must surface as null in the JSON, not as
    a literal NaN (which is not valid JSON and breaks jq / dashboards)."""
    r = client.get("/metrics")
    assert r.status_code == 200
    # if NaN literals leaked, json.loads would error
    import json as _json
    parsed = _json.loads(r.text)
    # top_factor_hit_rate will be NaN on a fresh run
    assert parsed["top_factor_hit_rate"] is None
    # per_class.draw.f1 will be NaN (never predicted)
    assert parsed["per_class"]["draw"]["f1"] is None


def test_post_prediction_validation_error_on_missing_field(client):
    r = client.post("/predictions", json={"home_id": "x"})
    assert r.status_code == 422  # pydantic validation


# -- Task 33: raw vs calibrated surfaces in the API -----------------------


def test_list_predictions_includes_raw_and_calibrator(client, env, tmp_path):
    """The dashboard needs to show both raw and calibrated
    confidence. Pin that the API exposes them on every row.
    """
    from soccer_agent.db import Database
    _fx, db_path = env
    case = EVAL_CASES[0]
    pred_id = f"test-raw-cal-{case.match_id}"
    db = Database(str(db_path))
    db.insert_prediction({
        "prediction_id": pred_id,
        "match_id": case.match_id,
        "created_at": "2025-05-01T00:00:00",
        "signals": {},
        "reasoner_outputs": [],
        "model_versions": {"reasoner": "stub", "calibrator": "isotonic@UCL"},
        "raw_pick": "home",
        "raw_confidence": 0.73,
        "raw_probs": {"home": 0.73, "draw": 0.15, "away": 0.12},
        "final_pick": "home",
        "final_confidence": 0.62,
        "final_probs": {"home": 0.62, "draw": 0.22, "away": 0.16},
        "final_rationale": "test rationale",
        "warnings": [],
        "v": 2,
    })
    db.close()
    r = client.get("/predictions")
    assert r.status_code == 200
    rows = r.json()
    matching = [r for r in rows if r["match_id"] == case.match_id]
    assert matching, "inserted row missing from list"
    row = matching[0]
    # The calibrated number is exposed as `confidence` (public API)
    # but raw_confidence and calibrator come from the model_versions
    # block as a backwards-compat shim OR from the top-level columns
    # depending on which path wrote the row. We support both: accept
    # the field at top level OR inside model_versions.
    raw = row.get("raw_confidence")
    if raw is None:
        raw = row.get("model_versions", {}).get("raw_confidence")
    assert raw is not None, f"raw_confidence missing in row: {row.keys()}"
    assert abs(float(raw) - 0.73) < 1e-9
    # The calibrator label is exposed.
    cal = row.get("calibrator")
    if cal is None:
        cal = row.get("model_versions", {}).get("calibrator")
    assert cal == "isotonic@UCL", f"calibrator label missing/wrong: {cal}"


def test_calibration_status_aggregates(client, env):
    """/calibration/status returns mean delta and per-calibrator
    counts over predictions that have raw_confidence recorded."""
    from soccer_agent.db import Database
    _fx, db_path = env
    db = Database(str(db_path))
    # Two calibrated, one uncalibrated.
    for i, (raw, final, cal) in enumerate([
        (0.60, 0.55, "isotonic@UCL"),
        (0.40, 0.45, "isotonic@UCL"),
        (0.70, 0.70, None),
    ]):
        db.insert_prediction({
            "prediction_id": f"calstat-{i}",
            "match_id": f"calstat-match-{i}",
            "created_at": f"2025-05-0{i+1}T00:00:00",
            "signals": {}, "reasoner_outputs": [],
            "model_versions": {"calibrator": cal} if cal else {},
            "raw_pick": "home", "raw_confidence": raw,
            "raw_probs": {"home": raw, "draw": 0.1, "away": 0.1},
            "final_pick": "home", "final_confidence": final,
            "final_probs": {"home": final, "draw": 0.1, "away": 0.1},
            "final_rationale": "x", "warnings": [], "v": 2,
        })
    db.close()
    r = client.get("/calibration/status")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 3
    assert body["n_calibrated"] == 2
    # mean of -0.05, +0.05, 0.0 = 0.0
    assert abs(body["mean_delta"] - 0.0) < 1e-9
    # mean of abs(...) = (0.05 + 0.05 + 0.0) / 3 ≈ 0.0333
    assert abs(body["abs_mean_delta"] - (0.05 + 0.05 + 0.0) / 3) < 1e-9
    assert body["calibrators"] == {"isotonic@UCL": 2}


def test_calibration_status_empty(client):
    """When no predictions have raw_confidence recorded, the
    endpoint returns n=0 with nulls (not a 500)."""
    r = client.get("/calibration/status")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 0
    assert body["n_calibrated"] == 0
    assert body["mean_delta"] is None
    assert body["calibrators"] == {}


def test_api_dashboard_includes_calibration_monitor(client, env):
    """/api/dashboard must include a calibration_monitor block so
    the JS doesn't need a second fetch."""
    r = client.get("/api/dashboard")
    assert r.status_code == 200
    body = r.json()
    assert "calibration_monitor" in body
    cm = body["calibration_monitor"]
    # The four key fields the JS tile renders.
    assert set(cm.keys()) >= {
        "n_with_raw", "n_calibrated", "mean_delta", "abs_mean_delta", "calibrators"
    }
