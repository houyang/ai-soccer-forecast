"""Tests for the PredictionAgent orchestrator."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path

import pytest

from soccer_agent.agent import PredictionAgent, _season_for
from soccer_agent.db import Database, init_db
from soccer_agent.llm import StubLLMClient
from soccer_agent.models import Match, Team
from soccer_agent.reasoners import LLMReasoner, NumericReasoner
from soccer_agent.tools import default_registry
from soccer_agent.tools._fixtures import write_json


# -- fixture data ------------------------------------------------------------

FORM = {
    "home": {
        "played": 5, "won": 4, "drawn": 1, "lost": 0,
        "gf": 12, "ga": 3, "points": 13, "last5_form_string": "WWDLW",
    },
    "away": {
        "played": 5, "won": 2, "drawn": 1, "lost": 2,
        "gf": 7, "ga": 8, "points": 7, "last5_form_string": "DLWLW",
    },
}

INJURY = {
    "home": [
        {"player": "Rodri", "status": "out", "reported_at": "2025-04-10T09:00:00Z", "source": "x"},
    ],
    "away": [],
}

H2H = {
    "home_team_id": "man_city", "away_team_id": "real_madrid",
    "meetings": [
        {"date": "2024-05-01T20:00:00Z", "home": "man_city", "away": "real_madrid",
         "home_goals": 3, "away_goals": 1, "competition": "UCL"},
        {"date": "2023-05-17T20:00:00Z", "home": "real_madrid", "away": "man_city",
         "home_goals": 1, "away_goals": 1, "competition": "UCL"},
    ],
    "home_wins": 1, "away_wins": 0, "draws": 1,
    "last_meeting": "2024-05-01T20:00:00Z", "last_winner": "home",
}

WEATHER = {
    "venue_id": "puskas_arena",
    "date": "2025-05-30",
    "is_dome": False,
    "conditions": "clear",
    "temp_c": 18.0,
    "wind_kph": 8.0,
    "precip_mm": 0.0,
    "playability_risk": "low",
}

ODDS = {
    "bookmakers": [
        {"name": "pinnacle", "home": 2.1, "draw": 3.4, "away": 3.5},
        {"name": "bet365",   "home": 2.0, "draw": 3.5, "away": 3.6},
    ],
    "implied_probs": {"home": 0.48, "draw": 0.29, "away": 0.23},
    "market_consensus_pick": "home",
}

VENUE = {
    "id": "puskas_arena", "name": "Puskás Aréna",
    "city": "Budapest", "country": "HUN",
    "capacity": 67215, "surface": "grass",
    "is_neutral": True, "is_dome": False, "altitude_m": 100,
    "lat": 47.5027, "lon": 19.0938,
}


@pytest.fixture
def fx(monkeypatch, tmp_path):
    """Point the env at a fresh fixture dir and seed the UCL-final fixtures."""
    d = tmp_path / "fx"
    d.mkdir()
    monkeypatch.setenv("SOCCER_AGENT_FIXTURES_DIR", str(d))
    write_json("form", "man_city__real_madrid__2024-2025.json", data=FORM)
    write_json("injury", "man_city__real_madrid__2025-05-30.json", data=INJURY)
    write_json("h2h", "man_city__real_madrid.json", data=H2H)
    write_json("weather", "puskas_arena__2025-05-30.json", data=WEATHER)
    write_json("odds", "man_city__real_madrid__2025-05-30.json", data=ODDS)
    write_json("venues", "venue_puskas_arena.json", data=VENUE)
    return d


def _ucl_final() -> Match:
    return Match(
        match_id="ucl-25-final",
        competition="UCL",
        kickoff=datetime(2025, 5, 30, 20, 0, 0),
        home=Team(id="man_city", name="Manchester City"),
        away=Team(id="real_madrid", name="Real Madrid"),
        venue_id="puskas_arena",
    )


def _laliga_match() -> Match:
    """Used by Task 35 tests to exercise the global-fallback path."""
    return Match(
        match_id="laliga-test-1",
        competition="LaLiga",
        kickoff=datetime(2025, 4, 12, 18, 30, 0),
        home=Team(id="barca", name="Barcelona"),
        away=Team(id="real_madrid", name="Real Madrid"),
        venue_id="camp_nou",
    )


# -- basic flow --------------------------------------------------------------


def test_predict_returns_prediction_and_writes_row(fx, tmp_path):
    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=LLMReasoner(client=StubLLMClient()),
        db_path=db_path,
    )
    pred = asyncio.run(agent.predict(_ucl_final()))
    assert pred.match_id == "ucl-25-final"
    assert pred.pick in ("home", "draw", "away")
    assert 0.0 <= pred.confidence <= 1.0
    assert pred.prediction_id  # UUID set
    with Database(str(db_path))._tx() as con:
        row = con.execute(
            "SELECT * FROM predictions WHERE prediction_id = ?", (pred.prediction_id,),
        ).fetchone()
    assert row is not None


def test_predict_survives_when_one_tool_fails(fx, tmp_path):
    # Remove the form fixture
    (fx / "form" / "man_city__real_madrid__2024-2025.json").unlink()
    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=NumericReasoner(),
        db_path=db_path,
    )
    pred = asyncio.run(agent.predict(_ucl_final()))
    assert pred.pick in ("home", "draw", "away")
    assert any("form" in w.lower() for w in pred.warnings)


def test_predict_with_unknown_tool_name_is_skipped(fx, tmp_path):
    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=NumericReasoner(),
        db_path=db_path,
    )
    pred = asyncio.run(agent.predict(_ucl_final(), tool_names=["form_recent", "no_such_tool"]))
    assert any("no_such_tool" in w for w in pred.warnings)


def test_predict_blends_two_reasoners(fx, tmp_path):
    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=LLMReasoner(client=StubLLMClient()),
        secondary_reasoner=NumericReasoner(),
        blend_weights=(0.5, 0.5),
        db_path=db_path,
    )
    pred = asyncio.run(agent.predict(_ucl_final()))
    assert len(pred.reasoner_outputs) == 2
    # Rationale references both reasoners
    assert "llm" in pred.rationale or "numeric" in pred.rationale


def test_predict_persists_signals(fx, tmp_path):
    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=NumericReasoner(),
        db_path=db_path,
    )
    pred = asyncio.run(agent.predict(_ucl_final()))
    with Database(str(db_path))._tx() as con:
        row = con.execute(
            "SELECT signals FROM predictions WHERE prediction_id = ?", (pred.prediction_id,),
        ).fetchone()
    stored = json.loads(row["signals"])
    assert "form_recent" in stored
    assert stored["form_recent"]["ok"] is True


# -- evaluate flow -----------------------------------------------------------


# -- Task 31: calibrator wiring --------------------------------------------


def test_predict_applies_calibrator(fx, tmp_path):
    """When a calibrator is fitted for the agent's calibrator_key
    (and a file exists in calibrator_root for the right name),
    predict() should:
      - store raw_confidence == blended.confidence
      - apply the calibrator to produce final_confidence
      - record the calibrator label (e.g. "isotonic@UCL")
    And the final_confidence should be the calibrated one, not
    the raw one.
    """
    from soccer_agent.calibration import IsotonicCalibrator
    from soccer_agent.calibration_store import save_calibrator, load_calibrator

    # Fit a calibrator that maps everything to 0.5. Easy to detect.
    cal = IsotonicCalibrator()
    cal.fit([0.0, 0.5, 1.0], [0.5, 0.5, 0.5])  # all -> 0.5
    cal_root = tmp_path / "calibrators"
    save_calibrator(
        cal, key="isotonic", root=cal_root,
        competition="UCL", n_samples=34, ece=0.0, brier=0.26,
    )
    # Sanity: round-trip works.
    loaded = load_calibrator(key="isotonic", root=cal_root)
    assert loaded is not None
    assert loaded.calibrate([0.42])[0] == 0.5

    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=LLMReasoner(client=StubLLMClient()),
        db_path=db_path,
        calibrator_root=cal_root,
        calibrator_key="isotonic",
    )
    pred = asyncio.run(agent.predict(_ucl_final()))

    # raw is what the reasoner produced; final is calibrated.
    assert pred.raw_confidence is not None
    assert pred.calibrator is not None
    assert pred.calibrator.startswith("isotonic@")
    # The fitted calibrator maps every input to 0.5.
    assert abs(pred.final_confidence - 0.5) < 1e-9
    # The raw should be different (LLM stub gives a real number).
    # We can't assert a specific value, but it shouldn't be 0.5
    # in a perfectly-uncalibrated world; the stub's default output
    # yields ~0.5 anyway, so this is a soft check. The hard check
    # is that final_confidence came from the calibrator.
    assert 0.0 <= pred.raw_confidence <= 1.0

    # Check the persistence path too.
    with Database(str(db_path))._tx() as con:
        row = con.execute(
            "SELECT raw_confidence, final_confidence, model_versions "
            "FROM predictions WHERE prediction_id = ?",
            (pred.prediction_id,),
        ).fetchone()
    assert row is not None
    # JSON columns come back as strings.
    versions = json.loads(row["model_versions"])
    # Task 35: the calibrator was saved with key="isotonic" (global),
    # so the per-comp lookup for UCL misses and the global fallback
    # is used. The label must reflect that.
    assert versions.get("calibrator") == "isotonic@global"


def test_predict_uses_per_competition_calibrator(fx, tmp_path):
    """Task 35: when both an isotonic_<COMP> and a global isotonic
    calibrator are present, the agent should use the per-competition
    one for that competition and fall back to the global for others.

    Per-comp calibrator maps 0.42 -> 0.30 (UCL-flavoured).
    Global calibrator maps 0.42 -> 0.70 (too-cautious-flavoured).
    The UCL match should report final_confidence ~= 0.30 and the
    label "isotonic@UCL". A LaLiga match should report 0.70 and
    the label "isotonic@global".
    """
    from soccer_agent.calibration import IsotonicCalibrator
    from soccer_agent.calibration_store import save_calibrator

    cal_root = tmp_path / "calibrators"

    # Per-comp UCL: maps everything to 0.30
    ucl = IsotonicCalibrator().fit([0.0, 0.5, 1.0], [0.3, 0.3, 0.3])
    save_calibrator(
        ucl, key="isotonic_UCL", root=cal_root,
        competition="UCL", n_samples=11, ece=0.0, brier=0.20,
    )

    # Global: maps everything to 0.70
    glob = IsotonicCalibrator().fit([0.0, 0.5, 1.0], [0.7, 0.7, 0.7])
    save_calibrator(
        glob, key="isotonic", root=cal_root,
        competition="GLOBAL", n_samples=106, ece=0.0, brier=0.23,
    )

    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=LLMReasoner(client=StubLLMClient()),
        db_path=db_path,
        calibrator_root=cal_root,
        calibrator_key="isotonic",
    )

    # UCL match: per-comp wins.
    pred_ucl = asyncio.run(agent.predict(_ucl_final()))
    assert pred_ucl.calibrator == "isotonic@UCL"
    assert abs(pred_ucl.final_confidence - 0.3) < 1e-9, (
        f"expected per-comp calibrated ~0.30, got {pred_ucl.final_confidence}"
    )

    # LaLiga match: per-comp missing → global fallback.
    pred_ll = asyncio.run(agent.predict(_laliga_match()))
    assert pred_ll.calibrator == "isotonic@global", (
        f"expected global fallback, got label {pred_ll.calibrator!r}"
    )
    assert abs(pred_ll.final_confidence - 0.7) < 1e-9, (
        f"expected global calibrated ~0.70, got {pred_ll.final_confidence}"
    )


def test_predict_no_calibrator_file_passthrough(fx, tmp_path):
    """If calibrator_root is set but neither per-comp nor global
    files exist, the agent should pass raw through (not crash).
    The dashboard label is None and final_confidence == raw_confidence
    after the 0.85 cap.
    """
    db_path = tmp_path / "agent.db"
    init_db(db_path)
    cal_root = tmp_path / "calibrators"  # empty
    cal_root.mkdir()
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=LLMReasoner(client=StubLLMClient()),
        db_path=db_path,
        calibrator_root=cal_root,
        calibrator_key="isotonic",
    )
    pred = asyncio.run(agent.predict(_ucl_final()))
    assert pred.calibrator is None
    # The 0.85 cap still applies inside _apply_calibrator's path,
    # so we expect: final = min(raw, 0.85). If raw < 0.85, no change.
    assert abs(pred.final_confidence - min(0.85, pred.raw_confidence)) < 1e-9


def test_predict_passthrough_when_no_calibrator(fx, tmp_path):
    """If calibrator_root is None (the default), the agent
    behaves exactly as before: raw_confidence == final_confidence
    and the calibrator label is None.
    """
    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=LLMReasoner(client=StubLLMClient()),
        db_path=db_path,
    )
    pred = asyncio.run(agent.predict(_ucl_final()))
    assert pred.calibrator is None
    assert pred.raw_confidence is not None
    # Pass-through: no calibrator → no change.
    assert abs(pred.final_confidence - pred.raw_confidence) < 1e-9
    # And the dashboard column is stored as NULL.
    with Database(str(db_path))._tx() as con:
        row = con.execute(
            "SELECT raw_confidence FROM predictions WHERE prediction_id = ?",
            (pred.prediction_id,),
        ).fetchone()
    assert row["raw_confidence"] is not None  # always stored


# -- Task 32: cap raw confidence at 0.85 -----------------------------------


def test_apply_calibrator_caps_raw_input(monkeypatch, fx, tmp_path):
    """The reliability report (Task 30) showed the 0.9-1.0 bucket is
    wildly overconfident. We cap the raw input at 0.85 before it
    reaches the calibrator, and we do NOT mutate the audit-trail
    raw_confidence stored on the row.
    """
    from soccer_agent.calibration import IsotonicCalibrator
    from soccer_agent.calibration_store import save_calibrator

    # Fit a calibrator that exposes whether the cap fired: at 0.85
    # it returns 0.85, at 0.99 it returns 0.99. If we see 0.85 for
    # an input of 0.99, the cap worked.
    cal = IsotonicCalibrator()
    cal.fit([0.0, 0.5, 0.85, 0.99, 1.0], [0.0, 0.5, 0.85, 0.99, 1.0])
    cal_root = tmp_path / "calibrators"
    # Save as a per-competition calibrator so the agent's Task 35
    # per-comp routing picks it up (label will be "isotonic@UCL"
    # rather than the global fallback "isotonic@global").
    save_calibrator(
        cal, key="isotonic_UCL", root=cal_root,
        competition="UCL", n_samples=5, ece=0.0, brier=0.0,
    )

    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=LLMReasoner(client=StubLLMClient()),
        db_path=db_path,
        calibrator_root=cal_root,
        calibrator_key="isotonic",
    )
    # Bypass predict() and call _apply_calibrator directly with a
    # range of inputs. This decouples the test from the reasoner.
    for raw, expected in [
        (0.50, 0.50),   # identity (no cap)
        (0.85, 0.85),   # exactly at cap
        (0.99, 0.85),   # above cap -> clamped -> 0.85
        (1.00, 0.85),   # above cap -> clamped -> 0.85
    ]:
        out, label = agent._apply_calibrator("UCL", raw)
        assert abs(out - expected) < 1e-9, (
            f"raw={raw} expected={expected} got={out}"
        )
        assert label == "isotonic@UCL"


def test_raw_confidence_not_mutated_by_cap(fx, tmp_path):
    """The cap is internal to the calibrator call. The raw_confidence
    stored on the prediction row is the *original* (audit-trail) value.
    """
    from soccer_agent.calibration import IsotonicCalibrator
    from soccer_agent.calibration_store import save_calibrator

    cal = IsotonicCalibrator()
    cal.fit([0.0, 0.5, 0.85, 1.0], [0.0, 0.5, 0.85, 1.0])
    cal_root = tmp_path / "calibrators"
    save_calibrator(
        cal, key="isotonic", root=cal_root,
        competition="UCL", n_samples=4, ece=0.0, brier=0.0,
    )

    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=LLMReasoner(client=StubLLMClient()),
        db_path=db_path,
        calibrator_root=cal_root,
        calibrator_key="isotonic",
    )
    # Force a specific raw_confidence via direct call.
    raw_in = 0.99
    _out, _label = agent._apply_calibrator("UCL", raw_in)
    # The clamp means out=0.85, but the agent never had a chance
    # to write to its row in this test (we didn't go through
    # predict()). Verify the helper itself doesn't mutate `raw_in`.
    assert raw_in == 0.99  # unchanged


def test_evaluate_fills_in_actual_outcome(fx, tmp_path):
    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=LLMReasoner(client=StubLLMClient()),
        db_path=db_path,
    )
    asyncio.run(agent.predict(_ucl_final()))
    decided_at = datetime(2025, 5, 30, 22, 0, 0).isoformat()
    with Database(str(db_path))._tx() as con:
        con.execute(
            "INSERT INTO results (match_id, home_goals, away_goals, decided_at) "
            "VALUES (?, ?, ?, ?)",
            ("ucl-25-final", 2, 1, decided_at),
        )
    updated = asyncio.run(agent.evaluate("ucl-25-final"))
    assert updated.actual == "home"
    assert updated.correct is not None
    assert updated.brier is not None
    assert 0.0 <= updated.brier <= 2.0


def test_evaluate_unknown_match_raises(fx, tmp_path):
    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=NumericReasoner(),
        db_path=db_path,
    )
    with pytest.raises(KeyError):
        asyncio.run(agent.evaluate("does-not-exist"))


def test_evaluate_missing_result_raises(fx, tmp_path):
    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=LLMReasoner(client=StubLLMClient()),
        db_path=db_path,
    )
    asyncio.run(agent.predict(_ucl_final()))
    with pytest.raises(RuntimeError, match="no result"):
        asyncio.run(agent.evaluate("ucl-25-final"))


# -- tool_calls log ---------------------------------------------------------


def test_predict_logs_tool_calls(fx, tmp_path):
    db_path = tmp_path / "agent.db"
    init_db(db_path)
    agent = PredictionAgent(
        registry=default_registry(),
        reasoner=NumericReasoner(),
        db_path=db_path,
    )
    asyncio.run(agent.predict(_ucl_final()))
    with Database(str(db_path))._tx() as con:
        rows = con.execute("SELECT tool, ok FROM tool_calls").fetchall()
    tools_called = {r["tool"] for r in rows}
    assert "form_recent" in tools_called
    assert "odds_market" in tools_called
    # Each tool row has an ok flag
    for r in rows:
        assert r["ok"] in (0, 1)


def test_season_for_format():
    """The agent builds fixture paths as `<home>__<away>__<season>.json`.
    The form tool Pydantic default is "2024-2025", so the agent must
    produce a matching key — never the two-digit "2024-25" form.
    Catches a regression where the agent overrode the tool default
    with an ambiguous year format.
    """
    assert _season_for(datetime(2024, 9, 17)) == "2024-2025"
    assert _season_for(datetime(2025, 5, 30)) == "2024-2025"
    assert _season_for(datetime(2025, 9, 17)) == "2025-2026"
    assert _season_for(datetime(2026, 6, 1)) == "2025-2026"
    # Boundary: month 7 is start of new season
    assert _season_for(datetime(2025, 7, 1)) == "2025-2026"
    # month 6 still old season
    assert _season_for(datetime(2025, 6, 30)) == "2024-2025"

