"""Tests for Task 18: the eval harness.

The harness is a single function (`run_eval`) that:
  1. For each case in `EVAL_CASES`, materializes fixtures, predicts with
     the agent, and (optionally) records the actual result.
  2. Pulls all joined rows back from the DB and runs the metric suite.
  3. Persists a single `eval_runs` row summarising the run (timestamp,
     reasoner name, n_cases, accuracy, brier_mean, etc).
  4. Returns the summary dict (and writes it to JSON if `output` is set).

It MUST be idempotent: a second call in the same DB must re-score the
existing predictions without inserting duplicates. Predictions are
keyed by `match_id` — running the same case twice should update, not
insert a new row.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from soccer_agent.eval.dataset import EVAL_CASES
from soccer_agent.eval.harness import EvalHarness, run_eval
from soccer_agent.eval.metrics import metric_summary


# -- helpers ------------------------------------------------------------------

def _isolated_paths(tmp_path: Path):
    """Return (fixtures_dir, db_path) that don't collide with anything else."""
    return tmp_path / "fixtures", tmp_path / "agent.db"


def _make_harness(tmp_path: Path, **kw) -> EvalHarness:
    fixtures, db = _isolated_paths(tmp_path)
    fixtures.mkdir()
    return EvalHarness(
        fixtures_dir=fixtures,
        db_path=db,
        tool_names=kw.get("tool_names", [
            "form_recent", "injury_news", "h2h_history",
            "weather_venue", "odds_market", "venue_info",
        ]),
        reasoner=kw.get("reasoner", "numeric"),
        calibrator_root=kw.get("calibrator_root"),
        calibrator_key=kw.get("calibrator_key", "isotonic"),
    )


def test_harness_supports_calibrator_args(tmp_path):
    """EvalHarness must plumb calibrator_root + calibrator_key into
    the agent (Task 35: per-competition calibrators depend on this)."""
    h = _make_harness(tmp_path, calibrator_key="isotonic_EPL")
    assert h.calibrator_key == "isotonic_EPL"
    # calibrator_root stays None by default — the agent's __init__
    # short-circuits calibration in that case (see agent.py).
    assert h.calibrator_root is None


# -- basic API surface --------------------------------------------------------

def test_run_eval_returns_summary_dict(tmp_path):
    h = _make_harness(tmp_path)
    summary = h.run()
    assert isinstance(summary, dict)
    # every metric the metrics module emits should be present
    for key in ("n_total", "n_resolved", "accuracy", "brier_mean",
                "log_loss", "per_class", "calibration_ece"):
        assert key in summary
    assert summary["n_total"] == len(EVAL_CASES)


def test_run_eval_writes_eval_runs_row(tmp_path):
    h = _make_harness(tmp_path)
    h.run()
    con = sqlite3.connect(str(h.db_path))
    rows = con.execute("SELECT * FROM eval_runs").fetchall()
    con.close()
    assert len(rows) == 1
    # columns: id, run_at, reasoner, n_cases, n_resolved, accuracy, brier_mean, log_loss, summary_json, ...
    # We don't pin the schema here; just check the run row exists.


def test_run_eval_writes_json_output_when_requested(tmp_path):
    h = _make_harness(tmp_path)
    out = tmp_path / "summary.json"
    summary = h.run(output=out)
    assert out.exists()
    loaded = json.loads(out.read_text())
    # JSON round-trip should preserve at least the headline metrics
    assert loaded["n_total"] == summary["n_total"]
    assert loaded["accuracy"] == summary["accuracy"]


def test_run_eval_persists_predictions_to_db(tmp_path):
    h = _make_harness(tmp_path)
    h.run()
    con = sqlite3.connect(str(h.db_path))
    n = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    con.close()
    assert n == len(EVAL_CASES)


def test_run_eval_uses_test_results_when_present(tmp_path):
    """Cases in EVAL_CASES carry `actual_winner` and `actual_score`. The
    harness should record those results so the predictions get graded.
    """
    h = _make_harness(tmp_path)
    h.run()
    con = sqlite3.connect(str(h.db_path))
    n_results = con.execute("SELECT COUNT(*) FROM results").fetchone()[0]
    con.close()
    assert n_results == len(EVAL_CASES)


# -- idempotency --------------------------------------------------------------

def test_run_eval_is_idempotent_on_repeat(tmp_path):
    """Running the harness twice in the same DB should NOT double-insert
    predictions. It should re-score the existing rows (and update the
    eval_runs row count by adding a new run, not by overwriting)."""
    h = _make_harness(tmp_path)
    h.run()
    h.run()  # second call
    con = sqlite3.connect(str(h.db_path))
    n_preds = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    n_runs = con.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0]
    con.close()
    assert n_preds == len(EVAL_CASES), "predictions must not double-insert"
    assert n_runs == 2, "eval_runs should accumulate (one per run)"


# -- reasoner selection -------------------------------------------------------

def test_run_eval_reasoner_override(tmp_path):
    """`reasoner='numeric'` is the default; another valid value is 'llm'
    (which uses the LLMReasoner; it falls back to numeric if no API key
    is set, so the test stays deterministic)."""
    h = _make_harness(tmp_path, reasoner="llm")
    summary = h.run()
    assert "reasoner" in summary
    assert summary["reasoner"] in ("llm", "numeric")  # may fall back


# -- per-class metrics exposed ------------------------------------------------

def test_run_eval_per_class_summary_has_three_classes(tmp_path):
    h = _make_harness(tmp_path)
    summary = h.run()
    assert set(summary["per_class"].keys()) == {"home", "draw", "away"}
    for cls, m in summary["per_class"].items():
        assert {"precision", "recall", "f1", "support"} <= set(m.keys())


# -- fixtures actually materialize -------------------------------------------

def test_run_eval_writes_form_fixtures(tmp_path):
    """If fixtures don't exist, the harness should materialize them.

    Form fixtures are stamped with season + home/away, so the count
    equals the number of unique (home, away, season) tuples across
    all cases — not the number of cases (Task 34 added 72 cases
    spanning two seasons, lifting pair-cardinality to ~99).
    """
    h = _make_harness(tmp_path)
    h.run()
    form_dir = h.fixtures_dir / "form"
    assert form_dir.exists()
    files = list(form_dir.glob("*.json"))
    # Derive expected season for each case (same rule as the factory).
    def season_of(kickoff):
        y, m = kickoff.year, kickoff.month
        return f"{y}-{y + 1}" if m >= 8 else f"{y - 1}-{y}"
    expected = {(c.home_id, c.away_id, season_of(c.kickoff))
                for c in EVAL_CASES}
    assert len(files) == len(expected), (
        f"expected {len(expected)} (home, away, season)-fixtures, "
        f"got {len(files)}"
    )


# -- run_eval module-level function ------------------------------------------

def test_run_eval_module_function(tmp_path):
    """The harness should also be callable via a top-level function for
    scripts / cron. Same behaviour as the class."""
    fixtures, db = _isolated_paths(tmp_path)
    fixtures.mkdir()
    summary = run_eval(fixtures_dir=fixtures, db_path=db)
    assert summary["n_total"] == len(EVAL_CASES)
