"""CLI surface tests.

We use click's CliRunner — no actual subprocess, but exercises the
real command group and entry points end-to-end.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from soccer_agent.api.cli import main
from soccer_agent.tools._fixtures import write_json
from soccer_agent.tools import default_registry


# ---- shared fixtures (mirror test_agent.py so CLI tests are self-contained) ----

FORM = {
    "team": "man_city",
    "as_of": "2025-05-30",
    "last_n": 5,
    "entries": [
        {"opponent": "arsenal",      "venue": "home", "gf": 2, "ga": 1, "result": "W", "date": "2025-05-26"},
        {"opponent": "tottenham",    "venue": "away", "gf": 1, "ga": 1, "result": "D", "date": "2025-05-22"},
        {"opponent": "fulham",       "venue": "home", "gf": 3, "ga": 0, "result": "W", "date": "2025-05-18"},
        {"opponent": "wolves",       "venue": "away", "gf": 2, "ga": 2, "result": "D", "date": "2025-05-14"},
        {"opponent": "bournemouth",  "venue": "home", "gf": 4, "ga": 1, "result": "W", "date": "2025-05-10"},
    ],
    "wins": 3, "draws": 2, "losses": 0, "gf": 12, "ga": 5, "form_string": "WWDWW",
}
INJURY = {
    "as_of": "2025-05-30",
    "home": [{"player": "haaland", "status": "fit", "note": ""}],
    "away": [{"player": "mbappe", "status": "doubtful", "note": "hamstring"}],
    "summary": "Away side carries one doubt",
}
H2H = {
    "home_team_id": "man_city", "away_team_id": "real_madrid",
    "meetings": [
        {"date": "2024-09-30T20:00:00", "competition": "UCL",
         "home_goals": 2, "away_goals": 1, "venue": "etihad", "winner": "home"},
    ],
    "home_wins": 1, "draws": 0, "away_wins": 0,
    "last_winner": "home", "last_meeting": "2024-09-30T20:00:00",
}
WEATHER = {"temp_c": 18.0, "wind_kph": 10.0, "precip_mm": 0.0,
           "conditions": "clear", "playability_risk": 0.05}
ODDS = {"bookmakers": [
    {"name": "Bet365", "home": 2.10, "draw": 3.40, "away": 3.60, "fetched_at": "2025-05-30T12:00:00"},
    {"name": "Pinnacle", "home": 2.05, "draw": 3.50, "away": 3.70, "fetched_at": "2025-05-30T12:01:00"},
],
    "market_consensus_pick": "home", "market_consensus_prob": 0.49}
VENUE = {"id": "puskas_arena", "name": "Puskás Aréna", "city": "Budapest",
         "country": "Hungary", "is_neutral": True, "is_dome": False, "altitude_m": 110,
         "lat": 47.5027, "lon": 19.0938}


def _seed_fixtures(root: Path) -> None:
    write_json("form", "man_city__real_madrid__2024-2025.json", data=FORM)
    write_json("injury", "man_city__real_madrid__2025-05-30.json", data=INJURY)
    write_json("h2h", "man_city__real_madrid.json", data=H2H)
    write_json("weather", "puskas_arena__2025-05-30.json", data=WEATHER)
    write_json("odds", "man_city__real_madrid__2025-05-30.json", data=ODDS)
    write_json("venues", "venue_puskas_arena.json", data=VENUE)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Seed fixture dir + point the DB at a tmp file."""
    fx = tmp_path / "fx"
    fx.mkdir()
    db = tmp_path / "agent.db"
    monkeypatch.setenv("SOCCER_AGENT_FIXTURES_DIR", str(fx))
    monkeypatch.setenv("SOCCER_AGENT_DB_PATH", str(db))
    _seed_fixtures(fx)
    return fx, db


# ---- test cases ----

def test_cli_predict_emits_json_to_stdout(env):
    """`soccer-agent predict ...` should print a JSON Prediction and exit 0."""
    from soccer_agent.api.cli import predict_cmd
    runner = CliRunner()
    result = runner.invoke(predict_cmd, [
        "--home-id", "man_city",
        "--away-id", "real_madrid",
        "--venue-id", "puskas_arena",
        "--kickoff", "2025-05-30T20:00:00",
        "--competition", "UCL",
        "--season", "2025/26",
        "--round", "final",
    ])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["match_id"]
    assert payload["pick"] in ("home", "draw", "away")
    assert 0.0 <= payload["confidence"] <= 1.0


def test_cli_predict_writes_to_db(env):
    """A successful `predict` should leave exactly one row in predictions table."""
    from soccer_agent.api.cli import predict_cmd
    from soccer_agent.db import Database
    runner = CliRunner()
    runner.invoke(predict_cmd, [
        "--home-id", "man_city",
        "--away-id", "real_madrid",
        "--venue-id", "puskas_arena",
        "--kickoff", "2025-05-30T20:00:00",
    ])
    db = Database(str(env[1]))
    rows = db.execute("SELECT COUNT(*) AS c FROM predictions")
    assert rows[0]["c"] == 1


def test_cli_evaluate_marks_correct(env):
    """`soccer-agent evaluate` should set correct=true when actual == pick."""
    from soccer_agent.api.cli import predict_cmd, evaluate_cmd
    runner = CliRunner()
    # First make a prediction
    pred_result = runner.invoke(predict_cmd, [
        "--home-id", "man_city",
        "--away-id", "real_madrid",
        "--venue-id", "puskas_arena",
        "--kickoff", "2025-05-30T20:00:00",
    ])
    assert pred_result.exit_code == 0, pred_result.output
    pred = json.loads(pred_result.output)
    # The numeric reasoner picks whichever has highest probs.
    # To force a known result, we first peek at the pick, then we
    # re-run the scenario: easier — set goals that map to the actual pick.
    actual_pick = pred["pick"]
    if actual_pick == "home":
        hg, ag = "2", "1"
    elif actual_pick == "away":
        hg, ag = "0", "2"
    else:
        hg, ag = "1", "1"
    # Then evaluate
    result = runner.invoke(evaluate_cmd, [
        "--match-id", pred["match_id"],
        "--home-goals", hg,
        "--away-goals", ag,
    ])
    assert result.exit_code == 0, result.output
    # Output is a single JSON line; the result block nests actual / was_correct / brier.
    payload = json.loads(result.output)
    assert payload["result"]["was_correct"] is True, result.output


def test_signal_source_accepts_tool_and_registry_strings():
    """Signal.source is a Literal used for UI badges ('live' / 'fixture').
    Tools that fail with a non-data error get source='tool' or 'registry';
    the agent must coerce those into the allowed set without crashing.
    Regression for the bug the integration test suite missed: the shell
    CLI run surfaced every tool as 'ok=False source=tool' because the
    Signal Literal didn't allow 'tool'.
    """
    from soccer_agent.models import Signal, ToolErrorPayload
    sig = Signal(
        tool="odds_market",
        ok=False,
        data={},
        source="tool",  # was rejected before — Literal only had live/fixture
        error=ToolErrorPayload(source="tool", message="oops", retriable=False),
    )
    # Source must round-trip — UI badges rely on the literal value.
    assert sig.source == "tool"


def test_cli_help_runs():
    """`--help` should print usage and exit 0 (sanity check on the group)."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "predict" in result.output
    assert "evaluate" in result.output
    assert "eval" in result.output


def test_cli_emit_handles_nan_as_null():
    """The _emit helper must replace NaN/inf with null (JSON has no NaN
    literal; downstream consumers like jq would reject the line)."""
    from soccer_agent.api.cli import _emit
    import math
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        _emit({"accuracy": math.nan, "brier": math.inf, "log_loss": 0.6, "per_class": {"draw": {"f1": math.nan}}})
    text = buf.getvalue().strip()
    import json as _json
    parsed = _json.loads(text)
    assert parsed["accuracy"] is None
    assert parsed["brier"] is None
    assert parsed["log_loss"] == 0.6
    assert parsed["per_class"]["draw"]["f1"] is None


def test_cli_eval_runs_against_dataset(env):
    """`soccer-agent eval` should run the harness over EVAL_CASES and
    print a JSON summary containing the headline metrics."""
    _, db_path = env
    runner = CliRunner()
    result = runner.invoke(main, ["eval"])
    assert result.exit_code == 0, result.output
    # last line of stdout is the JSON summary
    payload = json.loads(result.output.strip().splitlines()[-1])
    assert payload["n_total"] == len(__import__("soccer_agent.eval.dataset", fromlist=["EVAL_CASES"]).EVAL_CASES)
    for key in ("accuracy", "brier_mean", "log_loss", "per_class", "calibration_ece"):
        assert key in payload
    # and an eval_runs row landed in the DB
    con = sqlite3.connect(str(db_path))
    n = con.execute("SELECT COUNT(*) FROM eval_runs").fetchone()[0]
    con.close()
    assert n == 1


def test_cli_eval_writes_summary_to_output_file(env, tmp_path):
    """`soccer-agent eval --output PATH` should write the summary to PATH."""
    runner = CliRunner()
    out = tmp_path / "summary.json"
    result = runner.invoke(main, ["eval", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert "accuracy" in loaded
    assert "brier_mean" in loaded
