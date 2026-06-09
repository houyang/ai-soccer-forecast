"""Tests for scripts/build_elo_state.py (Task 27)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "build_elo_state.py"


def _write_matches(path: Path) -> None:
    """Write a tiny 4-match history: a wins twice at home, b wins
    once at home, a wins once away. Net: a > b."""
    rows = [
        {"date": "2024-08-01", "home_id": "a", "away_id": "b",
         "home_goals": 2, "away_goals": 0},
        {"date": "2024-08-08", "home_id": "b", "away_id": "a",
         "home_goals": 1, "away_goals": 2},
        {"date": "2024-08-15", "home_id": "a", "away_id": "b",
         "home_goals": 3, "away_goals": 1},
        {"date": "2024-08-22", "home_id": "b", "away_id": "a",
         "home_goals": 0, "away_goals": 1},
    ]
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_build_elo_state_writes_valid_json(tmp_path):
    matches = tmp_path / "matches.jsonl"
    out = tmp_path / "elo_state.json"
    _write_matches(matches)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--matches", str(matches), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert "ratings" in data
    assert "a" in data["ratings"] and "b" in data["ratings"]
    # a won 3 of 4 → should have a higher overall rating than b
    assert data["ratings"]["a"]["overall"] > data["ratings"]["b"]["overall"]


def test_build_elo_state_handles_unsorted_input(tmp_path):
    """Order in the file should not matter; date is used for sorting."""
    matches = tmp_path / "shuffled.jsonl"
    out = tmp_path / "elo_state.json"
    rows = [
        {"date": "2024-08-22", "home_id": "b", "away_id": "a",
         "home_goals": 0, "away_goals": 1},
        {"date": "2024-08-01", "home_id": "a", "away_id": "b",
         "home_goals": 2, "away_goals": 0},
        {"date": "2024-08-15", "home_id": "a", "away_id": "b",
         "home_goals": 3, "away_goals": 1},
        {"date": "2024-08-08", "home_id": "b", "away_id": "a",
         "home_goals": 1, "away_goals": 2},
    ]
    with matches.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--matches", str(matches), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    # Same expected outcome: a > b.
    assert data["ratings"]["a"]["overall"] > data["ratings"]["b"]["overall"]


def test_build_elo_state_rejects_missing_fields(tmp_path):
    matches = tmp_path / "bad.jsonl"
    out = tmp_path / "elo_state.json"
    matches.write_text(
        json.dumps({"date": "2024-08-01", "home_id": "a", "home_goals": 1})
        + "\n"  # missing away_id and away_goals
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--matches", str(matches), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "missing" in result.stderr.lower()


def test_build_elo_state_reports_top_teams(tmp_path):
    matches = tmp_path / "matches.jsonl"
    out = tmp_path / "elo_state.json"
    _write_matches(matches)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--matches", str(matches),
         "--out", str(out), "--report"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Top 10 teams" in result.stdout
    assert "a" in result.stdout
