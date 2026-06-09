"""Tests for the form_recent tool."""

from __future__ import annotations

import pytest

from soccer_agent.tools import ToolError, ToolRegistry
from soccer_agent.tools._fixtures import write_json
from soccer_agent.tools.form_recent import FormInput, FormRecentTool


SAMPLE_FORM = {
    "home": {
        "played": 5, "won": 3, "drawn": 1, "lost": 1,
        "gf": 8, "ga": 4, "points": 10, "last5_form_string": "WWDWL",
    },
    "away": {
        "played": 5, "won": 2, "drawn": 2, "lost": 1,
        "gf": 6, "ga": 5, "points": 8, "last5_form_string": "WDWLW",
    },
}


def test_form_recent_falls_back_to_fixture(monkeypatch, tmp_path):
    monkeypatch.setenv("SOCCER_AGENT_FIXTURES_DIR", str(tmp_path / "fx"))
    write_json("form", "man_city__real_madrid__2024-2025.json", data=SAMPLE_FORM)

    reg = ToolRegistry()
    reg.register(FormRecentTool())
    import asyncio
    res = asyncio.run(reg.run(
        "form_recent",
        FormInput(home_team_id="man_city", away_team_id="real_madrid", season="2024-2025"),
    ))
    assert res.ok is True
    assert res.data.home.points == 10
    assert res.data.away.last5_form_string == "WDWLW"


def test_form_recent_missing_fixture_returns_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("SOCCER_AGENT_FIXTURES_DIR", str(tmp_path / "fx"))
    reg = ToolRegistry()
    reg.register(FormRecentTool())
    import asyncio
    res = asyncio.run(reg.run(
        "form_recent",
        FormInput(home_team_id="unknown", away_team_id="unknown", season="2099-2100"),
    ))
    assert res.ok is False
    assert "no form fixture" in (res.error or "")


def test_form_recent_input_validates_season(monkeypatch, tmp_path):
    monkeypatch.setenv("SOCCER_AGENT_FIXTURES_DIR", str(tmp_path / "fx"))
    reg = ToolRegistry()
    reg.register(FormRecentTool())
    import asyncio
    # n_matches out of range — pass a dict to bypass test-time Pydantic validation;
    # the registry should catch it and return ok=False
    res = asyncio.run(reg.run(
        "form_recent",
        {"home_team_id": "a", "away_team_id": "b", "season": "x", "n_matches": -1},
    ))
    assert res.ok is False
    assert "payload does not match" in (res.error or "")


def test_form_recent_input_validates_n_matches_upper_bound(monkeypatch, tmp_path):
    monkeypatch.setenv("SOCCER_AGENT_FIXTURES_DIR", str(tmp_path / "fx"))
    reg = ToolRegistry()
    reg.register(FormRecentTool())
    import asyncio
    res = asyncio.run(reg.run(
        "form_recent",
        {"home_team_id": "a", "away_team_id": "b", "season": "x", "n_matches": 999},
    ))
    assert res.ok is False
