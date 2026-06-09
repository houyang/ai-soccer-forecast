"""Tests for the fixture loader."""

from __future__ import annotations

import json
import os

import pytest

from soccer_agent.config import Settings
from soccer_agent.tools._fixtures import (
    FixtureNotFound,
    fixture_path,
    load_json,
    load_json_or_raise,
    write_json,
)


@pytest.fixture
def fx_dir(tmp_path, monkeypatch):
    """Point SOCCER_AGENT_FIXTURES_DIR at a tmp directory."""
    monkeypatch.setenv("SOCCER_AGENT_FIXTURES_DIR", str(tmp_path / "fx"))
    return tmp_path / "fx"


def test_load_json_returns_default_when_missing(fx_dir):
    assert load_json("missing.json") is None
    assert load_json("missing.json", default=[]) == []


def test_load_json_round_trip(fx_dir):
    write_json("form", "home__away__2024.json", data={"home": {"points": 9}})
    data = load_json("form", "home__away__2024.json")
    assert data == {"home": {"points": 9}}


def test_load_json_or_raise(fx_dir):
    write_json("odds", "a__b__2024-05-30.json", data={"foo": 1})
    assert load_json_or_raise("odds", "a__b__2024-05-30.json") == {"foo": 1}
    with pytest.raises(FixtureNotFound):
        load_json_or_raise("odds", "nope.json")


def test_fixture_path_under_base(fx_dir):
    p = fixture_path("teams", "team_man_city.json")
    assert p.parts[-2:] == ("teams", "team_man_city.json")
    assert str(p).startswith(str(fx_dir))


def test_unsafe_key_part_rejected():
    with pytest.raises(ValueError):
        fixture_path("teams/../bad.json")


def test_write_json_creates_dirs(fx_dir):
    out = write_json("form", "a__b__2024.json", data={"x": 1})
    assert out.exists()
    with out.open() as f:
        assert json.load(f) == {"x": 1}


def test_load_json_handles_strings_lists_dicts(fx_dir):
    write_json("misc", "a.json", data=["a", "b"])
    write_json("misc", "b.json", data="hello")
    write_json("misc", "c.json", data={"nested": {"k": 1}})
    assert load_json("misc", "a.json") == ["a", "b"]
    assert load_json("misc", "b.json") == "hello"
    assert load_json("misc", "c.json") == {"nested": {"k": 1}}
