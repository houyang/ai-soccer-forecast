"""Tests for the five supporting tools: injury, h2h, weather, odds, venue."""

from __future__ import annotations

import asyncio

import pytest

from soccer_agent.tools import default_registry
from soccer_agent.tools._fixtures import write_json


# -- sample data per tool -----------------------------------------------------


SAMPLE_INJURY = {
    "home": [
        {
            "player": "Rodri", "status": "out",
            "reported_at": "2025-04-10T09:00:00Z",
            "source": "press_conf", "summary": "ACL — season over",
        }
    ],
    "away": [],
}


SAMPLE_H2H = {
    "meetings": [
        {"date": "2024-05-04T20:00:00Z", "home": "real_madrid", "away": "man_city",
         "home_goals": 1, "away_goals": 1, "competition": "UCL"},
        {"date": "2023-05-17T20:00:00Z", "home": "man_city", "away": "real_madrid",
         "home_goals": 4, "away_goals": 0, "competition": "UCL"},
    ],
    "home_wins": 1, "away_wins": 0, "draws": 1,
    "last_meeting": "2024-05-04T20:00:00Z", "last_winner": "draw",
}


SAMPLE_WEATHER = {
    "temp_c": 8.0, "precip_mm": 0.0, "wind_kph": 12.0,
    "conditions": "clear", "is_dome": False, "playability_risk": "low",
}


SAMPLE_ODDS = {
    "bookmakers": [
        {"name": "pinnacle", "home": 2.10, "draw": 3.40, "away": 3.60},
        {"name": "bet365",   "home": 2.05, "draw": 3.50, "away": 3.70},
    ],
    "implied_probs": {"home": 0.475, "draw": 0.275, "away": 0.250},
    "market_consensus_pick": "home",
}


SAMPLE_VENUE = {
    "id": "metlife_stadium",
    "name": "MetLife Stadium",
    "city": "East Rutherford", "country": "USA",
    "capacity": 82500, "surface": "grass", "is_neutral": True,
    "altitude_m": 7, "is_dome": False, "lat": 40.8136, "lon": -74.0743,
}


@pytest.fixture
def fx(monkeypatch, tmp_path):
    monkeypatch.setenv("SOCCER_AGENT_FIXTURES_DIR", str(tmp_path / "fx"))
    return tmp_path / "fx"


# -- injury ------------------------------------------------------------------


def test_injury_news_falls_back_to_fixture(fx):
    write_json("injury", "man_city__real_madrid__2025-05-30.json", data=SAMPLE_INJURY)
    reg = default_registry()
    res = asyncio.run(reg.run(
        "injury_news",
        {"home_team_id": "man_city", "away_team_id": "real_madrid", "kickoff_date": "2025-05-30"},
    ))
    assert res.ok is True
    assert res.data.home[0].player == "Rodri"
    assert res.data.away == []


def test_injury_news_missing_fixture(fx):
    reg = default_registry()
    res = asyncio.run(reg.run(
        "injury_news",
        {"home_team_id": "x", "away_team_id": "y", "kickoff_date": "2099-01-01"},
    ))
    assert res.ok is False
    assert "no injury fixture" in (res.error or "")


# -- h2h ---------------------------------------------------------------------


def test_h2h_falls_back_to_fixture(fx):
    write_json("h2h", "real_madrid__man_city.json", data=SAMPLE_H2H)
    reg = default_registry()
    res = asyncio.run(reg.run(
        "h2h_history",
        {"home_team_id": "real_madrid", "away_team_id": "man_city", "n_meetings": 10},
    ))
    assert res.ok is True
    assert res.data.home_wins == 1
    assert res.data.last_winner == "draw"


def test_h2h_missing_fixture(fx):
    reg = default_registry()
    res = asyncio.run(reg.run(
        "h2h_history",
        {"home_team_id": "x", "away_team_id": "y", "n_meetings": 10},
    ))
    assert res.ok is False
    assert "no h2h fixture" in (res.error or "")


# -- weather -----------------------------------------------------------------


def test_weather_falls_back_to_fixture(fx):
    write_json("weather", "metlife_stadium__2026-07-19.json", data=SAMPLE_WEATHER)
    reg = default_registry()
    res = asyncio.run(reg.run(
        "weather_venue",
        {"venue_id": "metlife_stadium", "date": "2026-07-19", "is_dome": False},
    ))
    assert res.ok is True
    assert res.data.temp_c == 8.0
    assert res.data.playability_risk == "low"


def test_weather_missing_fixture(fx):
    reg = default_registry()
    res = asyncio.run(reg.run(
        "weather_venue",
        {"venue_id": "x", "date": "2099-01-01", "is_dome": False},
    ))
    assert res.ok is False


# -- odds --------------------------------------------------------------------


def test_odds_falls_back_to_fixture(fx):
    write_json("odds", "man_city__real_madrid__2025-05-30.json", data=SAMPLE_ODDS)
    reg = default_registry()
    res = asyncio.run(reg.run(
        "odds_market",
        {"home_team_id": "man_city", "away_team_id": "real_madrid", "kickoff_date": "2025-05-30"},
    ))
    assert res.ok is True
    assert res.data.market_consensus_pick == "home"
    assert len(res.data.bookmakers) == 2


def test_odds_missing_fixture(fx):
    reg = default_registry()
    res = asyncio.run(reg.run(
        "odds_market",
        {"home_team_id": "x", "away_team_id": "y", "kickoff_date": "2099-01-01"},
    ))
    assert res.ok is False


# -- venue -------------------------------------------------------------------


def test_venue_falls_back_to_fixture(fx):
    write_json("venues", "venue_metlife_stadium.json", data=SAMPLE_VENUE)
    reg = default_registry()
    res = asyncio.run(reg.run("venue_info", {"venue_id": "metlife_stadium"}))
    assert res.ok is True
    assert res.data.capacity == 82500
    assert res.data.is_neutral is True


def test_venue_missing_fixture(fx):
    reg = default_registry()
    res = asyncio.run(reg.run("venue_info", {"venue_id": "nowhere"}))
    assert res.ok is False
    assert "no venue fixture" in (res.error or "")


# -- default_registry has all six --------------------------------------------


def test_default_registry_has_all_six_tools():
    reg = default_registry()
    assert set(reg.names) == {
        "form_recent", "injury_news", "h2h_history",
        "weather_venue", "odds_market", "venue_info",
    }
