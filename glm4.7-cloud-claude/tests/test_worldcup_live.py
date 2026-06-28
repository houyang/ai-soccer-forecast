# tests/test_worldcup_live.py
import json
from pathlib import Path

from soccer_agent.worldcup.live import parse_lineup_response


SAMPLE = {
    "get": "fixtures/lineups",
    "response": [
        {
            "team": {"id": 16, "name": "Mexico"},
            "coach": {"id": 1, "name": "Javier Aguirre"},
            "formation": "4-1-4-1",
            "startXI": [{"player": {"id": 270774, "name": "R. Rangel", "pos": "G", "grid": "1:1"}},
                        {"player": {"id": 11, "name": "X", "pos": "D", "grid": "2:1"}}],
            "substitutes": [{"player": {"id": 2098, "name": "G. Ochoa", "pos": "G", "grid": None}}],
        },
        {
            "team": {"id": 1531, "name": "South Africa"},
            "coach": {"id": 2, "name": "H. Broos"},
            "formation": "4-3-3",
            "startXI": [{"player": {"id": 50, "name": "R. Williams", "pos": "G", "grid": "1:1"}}],
            "substitutes": [],
        },
    ],
}


def test_parse_lineup_response():
    lineups = parse_lineup_response(SAMPLE, fixture_id=1489369)
    assert len(lineups) == 2
    mex = next(lu for lu in lineups if lu.team_id == 16)
    assert mex.formation == "4-1-4-1"
    assert mex.start_ids == (270774, 11)
    assert mex.sub_ids == (2098,)


def test_parse_empty_response_returns_empty():
    assert parse_lineup_response({"response": []}, fixture_id=1) == []


def test_fetcher_without_key_returns_none(tmp_path, monkeypatch):
    from soccer_agent.worldcup.live import LineupFetcher
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    f = LineupFetcher(cache_dir=tmp_path)
    assert f.fetch_fixture_lineups(1489369) is None
