from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from soccer.api_football import (
    ApiParam,
    JsonObject,
    fetch_world_cup_2026_match_preview_updates,
    fetch_world_cup_2026_match_updates,
    fetch_world_cup_2026_snapshot,
)


class FakeFootballApi:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, ApiParam]]] = []

    def get_json(self, endpoint: str, params: Mapping[str, ApiParam]) -> JsonObject:
        request_params = dict(params)
        self.calls.append((endpoint, request_params))

        if endpoint == "fixtures" and request_params.get("league") == 1:
            return {
                "response": [
                    {
                        "fixture": {
                            "id": 1,
                            "date": "2026-06-11T19:00:00Z",
                            "status": {"short": "FT"},
                        },
                        "league": {"round": "Group A - 1"},
                        "teams": {
                            "home": {"id": 1, "name": "Alpha"},
                            "away": {"id": 2, "name": "Beta"},
                        },
                        "goals": {"home": 2, "away": 0},
                    },
                    {
                        "fixture": {
                            "id": 2,
                            "date": "2026-06-18T19:00:00Z",
                            "status": {"short": "NS"},
                        },
                        "league": {"round": "Group A - 2"},
                        "teams": {
                            "home": {"id": 2, "name": "Beta"},
                            "away": {"id": 1, "name": "Alpha"},
                        },
                        "goals": {"home": None, "away": None},
                    },
                ]
            }
        if endpoint == "teams":
            return {
                "response": [
                    {"team": {"id": 1, "name": "Alpha"}},
                    {"team": {"id": 2, "name": "Beta"}},
                ]
            }
        if endpoint == "standings" and request_params.get("league") == 1:
            return {
                "response": [
                    {
                        "league": {
                            "standings": [
                                [
                                    {
                                        "team": {"id": 1, "name": "Alpha"},
                                        "group": "Group A",
                                    },
                                    {
                                        "team": {"id": 2, "name": "Beta"},
                                        "group": "Group A",
                                    },
                                ]
                            ]
                        }
                    }
                ]
            }
        if endpoint == "players/squads":
            team_id = int(request_params["team"])
            return {
                "response": [
                    {
                        "team": {"id": team_id},
                        "players": [
                            {
                                "id": team_id * 10 + 1,
                                "name": f"Player {team_id}",
                                "age": 25,
                                "position": "Attacker",
                            }
                        ],
                    }
                ]
            }
        if endpoint == "coachs":
            team_id = int(request_params["team"])
            return {"response": [{"id": team_id * 100, "name": f"Coach {team_id}"}]}
        if endpoint == "fixtures" and "team" in request_params:
            team_id = int(request_params["team"])
            return {
                "response": [
                    {
                        "teams": {
                            "home": {"id": team_id, "name": "Team"},
                            "away": {"id": 99, "name": "Opponent"},
                        },
                        "goals": {"home": 2, "away": 0},
                    }
                ]
            }
        if endpoint == "players/profiles":
            player_id = int(request_params["player"])
            return {"response": [{"player": {"id": player_id, "name": f"Player {player_id}"}}]}
        if endpoint == "players":
            player_id = int(request_params["id"])
            league_id = 30 + player_id
            club_id = 100 + player_id
            return {
                "response": [
                    {
                        "player": {"id": player_id},
                        "statistics": [
                            {
                                "team": {"id": club_id, "name": f"Club {club_id}"},
                                "league": {"id": league_id, "name": f"League {league_id}"},
                            }
                        ],
                    }
                ]
            }
        if endpoint == "trophies":
            return {"response": []}
        if endpoint == "teams/statistics":
            return {
                "response": {
                    "team": {"id": request_params["team"], "name": "Club"},
                    "fixtures": {
                        "wins": {"total": 20},
                        "draws": {"total": 5},
                        "loses": {"total": 5},
                    },
                }
            }
        if endpoint == "leagues":
            league_id = int(request_params["id"])
            return {
                "response": [
                    {
                        "league": {"id": league_id, "name": f"League {league_id}"},
                        "country": {"name": "Country"},
                    }
                ]
            }
        if endpoint == "standings":
            return {"response": [{"league": {"standings": [[{"team": {"id": 1}}]]}}]}
        if endpoint == "fixtures" and request_params.get("league") != 1:
            return {"response": [{"fixture": {"id": 100}}]}
        if endpoint == "fixtures/lineups":
            return {
                "response": [
                    {
                        "team": {"id": 1, "name": "Alpha"},
                        "formation": "4-3-3",
                        "startXI": [{"player": {"id": 11, "name": "Player 11"}}],
                    }
                ]
            }
        if endpoint == "fixtures/events":
            return {
                "response": [
                    {
                        "team": {"id": 1, "name": "Alpha"},
                        "type": "subst",
                        "player": {"id": 12, "name": "Player 12"},
                    }
                ]
            }
        if endpoint == "fixtures/statistics":
            return {"response": [{"team": {"id": 1, "name": "Alpha"}, "statistics": []}]}
        raise AssertionError(f"Unexpected API call: {endpoint} {request_params}")


def test_fetch_world_cup_snapshot_writes_related_api_payloads(tmp_path: Path) -> None:
    api = FakeFootballApi()

    summary = fetch_world_cup_2026_snapshot(api, tmp_path)

    assert summary.national_teams == 2
    assert summary.players == 2
    assert summary.coaches == 2
    assert summary.clubs == 2
    assert summary.leagues == 2
    assert (tmp_path / "fixtures_world_cup.json").exists()
    assert (tmp_path / "standings_world_cup.json").exists()
    assert (tmp_path / "team_1_squad.json").exists()
    assert (tmp_path / "player_11_statistics.json").exists()
    stored = json.loads((tmp_path / "teams_world_cup.json").read_text(encoding="utf-8"))
    assert stored["response"][0]["team"]["name"] == "Alpha"
    assert ("players/squads", {"team": 1}) in api.calls


def test_fetch_world_cup_match_updates_refreshes_completed_tactical_payloads(
    tmp_path: Path,
) -> None:
    api = FakeFootballApi()

    summary = fetch_world_cup_2026_match_updates(
        api,
        tmp_path,
        completed_round_limit=1,
    )

    assert summary.fixtures == 2
    assert summary.standings_refreshed is True
    assert summary.tactical_fixtures == 1
    assert (tmp_path / "fixtures_world_cup.json").exists()
    assert (tmp_path / "standings_world_cup.json").exists()
    assert (tmp_path / "fixture_1_lineups.json").exists()
    assert (tmp_path / "fixture_1_events.json").exists()
    assert (tmp_path / "fixture_1_statistics.json").exists()
    assert ("fixtures/lineups", {"fixture": 1}) in api.calls


def test_fetch_world_cup_match_preview_updates_fetches_prior_and_target_payloads(
    tmp_path: Path,
) -> None:
    api = FakeFootballApi()

    summary = fetch_world_cup_2026_match_preview_updates(api, tmp_path, "wc-2026-2")

    assert summary.target_fixture_id == 2
    assert summary.target_status == "NS"
    assert summary.prior_completed_fixtures == 1
    assert (tmp_path / "fixture_1_lineups.json").exists()
    assert (tmp_path / "fixture_2_lineups.json").exists()
    assert ("fixtures/lineups", {"fixture": 2}) in api.calls
