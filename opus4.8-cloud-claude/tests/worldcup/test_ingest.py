from __future__ import annotations

from typing import Any

from soccer.worldcup.ingest import _parse_matches, ingest_world_cup


class FakeClient:
    """Stands in for ApiFootballClient, returning canned API-shaped payloads."""

    def get(self, path: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        params = params or {}
        if path == "teams":
            return [
                {"team": {"id": 1, "name": "England"}},
                {"team": {"id": 2, "name": "Mexico"}},
            ]
        if path == "standings" and params.get("league") == 1:
            return [
                {
                    "league": {
                        "standings": [
                            [
                                {"team": {"id": 1}, "group": "Group A"},
                                {"team": {"id": 2}, "group": "Group A"},
                            ],
                            # synthetic block that must be ignored
                            [{"team": {"id": 1}, "group": "Ranking of third-placed teams"}],
                        ]
                    }
                }
            ]
        if path == "standings":  # club league standings (season 2025)
            lid = params["league"]
            club_id = {10: 100, 20: 200}[lid]
            return [
                {
                    "league": {
                        "name": "Premier League" if lid == 10 else "Liga MX",
                        "country": "England" if lid == 10 else "Mexico",
                        "standings": [
                            [
                                {
                                    "team": {"id": club_id},
                                    "all": {"played": 30, "win": 20, "draw": 5, "lose": 5},
                                },
                                {
                                    "team": {"id": 999},
                                    "all": {"played": 30, "win": 5, "draw": 5, "lose": 20},
                                },
                            ]
                        ],
                    }
                }
            ]
        if path == "fixtures" and "team" in params:
            tid = params["team"]
            return [
                {
                    "fixture": {"status": {"short": "FT"}},
                    "teams": {"home": {"id": tid}, "away": {"id": 555}},
                    "goals": {"home": 2, "away": 0},
                },
                {
                    "fixture": {"status": {"short": "NS"}},  # unplayed, ignored
                    "teams": {"home": {"id": tid}, "away": {"id": 556}},
                    "goals": {"home": None, "away": None},
                },
            ]
        if path == "fixtures":
            return [
                {
                    "fixture": {
                        "id": 9001,
                        "date": "2026-06-12T19:00:00+00:00",
                        "venue": {"name": "MetLife", "city": "East Rutherford"},
                        "status": {"short": "NS"},
                    },
                    "league": {"round": "Group Stage - 1"},
                    "teams": {"home": {"id": 1}, "away": {"id": 2}},
                    "goals": {"home": None, "away": None},
                }
            ]
        if path == "coachs":
            tid = params["team"]
            return [{"id": 500 + tid, "name": f"Coach{tid}", "age": 55}]
        if path == "players/squads":
            tid = params["team"]
            pid = 10 + tid
            return [
                {"players": [{"id": pid, "name": f"P{pid}", "age": 26, "position": "Attacker"}]}
            ]
        if path == "players":
            pid = params["id"]
            league_id, league_name, country = (
                (10, "Premier League", "England") if pid == 11 else (20, "Liga MX", "Mexico")
            )
            club_id = 100 if pid == 11 else 200
            return [
                {
                    "statistics": [
                        {
                            "team": {"id": club_id, "name": f"Club{club_id}"},
                            "league": {"id": league_id, "name": league_name, "country": country},
                            "games": {"appearences": 30, "rating": "7.2"},
                            "goals": {"total": 12},
                        }
                    ]
                }
            ]
        raise AssertionError(f"unexpected path {path} {params}")


def test_ingest_builds_normalized_dataset() -> None:
    wc = ingest_world_cup(FakeClient())  # type: ignore[arg-type]

    assert set(wc.teams) == {1, 2}
    assert wc.teams[2].is_host is True
    assert wc.teams[1].is_host is False
    # synthetic standings block ignored: every team keeps its real group
    assert {t.group for t in wc.teams.values()} == {"Group A"}

    # one group-stage match, correctly grouped
    assert len(wc.matches) == 1
    assert wc.matches[0].group == "Group A"
    assert wc.matches[0].matchday == 1
    assert wc.matches[0].played is False

    # recent record derived from finished fixtures only (1 win, NS ignored)
    assert (wc.teams[1].recent_w, wc.teams[1].recent_d, wc.teams[1].recent_l) == (1, 0, 0)

    # players carry derived club + stats; coach record mirrors the team record
    assert wc.players[11].club_id == 100
    assert wc.players[11].goals == 12
    assert wc.players[11].rating == 7.2
    assert wc.coaches[501].wins == 1

    # clubs and leagues derived, with last-season record from club standings
    assert wc.clubs[100].wins == 20
    assert wc.clubs[100].league_id == 10
    assert wc.leagues[10].name == "Premier League"
    assert wc.leagues[10].n_teams == 2


def _fixture(fid: int, rnd: str, home: int, away: int, status: str) -> dict[str, Any]:
    return {
        "fixture": {
            "id": fid,
            "date": "2026-06-28T19:00:00+00:00",
            "venue": {"name": "SoFi Stadium", "city": None},
            "status": {"short": status},
        },
        "league": {"round": rnd},
        "teams": {"home": {"id": home}, "away": {"id": away}},
        "goals": {"home": None, "away": None},
    }


def test_parse_matches_keeps_knockout_round_name() -> None:
    fixtures = [
        _fixture(1, "Group Stage - 1", 10, 20, "FT"),
        _fixture(2, "Round of 32", 30, 40, "NS"),
    ]
    team_group = {10: "Group A", 20: "Group A"}
    matches = _parse_matches(fixtures, team_group)
    by_id = {m.fixture_id: m for m in matches}
    assert by_id[1].round_name == "Group Stage - 1"
    assert by_id[1].group == "Group A"
    assert by_id[2].round_name == "Round of 32"
    assert by_id[2].group == ""
    assert by_id[2].matchday == 0
