from __future__ import annotations

from typing import Any

from soccer.worldcup.entities import WorldCup
from soccer.worldcup.live import refresh_live


class FakeClient:
    """Stands in for ApiFootballClient: serves canned responses by (path, fixture)."""

    def __init__(self, fixtures: list[dict[str, Any]], lineups: dict[int, list[dict[str, Any]]]):
        self._fixtures = fixtures
        self._lineups = lineups
        self.forced: list[str] = []

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        *,
        force_refresh: bool = False,
    ) -> list[dict[str, Any]]:
        params = params or {}
        if path == "fixtures":
            if force_refresh:
                self.forced.append(path)
            return self._fixtures
        if path == "fixtures/lineups":
            return self._lineups.get(int(params["fixture"]), [])
        raise AssertionError(f"unexpected path {path}")


def _fixture(fid: int, home: int, away: int, status: str, hg: int, ag: int) -> dict[str, Any]:
    return {
        "fixture": {"id": fid, "status": {"short": status}},
        "teams": {"home": {"id": home}, "away": {"id": away}},
        "goals": {"home": hg, "away": ag},
    }


def _lineup_block(
    team: int, formation: str, starters: list[int], subs: list[int]
) -> dict[str, Any]:
    return {
        "team": {"id": team},
        "formation": formation,
        "startXI": [{"player": {"id": p}} for p in starters],
        "substitutes": [{"player": {"id": p}} for p in subs],
    }


def test_refresh_live_applies_results_and_lineups(sample_world_cup: WorldCup) -> None:
    # sample fixture 9001 is England(1) vs Mexico(2), matchday 1, currently unplayed.
    client = FakeClient(
        fixtures=[_fixture(9001, 1, 2, "FT", 2, 0)],
        lineups={
            9001: [
                _lineup_block(1, "4-3-3", [1, 2], []),
                _lineup_block(2, "5-4-1", [3, 4], []),
            ]
        },
    )
    updated = refresh_live(sample_world_cup, client)
    played = next(m for m in updated.matches if m.fixture_id == 9001)
    assert played.played and played.home_goals == 2 and played.away_goals == 0
    assert client.forced == ["fixtures"]  # fixtures pulled fresh, not from cache
    assert {lu.team_id for lu in updated.lineups} == {1, 2}
    eng = next(lu for lu in updated.lineups if lu.team_id == 1)
    assert eng.formation == "4-3-3" and eng.start_ids == (1, 2)


def test_refresh_live_skips_unfinished_and_tolerates_missing_lineups(
    sample_world_cup: WorldCup,
) -> None:
    client = FakeClient(fixtures=[_fixture(9001, 1, 2, "NS", 0, 0)], lineups={})
    updated = refresh_live(sample_world_cup, client)
    assert not any(m.played for m in updated.matches)
    assert updated.lineups == ()


def test_refresh_fixture_attaches_confirmed_lineup_pre_match(
    sample_world_cup: WorldCup,
) -> None:
    from soccer.worldcup.live import refresh_fixture

    # Pre-match: status not finished, but the official lineup is already published.
    client = FakeClient(
        fixtures=[_fixture(9001, 1, 2, "NS", 0, 0)],
        lineups={9001: [_lineup_block(1, "4-2-3-1", [1, 2], [3])]},
    )
    updated = refresh_fixture(sample_world_cup, client, 9001)
    assert not any(m.played for m in updated.matches)  # no result yet
    home = next(lu for lu in updated.lineups if lu.team_id == 1)
    assert home.fixture_id == 9001
    assert home.formation == "4-2-3-1"
    assert home.start_ids == (1, 2)


def test_refresh_fixture_fills_result_when_finished(sample_world_cup: WorldCup) -> None:
    from soccer.worldcup.live import refresh_fixture

    client = FakeClient(fixtures=[_fixture(9001, 1, 2, "FT", 3, 1)], lineups={})
    updated = refresh_fixture(sample_world_cup, client, 9001)
    played = next(m for m in updated.matches if m.fixture_id == 9001)
    assert played.played and played.home_goals == 3 and played.away_goals == 1
