"""API-Football snapshot fetching.

The project keeps API access at the edge: this module fetches provider JSON and stores it
unchanged on disk. Prediction code reads those local snapshots so tests and model runs remain
deterministic after data collection.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlencode
from urllib.request import Request, urlopen

JsonObject = dict[str, object]
ApiParam = str | int | float | bool

DEFAULT_API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"
DEFAULT_WORLD_CUP_LEAGUE_ID = 1
DEFAULT_WORLD_CUP_SEASON = 2026
DEFAULT_CLUB_SEASON = 2025


class FootballApiClient(Protocol):
    """Small interface for API-Football calls."""

    def get_json(self, endpoint: str, params: Mapping[str, ApiParam]) -> JsonObject:
        """Return decoded JSON for an endpoint and query parameters."""


@dataclass(frozen=True)
class ApiFootballClient:
    """HTTP client for the direct API-Sports API-Football host."""

    api_key: str
    base_url: str = DEFAULT_API_FOOTBALL_BASE_URL
    timeout_seconds: float = 30.0

    def get_json(self, endpoint: str, params: Mapping[str, ApiParam]) -> JsonObject:
        clean_endpoint = endpoint.strip("/")
        query = urlencode({key: str(value) for key, value in sorted(params.items())})
        url = f"{self.base_url.rstrip('/')}/{clean_endpoint}"
        if query:
            url = f"{url}?{query}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "soccer-forecast/0.1",
                "x-apisports-key": self.api_key,
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            payload = cast(bytes, response.read()).decode("utf-8")
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise TypeError(f"API-Football response for {endpoint!r} was not a JSON object")
        result = cast(JsonObject, decoded)
        errors = result.get("errors")
        if errors not in ({}, [], None):
            raise RuntimeError(f"API-Football returned errors for {endpoint!r}: {errors!r}")
        return result


@dataclass(frozen=True)
class WorldCupSnapshotSummary:
    """Counts from a completed snapshot fetch."""

    output_dir: Path
    national_teams: int
    players: int
    coaches: int
    clubs: int
    leagues: int
    files_written: int


def fetch_world_cup_2026_snapshot(
    client: FootballApiClient,
    output_dir: Path,
    *,
    world_cup_league_id: int = DEFAULT_WORLD_CUP_LEAGUE_ID,
    world_cup_season: int = DEFAULT_WORLD_CUP_SEASON,
    club_season: int = DEFAULT_CLUB_SEASON,
    recent_fixture_count: int = 20,
    request_delay_seconds: float = 0.0,
) -> WorldCupSnapshotSummary:
    """Fetch World Cup 2026 data and related player, club, and league snapshots.

    API-Football is used as a source of raw data only. The snapshot intentionally stores
    provider responses without translating them so the ranking layer can be re-run without
    another network call.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0

    fixtures, wrote = _get_or_fetch_snapshot(
        client,
        output_dir / "fixtures_world_cup.json",
        "fixtures",
        {"league": world_cup_league_id, "season": world_cup_season},
        request_delay_seconds,
    )
    files_written += wrote

    teams_payload, wrote = _get_or_fetch_snapshot(
        client,
        output_dir / "teams_world_cup.json",
        "teams",
        {"league": world_cup_league_id, "season": world_cup_season},
        request_delay_seconds,
    )
    files_written += wrote

    _standings_payload, wrote = _get_or_fetch_snapshot(
        client,
        output_dir / "standings_world_cup.json",
        "standings",
        {"league": world_cup_league_id, "season": world_cup_season},
        request_delay_seconds,
    )
    files_written += wrote

    team_ids = _team_ids_from_payloads(fixtures, teams_payload)
    player_ids: set[int] = set()
    coach_ids: set[int] = set()
    club_league_pairs: set[tuple[int, int]] = set()

    for team_id in sorted(team_ids):
        squad, wrote = _get_or_fetch_snapshot(
            client,
            output_dir / f"team_{team_id}_squad.json",
            "players/squads",
            {"team": team_id},
            request_delay_seconds,
        )
        files_written += wrote
        player_ids.update(_player_ids_from_squad(squad))

        coaches, wrote = _get_or_fetch_snapshot(
            client,
            output_dir / f"team_{team_id}_coaches.json",
            "coachs",
            {"team": team_id},
            request_delay_seconds,
        )
        files_written += wrote
        coach_ids.update(_coach_ids_from_payload(coaches))

        _recent_fixtures, wrote = _get_or_fetch_snapshot(
            client,
            output_dir / f"team_{team_id}_recent_fixtures.json",
            "fixtures",
            {"team": team_id, "last": recent_fixture_count},
            request_delay_seconds,
        )
        files_written += wrote

    for player_id in sorted(player_ids):
        statistics, wrote = _get_or_fetch_snapshot(
            client,
            output_dir / f"player_{player_id}_statistics.json",
            "players",
            {"id": player_id, "season": club_season},
            request_delay_seconds,
        )
        files_written += wrote
        club_league_pairs.update(_club_league_pairs_from_player_statistics(statistics))

    for coach_id in sorted(coach_ids):
        _trophies, wrote = _get_or_fetch_snapshot(
            client,
            output_dir / f"coach_{coach_id}_trophies.json",
            "trophies",
            {"coach": coach_id},
            request_delay_seconds,
        )
        files_written += wrote

    seen_leagues: set[int] = set()
    seen_clubs: set[int] = set()
    for club_id, league_id in sorted(club_league_pairs):
        seen_clubs.add(club_id)
        seen_leagues.add(league_id)
        _statistics, wrote = _get_or_fetch_snapshot(
            client,
            output_dir / f"club_{club_id}_statistics_league_{league_id}.json",
            "teams/statistics",
            {"team": club_id, "league": league_id, "season": club_season},
            request_delay_seconds,
        )
        files_written += wrote

    for league_id in sorted(seen_leagues):
        _league, wrote = _get_or_fetch_snapshot(
            client,
            output_dir / f"league_{league_id}.json",
            "leagues",
            {"id": league_id, "season": club_season},
            request_delay_seconds,
        )
        files_written += wrote

        _standings, wrote = _get_or_fetch_snapshot(
            client,
            output_dir / f"league_{league_id}_standings.json",
            "standings",
            {"league": league_id, "season": club_season},
            request_delay_seconds,
        )
        files_written += wrote

        _league_fixtures, wrote = _get_or_fetch_snapshot(
            client,
            output_dir / f"league_{league_id}_fixtures.json",
            "fixtures",
            {"league": league_id, "season": club_season},
            request_delay_seconds,
        )
        files_written += wrote

    return WorldCupSnapshotSummary(
        output_dir=output_dir,
        national_teams=len(team_ids),
        players=len(player_ids),
        coaches=len(coach_ids),
        clubs=len(seen_clubs),
        leagues=len(seen_leagues),
        files_written=files_written,
    )


def _write_snapshot(path: Path, payload: JsonObject) -> int:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1


def _get_or_fetch_snapshot(
    client: FootballApiClient,
    path: Path,
    endpoint: str,
    params: Mapping[str, ApiParam],
    request_delay_seconds: float,
) -> tuple[JsonObject, int]:
    if path.exists():
        return _read_snapshot(path), 0
    payload = client.get_json(endpoint, params)
    if request_delay_seconds > 0:
        time.sleep(request_delay_seconds)
    return payload, _write_snapshot(path, payload)


def _read_snapshot(path: Path) -> JsonObject:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(JsonObject, decoded)


def _team_ids_from_payloads(*payloads: JsonObject) -> set[int]:
    team_ids: set[int] = set()
    for payload in payloads:
        for item in _response_items(payload):
            team = _mapping(item.get("team"))
            if team:
                team_id = _int_value(team.get("id"))
                if team_id is not None:
                    team_ids.add(team_id)
            teams = _mapping(item.get("teams"))
            if teams:
                for side in ("home", "away"):
                    side_team = _mapping(teams.get(side))
                    if side_team:
                        team_id = _int_value(side_team.get("id"))
                        if team_id is not None:
                            team_ids.add(team_id)
    return team_ids


def _player_ids_from_squad(payload: JsonObject) -> set[int]:
    player_ids: set[int] = set()
    for item in _response_items(payload):
        players = item.get("players")
        if not isinstance(players, list):
            continue
        for player in players:
            player_item = _mapping(player)
            if not player_item:
                continue
            player_id = _int_value(player_item.get("id"))
            if player_id is not None:
                player_ids.add(player_id)
    return player_ids


def _coach_ids_from_payload(payload: JsonObject) -> set[int]:
    coach_ids: set[int] = set()
    for item in _response_items(payload):
        coach_id = _int_value(item.get("id"))
        if coach_id is not None:
            coach_ids.add(coach_id)
    return coach_ids


def _club_league_pairs_from_player_statistics(payload: JsonObject) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for item in _response_items(payload):
        statistics = item.get("statistics")
        if not isinstance(statistics, list):
            continue
        for statistic in statistics:
            stat_item = _mapping(statistic)
            if not stat_item:
                continue
            team = _mapping(stat_item.get("team"))
            league = _mapping(stat_item.get("league"))
            if not team or not league:
                continue
            club_id = _int_value(team.get("id"))
            league_id = _int_value(league.get("id"))
            if club_id is not None and league_id is not None:
                pairs.add((club_id, league_id))
    return pairs


def _response_items(payload: JsonObject) -> tuple[JsonObject, ...]:
    response = payload.get("response", payload)
    if isinstance(response, list):
        return tuple(item for item in (_mapping(value) for value in response) if item)
    if isinstance(response, dict):
        return (cast(JsonObject, response),)
    return ()


def _mapping(value: object) -> JsonObject | None:
    if isinstance(value, dict):
        return cast(JsonObject, value)
    return None


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return None
