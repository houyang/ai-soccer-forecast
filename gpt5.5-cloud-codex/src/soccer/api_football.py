"""API-Football snapshot fetching.

The project keeps API access at the edge: this module fetches provider JSON and stores it
unchanged on disk. Prediction code reads those local snapshots so tests and model runs remain
deterministic after data collection.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
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


@dataclass(frozen=True)
class WorldCupMatchUpdateSummary:
    """Counts from a refreshed World Cup match-update snapshot fetch."""

    output_dir: Path
    fixtures: int
    standings_refreshed: bool
    tactical_fixtures: int
    files_written: int


@dataclass(frozen=True)
class WorldCupMatchPreviewUpdateSummary:
    """Counts from a refreshed single-match preview snapshot fetch."""

    output_dir: Path
    target_fixture_id: int
    target_status: str
    prior_completed_fixtures: int
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


def fetch_world_cup_2026_match_updates(
    client: FootballApiClient,
    output_dir: Path,
    *,
    world_cup_league_id: int = DEFAULT_WORLD_CUP_LEAGUE_ID,
    world_cup_season: int = DEFAULT_WORLD_CUP_SEASON,
    completed_round_limit: int | None = None,
    request_delay_seconds: float = 0.0,
) -> WorldCupMatchUpdateSummary:
    """Refresh World Cup fixtures, standings, and tactical snapshots.

    Unlike the full roster fetch, match updates are intentionally overwritten because
    fixture status, scores, standings, lineups, and substitutions change during the
    tournament. The function still stores raw provider JSON so model runs remain
    deterministic after refresh.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0

    fixtures = client.get_json(
        "fixtures",
        {"league": world_cup_league_id, "season": world_cup_season},
    )
    if request_delay_seconds > 0:
        time.sleep(request_delay_seconds)
    files_written += _write_snapshot(output_dir / "fixtures_world_cup.json", fixtures)

    standings = client.get_json(
        "standings",
        {"league": world_cup_league_id, "season": world_cup_season},
    )
    if request_delay_seconds > 0:
        time.sleep(request_delay_seconds)
    files_written += _write_snapshot(output_dir / "standings_world_cup.json", standings)

    tactical_fixture_ids = tuple(
        _completed_group_fixture_ids(fixtures, completed_round_limit=completed_round_limit)
    )
    for fixture_id in tactical_fixture_ids:
        for endpoint, suffix in (
            ("fixtures/lineups", "lineups"),
            ("fixtures/events", "events"),
            ("fixtures/statistics", "statistics"),
        ):
            payload = client.get_json(endpoint, {"fixture": fixture_id})
            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
            files_written += _write_snapshot(
                output_dir / f"fixture_{fixture_id}_{suffix}.json",
                payload,
            )

    return WorldCupMatchUpdateSummary(
        output_dir=output_dir,
        fixtures=len(_response_items(fixtures)),
        standings_refreshed=True,
        tactical_fixtures=len(tactical_fixture_ids),
        files_written=files_written,
    )


def fetch_world_cup_2026_match_preview_updates(
    client: FootballApiClient,
    output_dir: Path,
    match_id: str,
    *,
    world_cup_league_id: int = DEFAULT_WORLD_CUP_LEAGUE_ID,
    world_cup_season: int = DEFAULT_WORLD_CUP_SEASON,
    request_delay_seconds: float = 0.0,
) -> WorldCupMatchPreviewUpdateSummary:
    """Refresh the mutable snapshots needed to preview one not-started match."""

    output_dir.mkdir(parents=True, exist_ok=True)
    files_written = 0
    target_fixture_id = _fixture_id_from_match_id(match_id)

    fixtures = client.get_json(
        "fixtures",
        {"league": world_cup_league_id, "season": world_cup_season},
    )
    if request_delay_seconds > 0:
        time.sleep(request_delay_seconds)
    files_written += _write_snapshot(output_dir / "fixtures_world_cup.json", fixtures)

    target_fixture = _fixture_item_by_id(fixtures, target_fixture_id)
    if target_fixture is None:
        raise ValueError(f"World Cup fixture {match_id!r} was not found in provider fixtures")

    target_status = _fixture_status(target_fixture) or "unknown"
    if target_status not in {"NS", "TBD"}:
        raise ValueError(
            f"Fixture {match_id!r} has status {target_status!r}; "
            "single-match previews require a match that has not started"
        )

    standings = client.get_json(
        "standings",
        {"league": world_cup_league_id, "season": world_cup_season},
    )
    if request_delay_seconds > 0:
        time.sleep(request_delay_seconds)
    files_written += _write_snapshot(output_dir / "standings_world_cup.json", standings)

    target_kickoff = _fixture_datetime(target_fixture)
    target_team_ids = _fixture_team_ids(target_fixture)
    prior_fixture_ids = tuple(
        _completed_group_fixture_ids_before(
            fixtures,
            before=target_kickoff,
            team_ids=target_team_ids,
        )
    )
    tactical_fixture_ids = sorted({*prior_fixture_ids, target_fixture_id})
    for fixture_id in tactical_fixture_ids:
        for endpoint, suffix in (
            ("fixtures/lineups", "lineups"),
            ("fixtures/events", "events"),
            ("fixtures/statistics", "statistics"),
        ):
            payload = client.get_json(endpoint, {"fixture": fixture_id})
            if request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
            files_written += _write_snapshot(
                output_dir / f"fixture_{fixture_id}_{suffix}.json",
                payload,
            )

    return WorldCupMatchPreviewUpdateSummary(
        output_dir=output_dir,
        target_fixture_id=target_fixture_id,
        target_status=target_status,
        prior_completed_fixtures=len(prior_fixture_ids),
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


def _completed_group_fixture_ids(
    payload: JsonObject,
    *,
    completed_round_limit: int | None,
) -> Iterable[int]:
    completed_statuses = {"FT", "AET", "PEN"}
    for item in _response_items(payload):
        fixture = _mapping(item.get("fixture")) or item
        fixture_id = _int_value(fixture.get("id"))
        status = _mapping(fixture.get("status")) or {}
        if fixture_id is None or _first_text(status, ("short",)) not in completed_statuses:
            continue
        round_number = _round_number_from_fixture(item)
        if completed_round_limit is not None and (
            round_number is None or round_number > completed_round_limit
        ):
            continue
        if _round_name_from_fixture(item) is not None:
            yield fixture_id


def _completed_group_fixture_ids_before(
    payload: JsonObject,
    *,
    before: datetime | None,
    team_ids: set[int],
) -> Iterable[int]:
    completed_statuses = {"FT", "AET", "PEN"}
    for item in _response_items(payload):
        fixture = _mapping(item.get("fixture")) or item
        fixture_id = _int_value(fixture.get("id"))
        if fixture_id is None or _fixture_status(item) not in completed_statuses:
            continue
        kickoff = _fixture_datetime(item)
        if before is not None and kickoff is not None and kickoff >= before:
            continue
        if team_ids and not _fixture_team_ids(item).intersection(team_ids):
            continue
        if _round_name_from_fixture(item) is not None:
            yield fixture_id


def _fixture_id_from_match_id(match_id: str) -> int:
    raw_match_id = match_id.removeprefix("wc-2026-")
    fixture_id = _int_value(raw_match_id)
    if fixture_id is None:
        raise ValueError(f"World Cup match id {match_id!r} must include a numeric fixture id")
    return fixture_id


def _fixture_item_by_id(payload: JsonObject, fixture_id: int) -> JsonObject | None:
    for item in _response_items(payload):
        fixture = _mapping(item.get("fixture")) or item
        if _int_value(fixture.get("id")) == fixture_id:
            return item
    return None


def _fixture_status(item: JsonObject) -> str | None:
    fixture = _mapping(item.get("fixture")) or item
    status = _mapping(fixture.get("status")) or {}
    return _first_text(status, ("short",))


def _fixture_team_ids(item: JsonObject) -> set[int]:
    teams = _mapping(item.get("teams")) or {}
    team_ids: set[int] = set()
    for side in ("home", "away"):
        team = _mapping(teams.get(side))
        if team is None:
            continue
        team_id = _int_value(team.get("id"))
        if team_id is not None:
            team_ids.add(team_id)
    return team_ids


def _fixture_datetime(item: JsonObject) -> datetime | None:
    fixture = _mapping(item.get("fixture")) or item
    timestamp = _int_value(fixture.get("timestamp"))
    if timestamp is not None:
        return datetime.fromtimestamp(timestamp, UTC)

    value = _first_text(fixture, ("date",)) or _first_text(item, ("kickoff",))
    if value is None:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _round_number_from_fixture(item: JsonObject) -> int | None:
    round_name = _round_name_from_fixture(item)
    if round_name is None:
        return None
    for token in reversed(round_name.replace("_", " ").split()):
        if token.isdigit():
            return int(token)
    return None


def _round_name_from_fixture(item: JsonObject) -> str | None:
    league = _mapping(item.get("league")) or {}
    return _first_text(league, ("round",)) or _first_text(item, ("round", "stage"))


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


def _first_text(item: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return None
