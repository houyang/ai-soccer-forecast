"""Live FIFA World Cup catalog loading.

The loader is intentionally limited to schedule catalog data: match identity, teams,
kickoff, and venue. Rich prediction inputs such as form, injuries, odds, and weather
remain separate tools.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, cast
from urllib.request import Request, urlopen

from soccer.fixture_tools import FixtureCatalog
from soccer.models import (
    InjuryReport,
    MatchRequest,
    TeamForm,
    Venue,
    Weather,
)

DEFAULT_WORLD_CUP_SCHEDULE_URL = (
    "https://www.fifa.com/en/tournaments/mens/worldcup/"
    "canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums"
)
WORLD_CUP_COMPETITION_ID = "world-cup-2026"
WORLD_CUP_COMPETITION_NAME = "FIFA World Cup 2026"


class HttpClient(Protocol):
    def get_text(self, url: str) -> str:
        """Return response text for a URL."""


@dataclass(frozen=True)
class UrlLibHttpClient:
    timeout_seconds: float = 20.0

    def get_text(self, url: str) -> str:
        request = Request(url, headers={"User-Agent": "soccer-forecast/0.1"})
        with urlopen(request, timeout=self.timeout_seconds) as response:
            body = cast(bytes, response.read())
        return body.decode("utf-8")


def load_world_cup_catalog(
    source_url: str = DEFAULT_WORLD_CUP_SCHEDULE_URL,
    *,
    client: HttpClient | None = None,
) -> FixtureCatalog:
    """Fetch and parse a World Cup catalog from a live schedule source."""

    http_client = client or UrlLibHttpClient()
    payload = http_client.get_text(source_url)
    return catalog_from_payload(payload)


def catalog_from_payload(payload: str) -> FixtureCatalog:
    """Build a catalog from JSON or HTML containing embedded JSON schedule data."""

    data_items = _json_documents_from_payload(payload)
    entries = tuple(_match_from_item(item) for item in _find_match_items(data_items))
    if not entries:
        raise ValueError(
            "No World Cup matches found in live catalog payload. "
            "Point source_url at a JSON schedule feed or a rendered page that embeds match data."
        )

    requests = {match.match_id: match for match, _venue in entries}
    match_ids = tuple(
        match.match_id for match, _venue in sorted(entries, key=lambda item: item[0].kickoff)
    )
    teams = sorted(
        {match.home_team for match, _venue in entries}
        | {match.away_team for match, _venue in entries}
    )

    return FixtureCatalog(
        requests=requests,
        competitions={WORLD_CUP_COMPETITION_ID: match_ids},
        forms={team: TeamForm(team, 0, 0, 0, 0, 0, 0) for team in teams},
        injuries={
            team: InjuryReport(team=team, source="live-catalog-placeholder") for team in teams
        },
        head_to_heads={},
        venues={match.match_id: venue for match, venue in entries},
        weather={
            match.match_id: Weather(
                temperature_c=0.0,
                wind_kph=0.0,
                precipitation_mm=0.0,
                summary="weather unavailable from live catalog",
            )
            for match, _venue in entries
        },
        odds={},
    )


def _json_documents_from_payload(payload: str) -> tuple[object, ...]:
    stripped = payload.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return (json.loads(stripped),)

    documents: list[object] = []
    for script_body in re.findall(
        r"<script[^>]+type=[\"']application/json[\"'][^>]*>(.*?)</script>",
        payload,
        flags=re.IGNORECASE | re.DOTALL,
    ):
        try:
            documents.append(json.loads(html.unescape(script_body)))
        except json.JSONDecodeError:
            continue
    return tuple(documents)


def _find_match_items(documents: Iterable[object]) -> Iterable[dict[str, object]]:
    for document in documents:
        yield from _walk_for_match_items(document)


def _walk_for_match_items(value: object) -> Iterable[dict[str, object]]:
    if isinstance(value, dict):
        if _looks_like_match(value):
            yield value
        for child in value.values():
            yield from _walk_for_match_items(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_for_match_items(child)


def _looks_like_match(item: dict[str, object]) -> bool:
    return (
        _first_string(item, ("match_id", "matchId", "id", "matchNumber", "match_no")) is not None
        and _team_name(item, ("home_team", "homeTeam", "home", "team1")) is not None
        and _team_name(item, ("away_team", "awayTeam", "away", "team2")) is not None
        and _first_string(item, ("kickoff", "date", "datetime", "startTime", "utcDate")) is not None
    )


def _match_from_item(item: dict[str, object]) -> tuple[MatchRequest, Venue]:
    match_id = _required_string(item, ("match_id", "matchId", "id", "matchNumber", "match_no"))
    home_team = _required_team_name(item, ("home_team", "homeTeam", "home", "team1"))
    away_team = _required_team_name(item, ("away_team", "awayTeam", "away", "team2"))
    kickoff = _parse_datetime(
        _required_string(item, ("kickoff", "date", "datetime", "startTime", "utcDate"))
    )
    venue = _venue_from_item(item)
    return (
        MatchRequest(
            match_id=f"wc-2026-{_slug(match_id)}",
            competition=WORLD_CUP_COMPETITION_NAME,
            home_team=home_team,
            away_team=away_team,
            kickoff=kickoff,
            neutral_site=venue.home_team is None or venue.home_team != home_team,
        ),
        venue,
    )


def _venue_from_item(item: dict[str, object]) -> Venue:
    venue_value = item.get("venue") or item.get("stadium")
    if isinstance(venue_value, dict):
        name = _first_string(venue_value, ("name", "venueName", "stadiumName")) or "Unknown venue"
        city = _first_string(venue_value, ("city", "hostCity")) or "Unknown city"
        country = _first_string(venue_value, ("country", "countryName")) or "Unknown country"
        return Venue(name=name, city=city, country=country)

    name = _first_string(item, ("venue", "venueName", "stadium", "stadiumName")) or "Unknown venue"
    city = _first_string(item, ("city", "hostCity")) or "Unknown city"
    country = _first_string(item, ("country", "countryName")) or "Unknown country"
    return Venue(name=name, city=city, country=country)


def _required_string(item: dict[str, object], keys: tuple[str, ...]) -> str:
    value = _first_string(item, keys)
    if value is None:
        raise ValueError(f"Live catalog match is missing one of: {', '.join(keys)}")
    return value


def _first_string(item: dict[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _required_team_name(item: dict[str, object], keys: tuple[str, ...]) -> str:
    value = _team_name(item, keys)
    if value is None:
        raise ValueError(f"Live catalog match is missing team data for: {', '.join(keys)}")
    return value


def _team_name(item: dict[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            name = _first_string(value, ("name", "teamName", "country", "countryName"))
            if name is not None:
                return name
    return None


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("Live catalog match has an empty match id")
    return slug
