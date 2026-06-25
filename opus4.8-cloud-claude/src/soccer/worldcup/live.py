"""Incremental live refresh: merge actual results and lineups into an existing dataset.

This path never re-fetches the static entities (teams/players/clubs/coaches) -- it only
pulls fresh ``fixtures`` (to fill scorelines) and ``fixtures/lineups`` for finished matches,
so refreshing mid-tournament costs only a handful of API calls. The injected client matches
:class:`~soccer.worldcup.apifootball.ApiFootballClient`'s ``get`` signature, so tests pass a
fake with no network.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Protocol

from soccer.worldcup.entities import Lineup, WcMatch, WorldCup
from soccer.worldcup.ingest import WC_LEAGUE_ID, WC_SEASON, _safe_int

logger = logging.getLogger(__name__)

Json = dict[str, Any]
_FINISHED = {"FT", "AET", "PEN"}


class _Client(Protocol):
    def get(
        self, path: str, params: dict[str, Any] | None = ..., *, force_refresh: bool = ...
    ) -> list[Json]: ...


def _results_by_fixture(fixtures: list[Json]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for item in fixtures:
        fixture = item.get("fixture", {})
        if fixture.get("status", {}).get("short") not in _FINISHED:
            continue
        goals = item.get("goals", {})
        out[_safe_int(fixture.get("id"))] = (
            _safe_int(goals.get("home")),
            _safe_int(goals.get("away")),
        )
    return out


def _apply_results(
    matches: tuple[WcMatch, ...], results: dict[int, tuple[int, int]]
) -> tuple[WcMatch, ...]:
    updated: list[WcMatch] = []
    for match in matches:
        if match.fixture_id in results and not match.played:
            home_goals, away_goals = results[match.fixture_id]
            updated.append(replace(match, home_goals=home_goals, away_goals=away_goals))
        else:
            updated.append(match)
    return tuple(updated)


def _parse_lineups(fixture_id: int, blocks: list[Json]) -> list[Lineup]:
    out: list[Lineup] = []
    for block in blocks:
        team_id = _safe_int(block.get("team", {}).get("id"))
        start_ids = tuple(
            _safe_int(entry.get("player", {}).get("id")) for entry in block.get("startXI", [])
        )
        sub_ids = tuple(
            _safe_int(entry.get("player", {}).get("id")) for entry in block.get("substitutes", [])
        )
        out.append(
            Lineup(
                fixture_id=fixture_id,
                team_id=team_id,
                formation=str(block.get("formation") or ""),
                start_ids=start_ids,
                sub_ids=sub_ids,
            )
        )
    return out


def refresh_live(wc: WorldCup, client: _Client) -> WorldCup:
    """Return a copy of ``wc`` with results filled in and lineups attached."""
    fixtures = client.get(
        "fixtures", {"league": WC_LEAGUE_ID, "season": WC_SEASON}, force_refresh=True
    )
    matches = _apply_results(wc.matches, _results_by_fixture(fixtures))
    lineups: list[Lineup] = []
    for match in matches:
        if not match.played:
            continue
        blocks = client.get("fixtures/lineups", {"fixture": match.fixture_id})
        if not blocks:
            logger.warning("lineups unavailable for fixture %s", match.fixture_id)
            continue
        lineups.extend(_parse_lineups(match.fixture_id, blocks))
    return replace(wc, matches=matches, lineups=tuple(lineups))


def refresh_fixture(wc: WorldCup, client: _Client, fixture_id: int) -> WorldCup:
    """Merge a single fixture's latest result and lineup into ``wc``.

    Costs at most two API calls. Used by ``wc card --refresh`` to pick up an official lineup
    (or a finished scoreline) for the one match being previewed.
    """
    fixtures = client.get("fixtures", {"id": fixture_id}, force_refresh=True)
    matches = _apply_results(wc.matches, _results_by_fixture(fixtures))
    blocks = client.get("fixtures/lineups", {"fixture": fixture_id})
    new_lineups = _parse_lineups(fixture_id, blocks) if blocks else []
    kept = [lu for lu in wc.lineups if lu.fixture_id != fixture_id]
    return replace(wc, matches=matches, lineups=tuple(kept) + tuple(new_lineups))
