"""Pull the FIFA 2026 World Cup dataset from API-Football and normalize it.

Layered fetch in dependency order: teams -> groups -> fixtures -> squads -> coaches ->
recent national-team results -> per-player club stats -> derived clubs and leagues. Every
network call goes through :class:`ApiFootballClient`, so a cache makes the whole ingest
replayable for free. Per-player detail failures are non-fatal: the player keeps its
squad-level fields and the ranking layer falls back to neutral values.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from soccer.worldcup.apifootball import ApiFootballClient, ApiFootballError
from soccer.worldcup.entities import (
    Club,
    Coach,
    League,
    NationalTeam,
    Player,
    WcMatch,
    WorldCup,
)
from soccer.worldcup.reference import confederation, league_attendance

logger = logging.getLogger(__name__)

Json = dict[str, Any]

WC_LEAGUE_ID = 1
WC_SEASON = 2026
CLUB_SEASON = 2025
_FINISHED = {"FT", "AET", "PEN"}


def _matchday(round_name: str) -> int:
    # "Group Stage - 2" -> 2
    try:
        return int(round_name.rsplit("-", 1)[1].strip())
    except (IndexError, ValueError):
        return 1


def _safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | float | str):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    return default


def _group_by_team(standings_rows: list[Json]) -> dict[int, str]:
    # Keep only the real "Group A".."Group L" blocks; the API also returns a synthetic
    # "Ranking of third-placed teams" block whose rows would otherwise clobber the real
    # group of every team it lists.
    out: dict[int, str] = {}
    for block in standings_rows:
        for group in block.get("league", {}).get("standings", []):
            for row in group:
                name = str(row.get("group") or "")
                team_id = _safe_int(row.get("team", {}).get("id"), -1)
                if team_id >= 0 and name.startswith("Group "):
                    out[team_id] = name
    return out


def _parse_matches(fixtures: list[Json], team_group: dict[int, str]) -> list[WcMatch]:
    matches: list[WcMatch] = []
    for item in fixtures:
        fx = item.get("fixture", {})
        teams = item.get("teams", {})
        goals = item.get("goals", {})
        home_id = _safe_int(teams.get("home", {}).get("id"))
        away_id = _safe_int(teams.get("away", {}).get("id"))
        status = fx.get("status", {}).get("short", "")
        played = status in _FINISHED
        venue = fx.get("venue", {}) or {}
        venue_name = " / ".join(p for p in (venue.get("name"), venue.get("city")) if p)
        matches.append(
            WcMatch(
                fixture_id=_safe_int(fx.get("id")),
                matchday=_matchday(str(item.get("league", {}).get("round", ""))),
                group=team_group.get(home_id, ""),
                home_id=home_id,
                away_id=away_id,
                kickoff=datetime.fromisoformat(fx.get("date")),
                venue=venue_name,
                home_goals=_safe_int(goals.get("home")) if played else None,
                away_goals=_safe_int(goals.get("away")) if played else None,
            )
        )
    return matches


def _recent_record(client: ApiFootballClient, team_id: int) -> tuple[int, int, int]:
    try:
        fixtures = client.get("fixtures", {"team": team_id, "last": 15})
    except ApiFootballError:
        logger.warning("recent fixtures unavailable for team %s", team_id)
        return 0, 0, 0
    w = d = ll = 0
    for item in fixtures:
        if item.get("fixture", {}).get("status", {}).get("short") not in _FINISHED:
            continue
        teams = item.get("teams", {})
        is_home = _safe_int(teams.get("home", {}).get("id")) == team_id
        goals = item.get("goals", {})
        gf = _safe_int(goals.get("home") if is_home else goals.get("away"))
        ga = _safe_int(goals.get("away") if is_home else goals.get("home"))
        if gf > ga:
            w += 1
        elif gf == ga:
            d += 1
        else:
            ll += 1
    return w, d, ll


def _primary_stat(statistics: list[Json]) -> Json | None:
    best: Json | None = None
    best_apps = -1
    for stat in statistics:
        apps = _safe_int(stat.get("games", {}).get("appearences"))
        country = stat.get("league", {}).get("country")
        if country in (None, "World"):
            continue
        if apps > best_apps:
            best, best_apps = stat, apps
    return best


def _ingest_player(
    client: ApiFootballClient,
    raw_player: Json,
    team_id: int,
    clubs: dict[int, Json],
) -> Player:
    pid = _safe_int(raw_player.get("id"))
    name = str(raw_player.get("name") or f"player-{pid}")
    age = raw_player.get("age")
    position = str(raw_player.get("position") or "Unknown")
    goals = 0
    rating_sum = 0.0
    rating_apps = 0
    appearances = 0
    club_id: int | None = None
    try:
        detail = client.get("players", {"id": pid, "season": CLUB_SEASON})
    except ApiFootballError:
        logger.warning("player detail unavailable for %s (%s)", name, pid)
        detail = []
    if detail:
        statistics = detail[0].get("statistics", [])
        for stat in statistics:
            goals += _safe_int(stat.get("goals", {}).get("total"))
            apps = _safe_int(stat.get("games", {}).get("appearences"))
            rating = stat.get("games", {}).get("rating")
            if rating is not None and apps:
                rating_sum += float(rating) * apps
                rating_apps += apps
        primary = _primary_stat(statistics)
        if primary is not None:
            appearances = _safe_int(primary.get("games", {}).get("appearences"))
            club = primary.get("team", {})
            league = primary.get("league", {})
            club_id = _safe_int(club.get("id"))
            clubs.setdefault(
                club_id,
                {
                    "name": club.get("name"),
                    "country": league.get("country"),
                    "league_id": _safe_int(league.get("id")),
                    "league_name": league.get("name"),
                },
            )
    rating = round(rating_sum / rating_apps, 3) if rating_apps else 0.0
    return Player(
        id=pid,
        name=name,
        age=None if age is None else _safe_int(age),
        position=position,
        club_id=club_id,
        goals=goals,
        rating=rating,
        appearances=appearances,
        wc_team_id=team_id,
    )


def _ingest_coach(
    client: ApiFootballClient, team_id: int, record: tuple[int, int, int]
) -> Coach | None:
    try:
        coaches = client.get("coachs", {"team": team_id})
    except ApiFootballError:
        logger.warning("coach unavailable for team %s", team_id)
        return None
    if not coaches:
        return None
    raw = coaches[0]
    w, d, ll = record
    return Coach(
        id=_safe_int(raw.get("id")),
        name=str(raw.get("name") or "Unknown"),
        age=None if raw.get("age") is None else _safe_int(raw.get("age")),
        wins=w,
        draws=d,
        losses=ll,
        titles=0,
        team_id=team_id,
    )


def _build_leagues_and_clubs(
    client: ApiFootballClient,
    club_stubs: dict[int, Json],
) -> tuple[dict[int, League], dict[int, Club]]:
    # Fetch each distinct league's last-season standings once, then derive club records.
    league_ids = {s["league_id"] for s in club_stubs.values() if s.get("league_id")}
    standings_by_league: dict[int, dict[int, Json]] = {}
    league_meta: dict[int, Json] = {}
    for lid in sorted(league_ids):
        try:
            rows = client.get("standings", {"league": lid, "season": CLUB_SEASON})
        except ApiFootballError:
            logger.warning("standings unavailable for league %s", lid)
            continue
        if not rows:
            continue
        league_info = rows[0].get("league", {})
        flat: dict[int, Json] = {}
        for group in league_info.get("standings", []):
            for row in group:
                flat[_safe_int(row.get("team", {}).get("id"))] = row
        standings_by_league[lid] = flat
        league_meta[lid] = {
            "name": league_info.get("name"),
            "country": league_info.get("country"),
        }

    leagues: dict[int, League] = {}
    for lid, flat in standings_by_league.items():
        played = sum(_safe_int(r.get("all", {}).get("played")) for r in flat.values())
        name = str(league_meta[lid].get("name") or f"league-{lid}")
        leagues[lid] = League(
            id=lid,
            name=name,
            country=str(league_meta[lid].get("country") or "Unknown"),
            n_teams=len(flat),
            matches_played=played // 2,
            avg_attendance=float(league_attendance(name)),
        )

    clubs: dict[int, Club] = {}
    for cid, stub in club_stubs.items():
        lid = stub.get("league_id") or None
        row = standings_by_league.get(lid, {}).get(cid, {}) if lid else {}
        allrec = row.get("all", {}) if row else {}
        clubs[cid] = Club(
            id=cid,
            name=str(stub.get("name") or f"club-{cid}"),
            country=str(stub.get("country") or "Unknown"),
            league_id=lid if lid in leagues else None,
            wins=_safe_int(allrec.get("win")),
            draws=_safe_int(allrec.get("draw")),
            losses=_safe_int(allrec.get("lose")),
            titles=0,
        )
    return leagues, clubs


def ingest_world_cup(client: ApiFootballClient) -> WorldCup:
    logger.info("fetching World Cup %s structure", WC_SEASON)
    teams_raw = client.get("teams", {"league": WC_LEAGUE_ID, "season": WC_SEASON})
    standings = client.get("standings", {"league": WC_LEAGUE_ID, "season": WC_SEASON})
    fixtures = client.get("fixtures", {"league": WC_LEAGUE_ID, "season": WC_SEASON})

    team_group = _group_by_team(standings)
    matches = tuple(m for m in _parse_matches(fixtures, team_group) if m.group)

    players: dict[int, Player] = {}
    coaches: dict[int, Coach] = {}
    teams: dict[int, NationalTeam] = {}
    club_stubs: dict[int, Json] = {}

    total = len(teams_raw)
    for i, item in enumerate(teams_raw, start=1):
        team = item.get("team", {})
        team_id = _safe_int(team.get("id"))
        name = str(team.get("name") or f"team-{team_id}")
        logger.info("[%d/%d] %s", i, total, name)
        record = _recent_record(client, team_id)
        coach = _ingest_coach(client, team_id, record)
        if coach is not None:
            coaches[coach.id] = coach
        try:
            squads = client.get("players/squads", {"team": team_id})
        except ApiFootballError:
            logger.warning("squad unavailable for %s", name)
            squads = []
        squad_players = squads[0].get("players", []) if squads else []
        player_ids: list[int] = []
        for raw_player in squad_players:
            player = _ingest_player(client, raw_player, team_id, club_stubs)
            players[player.id] = player
            player_ids.append(player.id)
        teams[team_id] = NationalTeam(
            id=team_id,
            name=name,
            group=team_group.get(team_id, ""),
            confederation=confederation(name),
            is_host=name in {"USA", "Canada", "Mexico"},
            player_ids=tuple(player_ids),
            coach_id=None if coach is None else coach.id,
            recent_w=record[0],
            recent_d=record[1],
            recent_l=record[2],
        )

    logger.info("deriving %d clubs / leagues", len(club_stubs))
    leagues, clubs = _build_leagues_and_clubs(client, club_stubs)

    return WorldCup(
        leagues=leagues,
        clubs=clubs,
        players=players,
        coaches=coaches,
        teams=teams,
        matches=matches,
    )
