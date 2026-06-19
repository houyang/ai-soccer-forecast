"""Typed entities for the FIFA 2026 World Cup dataset.

Every entity is a frozen dataclass with explicit ``to_dict``/``from_dict`` so the
normalized dataset round-trips through JSON without depending on dataclass internals.
Fields use plain JSON-friendly types (int/float/str/None and tuples of those).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class League:
    id: int
    name: str
    country: str
    n_teams: int
    matches_played: int
    avg_attendance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "country": self.country,
            "n_teams": self.n_teams,
            "matches_played": self.matches_played,
            "avg_attendance": self.avg_attendance,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> League:
        return cls(
            id=int(raw["id"]),
            name=raw["name"],
            country=raw["country"],
            n_teams=int(raw["n_teams"]),
            matches_played=int(raw["matches_played"]),
            avg_attendance=float(raw["avg_attendance"]),
        )


@dataclass(frozen=True)
class Club:
    id: int
    name: str
    country: str
    league_id: int | None
    wins: int
    draws: int
    losses: int
    titles: int

    @property
    def played(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.played if self.played else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "country": self.country,
            "league_id": self.league_id,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "titles": self.titles,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Club:
        return cls(
            id=int(raw["id"]),
            name=raw["name"],
            country=raw["country"],
            league_id=None if raw["league_id"] is None else int(raw["league_id"]),
            wins=int(raw["wins"]),
            draws=int(raw["draws"]),
            losses=int(raw["losses"]),
            titles=int(raw["titles"]),
        )


@dataclass(frozen=True)
class Player:
    id: int
    name: str
    age: int | None
    position: str
    club_id: int | None
    goals: int
    rating: float
    appearances: int
    wc_team_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "position": self.position,
            "club_id": self.club_id,
            "goals": self.goals,
            "rating": self.rating,
            "appearances": self.appearances,
            "wc_team_id": self.wc_team_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Player:
        return cls(
            id=int(raw["id"]),
            name=raw["name"],
            age=None if raw["age"] is None else int(raw["age"]),
            position=raw["position"],
            club_id=None if raw["club_id"] is None else int(raw["club_id"]),
            goals=int(raw["goals"]),
            rating=float(raw["rating"]),
            appearances=int(raw["appearances"]),
            wc_team_id=int(raw["wc_team_id"]),
        )


@dataclass(frozen=True)
class Coach:
    id: int
    name: str
    age: int | None
    wins: int
    draws: int
    losses: int
    titles: int
    team_id: int

    @property
    def played(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.played if self.played else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "titles": self.titles,
            "team_id": self.team_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Coach:
        return cls(
            id=int(raw["id"]),
            name=raw["name"],
            age=None if raw["age"] is None else int(raw["age"]),
            wins=int(raw["wins"]),
            draws=int(raw["draws"]),
            losses=int(raw["losses"]),
            titles=int(raw["titles"]),
            team_id=int(raw["team_id"]),
        )


@dataclass(frozen=True)
class NationalTeam:
    id: int
    name: str
    group: str
    confederation: str
    is_host: bool
    player_ids: tuple[int, ...]
    coach_id: int | None
    recent_w: int
    recent_d: int
    recent_l: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "group": self.group,
            "confederation": self.confederation,
            "is_host": self.is_host,
            "player_ids": list(self.player_ids),
            "coach_id": self.coach_id,
            "recent_w": self.recent_w,
            "recent_d": self.recent_d,
            "recent_l": self.recent_l,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> NationalTeam:
        return cls(
            id=int(raw["id"]),
            name=raw["name"],
            group=raw["group"],
            confederation=raw["confederation"],
            is_host=bool(raw["is_host"]),
            player_ids=tuple(int(x) for x in raw["player_ids"]),
            coach_id=None if raw["coach_id"] is None else int(raw["coach_id"]),
            recent_w=int(raw["recent_w"]),
            recent_d=int(raw["recent_d"]),
            recent_l=int(raw["recent_l"]),
        )


@dataclass(frozen=True)
class WcMatch:
    fixture_id: int
    matchday: int
    group: str
    home_id: int
    away_id: int
    kickoff: datetime
    venue: str
    home_goals: int | None
    away_goals: int | None

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "matchday": self.matchday,
            "group": self.group,
            "home_id": self.home_id,
            "away_id": self.away_id,
            "kickoff": self.kickoff.isoformat(),
            "venue": self.venue,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WcMatch:
        return cls(
            fixture_id=int(raw["fixture_id"]),
            matchday=int(raw["matchday"]),
            group=raw["group"],
            home_id=int(raw["home_id"]),
            away_id=int(raw["away_id"]),
            kickoff=datetime.fromisoformat(raw["kickoff"]),
            venue=raw["venue"],
            home_goals=None if raw["home_goals"] is None else int(raw["home_goals"]),
            away_goals=None if raw["away_goals"] is None else int(raw["away_goals"]),
        )


@dataclass(frozen=True)
class Lineup:
    fixture_id: int
    team_id: int
    formation: str
    start_ids: tuple[int, ...]
    sub_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "team_id": self.team_id,
            "formation": self.formation,
            "start_ids": list(self.start_ids),
            "sub_ids": list(self.sub_ids),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Lineup:
        return cls(
            fixture_id=int(raw["fixture_id"]),
            team_id=int(raw["team_id"]),
            formation=str(raw["formation"]),
            start_ids=tuple(int(x) for x in raw["start_ids"]),
            sub_ids=tuple(int(x) for x in raw["sub_ids"]),
        )


@dataclass(frozen=True)
class WorldCup:
    """Root of the normalized dataset. Entities are keyed by id for O(1) lookup."""

    leagues: dict[int, League] = field(default_factory=dict)
    clubs: dict[int, Club] = field(default_factory=dict)
    players: dict[int, Player] = field(default_factory=dict)
    coaches: dict[int, Coach] = field(default_factory=dict)
    teams: dict[int, NationalTeam] = field(default_factory=dict)
    matches: tuple[WcMatch, ...] = ()
    lineups: tuple[Lineup, ...] = ()

    def squad(self, team_id: int) -> list[Player]:
        team = self.teams[team_id]
        return [self.players[pid] for pid in team.player_ids if pid in self.players]

    def groups(self) -> dict[str, list[NationalTeam]]:
        out: dict[str, list[NationalTeam]] = {}
        for team in self.teams.values():
            out.setdefault(team.group, []).append(team)
        return {g: sorted(ts, key=lambda t: t.name) for g, ts in sorted(out.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "leagues": [v.to_dict() for v in self.leagues.values()],
            "clubs": [v.to_dict() for v in self.clubs.values()],
            "players": [v.to_dict() for v in self.players.values()],
            "coaches": [v.to_dict() for v in self.coaches.values()],
            "teams": [v.to_dict() for v in self.teams.values()],
            "matches": [m.to_dict() for m in self.matches],
            "lineups": [lu.to_dict() for lu in self.lineups],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WorldCup:
        return cls(
            leagues={x["id"]: League.from_dict(x) for x in raw.get("leagues", [])},
            clubs={x["id"]: Club.from_dict(x) for x in raw.get("clubs", [])},
            players={x["id"]: Player.from_dict(x) for x in raw.get("players", [])},
            coaches={x["id"]: Coach.from_dict(x) for x in raw.get("coaches", [])},
            teams={x["id"]: NationalTeam.from_dict(x) for x in raw.get("teams", [])},
            matches=tuple(WcMatch.from_dict(x) for x in raw.get("matches", [])),
            lineups=tuple(Lineup.from_dict(x) for x in raw.get("lineups", [])),
        )
