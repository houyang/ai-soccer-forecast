# soccer_agent/worldcup/entities.py
"""Pydantic models for the cached FIFA 2026 World Cup dataset."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class League(BaseModel):
    id: int
    name: str
    country: str
    n_teams: int
    matches_played: int
    avg_attendance: float


class Club(BaseModel):
    id: int
    name: str
    country: str
    league_id: Optional[int] = None
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


class Player(BaseModel):
    id: int
    name: str
    age: Optional[int] = None
    position: str  # Goalkeeper | Defender | Midfielder | Attacker
    club_id: Optional[int] = None
    goals: int
    rating: float
    appearances: int
    wc_team_id: int


class Coach(BaseModel):
    id: int
    name: str
    age: Optional[int] = None
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


class NationalTeam(BaseModel):
    id: int
    name: str
    group: str
    confederation: str
    is_host: bool
    player_ids: tuple[int, ...] = Field(default_factory=tuple)
    coach_id: Optional[int] = None
    recent_w: int
    recent_d: int
    recent_l: int


class WcMatch(BaseModel):
    fixture_id: int
    matchday: int
    group: str
    home_id: int
    away_id: int
    kickoff: datetime
    venue: str
    home_goals: Optional[int] = None
    away_goals: Optional[int] = None
    round_name: str = ""

    @property
    def played(self) -> bool:
        return self.home_goals is not None and self.away_goals is not None


class Lineup(BaseModel):
    fixture_id: int
    team_id: int
    formation: str
    start_ids: tuple[int, ...] = Field(default_factory=tuple)
    sub_ids: tuple[int, ...] = Field(default_factory=tuple)


class WorldCup(BaseModel):
    leagues: dict[int, League] = Field(default_factory=dict)
    clubs: dict[int, Club] = Field(default_factory=dict)
    players: dict[int, Player] = Field(default_factory=dict)
    coaches: dict[int, Coach] = Field(default_factory=dict)
    teams: dict[int, NationalTeam] = Field(default_factory=dict)
    matches: list[WcMatch] = Field(default_factory=list)
    lineups: list[Lineup] = Field(default_factory=list)

    def squad(self, team_id: int) -> list[Player]:
        team = self.teams[team_id]
        return [self.players[pid] for pid in team.player_ids if pid in self.players]

    def groups(self) -> dict[str, list[NationalTeam]]:
        out: dict[str, list[NationalTeam]] = {}
        for team in self.teams.values():
            out.setdefault(team.group, []).append(team)
        return {g: sorted(ts, key=lambda t: t.name) for g, ts in sorted(out.items())}

    @classmethod
    def from_dict(cls, raw: dict) -> "WorldCup":
        return cls(
            leagues={x["id"]: League(**x) for x in raw.get("leagues", [])},
            clubs={x["id"]: Club(**x) for x in raw.get("clubs", [])},
            players={x["id"]: Player(**x) for x in raw.get("players", [])},
            coaches={x["id"]: Coach(**x) for x in raw.get("coaches", [])},
            teams={x["id"]: NationalTeam(**x) for x in raw.get("teams", [])},
            matches=[WcMatch(**x) for x in raw.get("matches", [])],
            lineups=[Lineup(**x) for x in raw.get("lineups", [])],
        )
