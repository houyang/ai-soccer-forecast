"""Assemble a single-match preview card: lineups, coaches, and a lineup-aware prediction.

Pure and offline. ``build_card`` projects each side's lineup, runs the lineup-aware forecast,
and packages everything (including a JSON-friendly ``to_dict``) for the PDF renderer and the
``wc card`` command.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from soccer.worldcup.entities import WorldCup
from soccer.worldcup.lineup import ProjectedLineup, project_lineup
from soccer.worldcup.predict import MatchPrediction, predict_one, top_scorelines
from soccer.worldcup.ranking import Rankings

_NEUTRAL = 50.0


@dataclass(frozen=True)
class PlayerLine:
    player_id: int
    name: str
    position: str
    rating: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "position": self.position,
            "rating": self.rating,
        }


@dataclass(frozen=True)
class TeamCard:
    team_id: int
    name: str
    coach_name: str | None
    coach_record: tuple[int, int, int] | None
    formation: str
    starters: tuple[PlayerLine, ...]
    subs: tuple[PlayerLine, ...]
    source: str
    source_matchday: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "name": self.name,
            "coach_name": self.coach_name,
            "coach_record": list(self.coach_record) if self.coach_record else None,
            "formation": self.formation,
            "starters": [p.to_dict() for p in self.starters],
            "subs": [p.to_dict() for p in self.subs],
            "source": self.source,
            "source_matchday": self.source_matchday,
        }


@dataclass(frozen=True)
class MatchCard:
    fixture_id: int
    group: str
    kickoff: datetime
    venue: str
    home: TeamCard
    away: TeamCard
    prediction: MatchPrediction
    top_scorelines: tuple[tuple[int, int, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id,
            "group": self.group,
            "kickoff": self.kickoff.isoformat(),
            "venue": self.venue,
            "home": self.home.to_dict(),
            "away": self.away.to_dict(),
            "prediction": self.prediction.to_dict(),
            "top_scorelines": [list(s) for s in self.top_scorelines],
        }


def _player_line(wc: WorldCup, rankings: Rankings, player_id: int) -> PlayerLine:
    player = wc.players.get(player_id)
    rating = round(rankings.players.get(player_id, _NEUTRAL), 1)
    if player is None:
        return PlayerLine(player_id=player_id, name=f"#{player_id}", position="?", rating=rating)
    return PlayerLine(
        player_id=player_id,
        name=player.name,
        position=player.position,
        rating=rating,
    )


def _team_card(wc: WorldCup, rankings: Rankings, team_id: int, lineup: ProjectedLineup) -> TeamCard:
    team = wc.teams[team_id]
    coach = wc.coaches.get(team.coach_id) if team.coach_id is not None else None
    coach_name = coach.name if coach else None
    coach_record = (coach.wins, coach.draws, coach.losses) if coach else None
    starters = tuple(_player_line(wc, rankings, pid) for pid in lineup.start_ids)
    subs = tuple(_player_line(wc, rankings, pid) for pid in lineup.sub_ids)
    return TeamCard(
        team_id=team_id,
        name=team.name,
        coach_name=coach_name,
        coach_record=coach_record,
        formation=lineup.formation,
        starters=starters,
        subs=subs,
        source=lineup.source,
        source_matchday=lineup.source_matchday,
    )


def build_card(wc: WorldCup, rankings: Rankings, fixture_id: int) -> MatchCard:
    match = next((m for m in wc.matches if m.fixture_id == fixture_id), None)
    if match is None:
        raise ValueError(f"fixture {fixture_id} not found in dataset")
    home_lineup = project_lineup(wc, rankings, match.home_id, fixture_id)
    away_lineup = project_lineup(wc, rankings, match.away_id, fixture_id)
    prediction = predict_one(wc, rankings, fixture_id, home_lineup, away_lineup)
    tops = tuple(top_scorelines(prediction.lambda_home, prediction.lambda_away, 3))
    return MatchCard(
        fixture_id=fixture_id,
        group=match.group,
        kickoff=match.kickoff,
        venue=match.venue,
        home=_team_card(wc, rankings, match.home_id, home_lineup),
        away=_team_card(wc, rankings, match.away_id, away_lineup),
        prediction=prediction,
        top_scorelines=tops,
    )
