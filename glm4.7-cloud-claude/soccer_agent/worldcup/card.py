# soccer_agent/worldcup/card.py
"""Assemble a single-match preview card: lineups, coaches, and a lineup-aware prediction."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.lineup import project_lineup
from soccer_agent.worldcup.predict import MatchPrediction, predict_one, top_scorelines
from soccer_agent.worldcup.ranking import Rankings

_NEUTRAL = 50.0


@dataclass(frozen=True)
class PlayerLine:
    player_id: int
    name: str
    position: str
    rating: float

    def to_dict(self) -> dict[str, Any]:
        return {"player_id": self.player_id, "name": self.name, "position": self.position, "rating": self.rating}


@dataclass(frozen=True)
class TeamCard:
    team_id: int
    name: str
    coach_name: Optional[str]
    coach_record: Optional[tuple[int, int, int]]
    formation: str
    starters: tuple[PlayerLine, ...]
    subs: tuple[PlayerLine, ...]
    source: str
    source_matchday: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id, "name": self.name,
            "coach_name": self.coach_name,
            "coach_record": list(self.coach_record) if self.coach_record else None,
            "formation": self.formation,
            "starters": [p.to_dict() for p in self.starters],
            "subs": [p.to_dict() for p in self.subs],
            "source": self.source, "source_matchday": self.source_matchday,
        }


@dataclass(frozen=True)
class MatchCard:
    fixture_id: Optional[int]
    group: str
    kickoff: Optional[datetime]
    venue: str
    home: TeamCard
    away: TeamCard
    prediction: MatchPrediction
    top_scorelines: tuple[tuple[int, int, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_id": self.fixture_id, "group": self.group,
            "kickoff": self.kickoff.isoformat() if self.kickoff else None,
            "venue": self.venue,
            "home": self.home.to_dict(), "away": self.away.to_dict(),
            "prediction": self.prediction.to_dict(),
            "top_scorelines": [list(s) for s in self.top_scorelines],
        }


def _player_line(wc: WorldCup, rankings: Rankings, pid: int) -> PlayerLine:
    p = wc.players.get(pid)
    rating = round(rankings.players.get(pid, _NEUTRAL), 1)
    if p is None:
        return PlayerLine(pid, f"#{pid}", "?", rating)
    return PlayerLine(pid, p.name, p.position, rating)


def _team_card(wc: WorldCup, rankings: Rankings, team_id: int, lineup) -> TeamCard:
    team = wc.teams[team_id]
    coach = wc.coaches.get(team.coach_id) if team.coach_id is not None else None
    return TeamCard(
        team_id=team_id, name=team.name,
        coach_name=coach.name if coach else None,
        coach_record=(coach.wins, coach.draws, coach.losses) if coach else None,
        formation=lineup.formation,
        starters=tuple(_player_line(wc, rankings, pid) for pid in lineup.start_ids),
        subs=tuple(_player_line(wc, rankings, pid) for pid in lineup.sub_ids),
        source=lineup.source, source_matchday=lineup.source_matchday,
    )


def build_card(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float],
    home_id: int, away_id: int, fetcher=None, fixture_id: int | None = None,
) -> MatchCard:
    # Prefer a real dataset fixture for kickoff/venue/group; else synthesize a neutral card.
    m = None
    if fixture_id is not None:
        m = next((x for x in wc.matches if x.fixture_id == fixture_id), None)
    if m is None:
        m = next((x for x in wc.matches if x.matchday == 0 and {x.home_id, x.away_id} == {home_id, away_id}), None)
    fid = m.fixture_id if m else None
    group = m.group if m and m.group else "Knockout"
    kickoff = m.kickoff if m else None
    venue = m.venue if m else "TBD"

    hlu = project_lineup(wc, rankings, home_id, fid or 0, fetcher)
    alu = project_lineup(wc, rankings, away_id, fid or 0, fetcher)

    if fid is not None:
        pred = predict_one(wc, rankings, strengths, fid, hlu, alu)
    else:
        # Synthesize a prediction with a transient fixture entry is complex; reuse an R32 fixture's
        # home/away by swapping ids is fragile. Instead, require a fixture for prediction.
        raise ValueError("build_card requires a fixture_id (use an R32 fixture)")

    tops = tuple(top_scorelines(pred.lambda_home, pred.lambda_away, 3))
    return MatchCard(
        fixture_id=fid, group=group, kickoff=kickoff, venue=venue,
        home=_team_card(wc, rankings, home_id, hlu),
        away=_team_card(wc, rankings, away_id, alu),
        prediction=pred, top_scorelines=tops,
    )
