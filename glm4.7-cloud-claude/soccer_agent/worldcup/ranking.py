# soccer_agent/worldcup/ranking.py
"""Deterministic 0-100 rankings in dependency order: league -> club -> player -> coach -> team.

Each tier is a min-max-normalized blend of the fields below; unknowns fall back to neutral.
This is the STATIC (pre-knockout) rating; group-stage results recalibrate it in `form.py`.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from soccer_agent.worldcup.entities import Player, WorldCup
from soccer_agent.worldcup.reference import country_strength

W_LEAGUE_PLAYERS, W_LEAGUE_ATT, W_LEAGUE_COUNTRY = 0.45, 0.25, 0.30
W_CLUB_LEAGUE, W_CLUB_PLAYERS, W_CLUB_WINRATE, W_CLUB_TITLES = 0.40, 0.25, 0.25, 0.10
W_PLAYER_CLUB, W_PLAYER_LEAGUE, W_PLAYER_GOALS, W_PLAYER_RATING = 0.35, 0.15, 0.20, 0.30
W_COACH_SQUAD, W_COACH_WINRATE = 0.55, 0.45
W_TEAM_COUNTRY, W_TEAM_SQUAD, W_TEAM_COACH, W_TEAM_FORM, W_TEAM_DOMESTIC = 0.30, 0.30, 0.15, 0.15, 0.10
HOST_BONUS = 3.0
_NEUTRAL = 50.0
_RATING_FLOOR, _RATING_CEIL = 6.0, 8.0
_GOAL_EXPECT = {"Attacker": 18.0, "Midfielder": 9.0, "Defender": 4.0, "Goalkeeper": 1.0}
_SQUAD_CORE = 16


@dataclass(frozen=True)
class Rankings:
    leagues: dict[int, float] = field(default_factory=dict)
    clubs: dict[int, float] = field(default_factory=dict)
    players: dict[int, float] = field(default_factory=dict)
    coaches: dict[int, float] = field(default_factory=dict)
    teams: dict[int, float] = field(default_factory=dict)


def _minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi - lo < 1e-9:
        return dict.fromkeys(values, 0.5)
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _clamp(v: float) -> float:
    return max(0.0, min(100.0, v))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _wc_players_per_league(wc: WorldCup) -> dict[int, int]:
    counts = dict.fromkeys(wc.leagues, 0)
    for p in wc.players.values():
        club = wc.clubs.get(p.club_id) if p.club_id is not None else None
        if club is not None and club.league_id in counts:
            counts[club.league_id] += 1
    return counts


def _wc_players_per_club(wc: WorldCup) -> dict[int, int]:
    counts = dict.fromkeys(wc.clubs, 0)
    for p in wc.players.values():
        if p.club_id in counts:
            counts[p.club_id] += 1
    return counts


def rank_leagues(wc: WorldCup) -> dict[int, float]:
    players = _wc_players_per_league(wc)
    norm_p = _minmax({lid: float(n) for lid, n in players.items()})
    norm_att = _minmax({lid: lg.avg_attendance for lid, lg in wc.leagues.items()})
    out: dict[int, float] = {}
    for lid, lg in wc.leagues.items():
        score = (
            W_LEAGUE_PLAYERS * norm_p.get(lid, 0.0)
            + W_LEAGUE_ATT * norm_att.get(lid, 0.0)
            + W_LEAGUE_COUNTRY * (country_strength(lg.country) / 100.0)
        )
        out[lid] = _clamp(score * 100.0)
    return out


def rank_clubs(wc: WorldCup, league_scores: dict[int, float]) -> dict[int, float]:
    wc_players = _wc_players_per_club(wc)
    norm_p = _minmax({cid: float(n) for cid, n in wc_players.items()})
    norm_t = _minmax({cid: float(c.titles) for cid, c in wc.clubs.items()})
    out: dict[int, float] = {}
    for cid, c in wc.clubs.items():
        ls = league_scores.get(c.league_id, _NEUTRAL) if c.league_id else _NEUTRAL
        score = (
            W_CLUB_LEAGUE * (ls / 100.0)
            + W_CLUB_PLAYERS * norm_p.get(cid, 0.0)
            + W_CLUB_WINRATE * c.win_rate
            + W_CLUB_TITLES * norm_t.get(cid, 0.0)
        )
        out[cid] = _clamp(score * 100.0)
    return out


def _goal_score(p: Player) -> float:
    exp = _GOAL_EXPECT.get(p.position, 8.0)
    return min(p.goals / exp, 1.0) if exp else 0.0


def _rating_score(p: Player) -> float:
    if p.rating <= 0.0:
        return 0.5
    return max(0.0, min((p.rating - _RATING_FLOOR) / (_RATING_CEIL - _RATING_FLOOR), 1.0))


def rank_players(wc, league_scores, club_scores) -> dict[int, float]:
    out: dict[int, float] = {}
    for pid, p in wc.players.items():
        club = wc.clubs.get(p.club_id) if p.club_id is not None else None
        if club is not None:
            cs = club_scores.get(club.id, _NEUTRAL)
            ls = league_scores.get(club.league_id, _NEUTRAL) if club.league_id is not None else _NEUTRAL
        else:
            cs = ls = _NEUTRAL
        score = (
            W_PLAYER_CLUB * (cs / 100.0)
            + W_PLAYER_LEAGUE * (ls / 100.0)
            + W_PLAYER_GOALS * _goal_score(p)
            + W_PLAYER_RATING * _rating_score(p)
        )
        out[pid] = _clamp(score * 100.0)
    return out


def _squad_quality(wc, team_id, player_scores) -> float:
    scores = sorted((player_scores.get(p.id, _NEUTRAL) for p in wc.squad(team_id)), reverse=True)
    return _mean(scores[:_SQUAD_CORE]) if scores else _NEUTRAL


def rank_coaches(wc, player_scores) -> dict[int, float]:
    out: dict[int, float] = {}
    for cid, coach in wc.coaches.items():
        squad = _squad_quality(wc, coach.team_id, player_scores)
        out[cid] = _clamp((W_COACH_SQUAD * (squad / 100.0) + W_COACH_WINRATE * coach.win_rate) * 100.0)
    return out


def _domestic(wc, team_id, league_scores) -> float:
    vals: list[float] = []
    for p in wc.squad(team_id):
        club = wc.clubs.get(p.club_id) if p.club_id is not None else None
        if club and club.league_id:
            vals.append(league_scores.get(club.league_id, _NEUTRAL))
    return _mean(vals) if vals else _NEUTRAL


def rank_teams(wc, league_scores, player_scores, coach_scores) -> dict[int, float]:
    out: dict[int, float] = {}
    for tid, team in wc.teams.items():
        squad = _squad_quality(wc, tid, player_scores)
        coach = coach_scores.get(team.coach_id, _NEUTRAL) if team.coach_id else _NEUTRAL
        domestic = _domestic(wc, tid, league_scores)
        played = team.recent_w + team.recent_d + team.recent_l
        form = (team.recent_w + 0.5 * team.recent_d) / played if played else 0.5
        score = (
            W_TEAM_COUNTRY * (country_strength(team.name) / 100.0)
            + W_TEAM_SQUAD * (squad / 100.0)
            + W_TEAM_COACH * (coach / 100.0)
            + W_TEAM_FORM * form
            + W_TEAM_DOMESTIC * (domestic / 100.0)
        ) * 100.0
        if team.is_host:
            score += HOST_BONUS
        out[tid] = _clamp(score)
    return out


def rank_all(wc: WorldCup) -> Rankings:
    leagues = rank_leagues(wc)
    clubs = rank_clubs(wc, leagues)
    players = rank_players(wc, leagues, clubs)
    coaches = rank_coaches(wc, players)
    teams = rank_teams(wc, leagues, players, coaches)
    return Rankings(leagues=leagues, clubs=clubs, players=players, coaches=coaches, teams=teams)


def top_n(scores: dict[int, float], n: int) -> list[tuple[int, float]]:
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
