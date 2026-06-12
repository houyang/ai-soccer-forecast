"""Deterministic 0-100 rankings for the World Cup dataset.

Rankings are computed in dependency order so each tier is a pure function of the dataset
plus the already-computed lower tiers:

    league -> club -> player -> coach -> national team

Every weight is a named constant with a one-line rationale. Normalization is min-max
across the entities actually present, so scores are relative to this tournament's field.
Unknown/missing inputs fall back to neutral values rather than crashing or scoring zero.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from soccer.worldcup.entities import Player, WorldCup
from soccer.worldcup.reference import country_strength

# League weights: WC talent drawn from the league, gate receipts, country pedigree.
W_LEAGUE_PLAYERS = 0.45
W_LEAGUE_ATTENDANCE = 0.25
W_LEAGUE_COUNTRY = 0.30

# Club weights: division quality, WC talent, last-season results, honours.
W_CLUB_LEAGUE = 0.40
W_CLUB_PLAYERS = 0.25
W_CLUB_WINRATE = 0.25
W_CLUB_TITLES = 0.10

# Player weights: platform (club+league), output (goals), and match rating.
W_PLAYER_CLUB = 0.35
W_PLAYER_LEAGUE = 0.15
W_PLAYER_GOALS = 0.20
W_PLAYER_RATING = 0.30

# Coach weights: quality of the squad they command and their recent win rate.
W_COACH_SQUAD = 0.55
W_COACH_WINRATE = 0.45

# National-team weights: pedigree, squad quality, coach, recent form, domestic base.
W_TEAM_COUNTRY = 0.30
W_TEAM_SQUAD = 0.30
W_TEAM_COACH = 0.15
W_TEAM_FORM = 0.15
W_TEAM_DOMESTIC = 0.10
HOST_BONUS = 3.0  # added points for a host nation's static rating

_NEUTRAL = 50.0
_RATING_FLOOR = 6.0
_RATING_CEIL = 8.0
# Goals expected from a regular starter in a season, by position group.
_GOAL_EXPECTATION = {"Attacker": 18.0, "Midfielder": 9.0, "Defender": 4.0, "Goalkeeper": 1.0}
_SQUAD_CORE = 16  # count the strongest N players toward squad quality


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
    lo = min(values.values())
    hi = max(values.values())
    if hi - lo < 1e-9:
        return dict.fromkeys(values, 0.5)
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def _wc_players_per_league(wc: WorldCup) -> dict[int, int]:
    counts: dict[int, int] = dict.fromkeys(wc.leagues, 0)
    for player in wc.players.values():
        club = wc.clubs.get(player.club_id) if player.club_id is not None else None
        if club is not None and club.league_id in counts:
            counts[club.league_id] += 1
    return counts


def _wc_players_per_club(wc: WorldCup) -> dict[int, int]:
    counts: dict[int, int] = dict.fromkeys(wc.clubs, 0)
    for player in wc.players.values():
        if player.club_id in counts:
            counts[player.club_id] += 1
    return counts


def rank_leagues(wc: WorldCup) -> dict[int, float]:
    players = _wc_players_per_league(wc)
    norm_players = _minmax({lid: float(players.get(lid, 0)) for lid in wc.leagues})
    norm_att = _minmax({lid: lg.avg_attendance for lid, lg in wc.leagues.items()})
    out: dict[int, float] = {}
    for lid, league in wc.leagues.items():
        score = (
            W_LEAGUE_PLAYERS * norm_players.get(lid, 0.0)
            + W_LEAGUE_ATTENDANCE * norm_att.get(lid, 0.0)
            + W_LEAGUE_COUNTRY * (country_strength(league.country) / 100.0)
        )
        out[lid] = _clamp(score * 100.0)
    return out


def rank_clubs(wc: WorldCup, league_scores: dict[int, float]) -> dict[int, float]:
    wc_players = _wc_players_per_club(wc)
    norm_players = _minmax({cid: float(n) for cid, n in wc_players.items()})
    norm_titles = _minmax({cid: float(c.titles) for cid, c in wc.clubs.items()})
    out: dict[int, float] = {}
    for cid, club in wc.clubs.items():
        league_score = league_scores.get(club.league_id, _NEUTRAL) if club.league_id else _NEUTRAL
        score = (
            W_CLUB_LEAGUE * (league_score / 100.0)
            + W_CLUB_PLAYERS * norm_players.get(cid, 0.0)
            + W_CLUB_WINRATE * club.win_rate
            + W_CLUB_TITLES * norm_titles.get(cid, 0.0)
        )
        out[cid] = _clamp(score * 100.0)
    return out


def _goal_score(player: Player) -> float:
    expectation = _GOAL_EXPECTATION.get(player.position, 8.0)
    return min(player.goals / expectation, 1.0) if expectation else 0.0


def _rating_score(player: Player) -> float:
    if player.rating <= 0.0:  # missing rating -> neutral, not punished
        return 0.5
    return max(0.0, min((player.rating - _RATING_FLOOR) / (_RATING_CEIL - _RATING_FLOOR), 1.0))


def rank_players(
    wc: WorldCup, league_scores: dict[int, float], club_scores: dict[int, float]
) -> dict[int, float]:
    out: dict[int, float] = {}
    for pid, player in wc.players.items():
        club = wc.clubs.get(player.club_id) if player.club_id is not None else None
        if club is not None:
            club_score = club_scores.get(club.id, _NEUTRAL)
            league_score = (
                league_scores.get(club.league_id, _NEUTRAL)
                if club.league_id is not None
                else _NEUTRAL
            )
        else:
            club_score = _NEUTRAL
            league_score = _NEUTRAL
        score = (
            W_PLAYER_CLUB * (club_score / 100.0)
            + W_PLAYER_LEAGUE * (league_score / 100.0)
            + W_PLAYER_GOALS * _goal_score(player)
            + W_PLAYER_RATING * _rating_score(player)
        )
        out[pid] = _clamp(score * 100.0)
    return out


def _squad_quality(wc: WorldCup, team_id: int, player_scores: dict[int, float]) -> float:
    scores = sorted((player_scores.get(p.id, _NEUTRAL) for p in wc.squad(team_id)), reverse=True)
    return _mean(scores[:_SQUAD_CORE]) if scores else _NEUTRAL


def rank_coaches(wc: WorldCup, player_scores: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for cid, coach in wc.coaches.items():
        squad = _squad_quality(wc, coach.team_id, player_scores)
        score = W_COACH_SQUAD * (squad / 100.0) + W_COACH_WINRATE * coach.win_rate
        out[cid] = _clamp(score * 100.0)
    return out


def _domestic_strength(wc: WorldCup, team_id: int, league_scores: dict[int, float]) -> float:
    leagues: list[float] = []
    for player in wc.squad(team_id):
        club = wc.clubs.get(player.club_id) if player.club_id is not None else None
        if club and club.league_id:
            leagues.append(league_scores.get(club.league_id, _NEUTRAL))
    return _mean(leagues) if leagues else _NEUTRAL


def rank_teams(
    wc: WorldCup,
    league_scores: dict[int, float],
    player_scores: dict[int, float],
    coach_scores: dict[int, float],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for tid, team in wc.teams.items():
        squad = _squad_quality(wc, tid, player_scores)
        coach = coach_scores.get(team.coach_id, _NEUTRAL) if team.coach_id else _NEUTRAL
        domestic = _domestic_strength(wc, tid, league_scores)
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
