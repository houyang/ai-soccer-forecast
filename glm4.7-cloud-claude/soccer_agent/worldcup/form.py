"""Recalibrate team strength using actual group-stage results.

The static ranking is a pre-tournament prior; this blends it with each team's real
group-stage goal difference (regressed toward neutral for the 3-match small sample).
"""
from __future__ import annotations

from dataclasses import dataclass

from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.ranking import Rankings

W_FORM = 0.45          # weight on group-stage performance
SHRINK_K = 3.0         # small-sample shrinkage (3 games played)
GD_PER_POINT = 4.0     # 1 group-stage GD point ~ 4 rating points at full weight
_NEUTRAL = 50.0


@dataclass(frozen=True)
class TeamForm:
    team_id: int
    played: int
    wins: int
    draws: int
    losses: int
    gf: int
    ga: int
    gd: int
    pts: int
    attack: float   # goals-per-game, regressed
    defense: float  # goals-conceded-per-game, regressed


def compute_forms(wc: WorldCup) -> dict[int, TeamForm]:
    agg: dict[int, dict] = {tid: {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0}
                            for tid in wc.teams}
    for m in wc.matches:
        if m.matchday not in (1, 2, 3) or not m.played:
            continue
        for home, away, gid in ((True, False, m.home_id), (False, True, m.away_id)):
            a = agg[gid]
            a["p"] += 1
            if home:
                a["gf"] += m.home_goals
                a["ga"] += m.away_goals
                if m.home_goals > m.away_goals:
                    a["w"] += 1
                elif m.home_goals < m.away_goals:
                    a["l"] += 1
                else:
                    a["d"] += 1
            else:
                a["gf"] += m.away_goals
                a["ga"] += m.home_goals
                if m.away_goals > m.home_goals:
                    a["w"] += 1
                elif m.away_goals < m.home_goals:
                    a["l"] += 1
                else:
                    a["d"] += 1

    out: dict[int, TeamForm] = {}
    for tid, a in agg.items():
        p = a["p"]
        gf, ga = a["gf"], a["ga"]
        gd = gf - ga
        pts = 3 * a["w"] + a["d"]
        # Regress per-game rates toward 1.5 scored / 1.5 conceded (tournament average-ish).
        shrink = p / (p + SHRINK_K)
        attack = (gf / p if p else 1.5) * shrink + 1.5 * (1 - shrink)
        defense = (ga / p if p else 1.5) * shrink + 1.5 * (1 - shrink)
        out[tid] = TeamForm(tid, p, a["w"], a["d"], a["l"], gf, ga, gd, pts, attack, defense)
    return out


def recalibrated_strength(wc: WorldCup, rankings: Rankings, forms: dict[int, TeamForm]) -> dict[int, float]:
    out: dict[int, float] = {}
    for tid, team in wc.teams.items():
        static = rankings.teams.get(tid, _NEUTRAL)
        f = forms.get(tid)
        if f is None or f.played == 0:
            out[tid] = static
            continue
        # Convert group-stage GD to a 0-100-ish performance score around neutral.
        perf = _NEUTRAL + f.gd * GD_PER_POINT
        perf = max(0.0, min(100.0, perf))
        out[tid] = (1 - W_FORM) * static + W_FORM * perf
    return out
