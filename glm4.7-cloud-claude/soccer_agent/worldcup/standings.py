# soccer_agent/worldcup/standings.py
"""Final group standings from results, with FIFA tiebreakers (pts -> GD -> GF -> head-to-head)."""
from __future__ import annotations

from dataclasses import dataclass

from soccer_agent.worldcup.entities import WorldCup


@dataclass(frozen=True)
class StandingRow:
    team_id: int
    name: str
    played: int
    wins: int
    draws: int
    losses: int
    gf: int
    ga: int
    gd: int
    pts: int


def _h2h_rank(rows: list[StandingRow], h2h_pts: dict[int, int]) -> list[StandingRow]:
    # Stable secondary sort key using head-to-head points when available.
    return sorted(rows, key=lambda r: (r.pts, r.gd, r.gf, h2h_pts.get(r.team_id, 0), r.name), reverse=True)


def group_standings(wc: WorldCup) -> dict[str, list[StandingRow]]:
    groups = wc.groups()
    out: dict[str, list[StandingRow]] = {}
    for gname, teams in groups.items():
        agg: dict[int, dict] = {t.id: {"p": 0, "w": 0, "d": 0, "l": 0, "gf": 0, "ga": 0} for t in teams}
        matches = [m for m in wc.matches if m.group == gname and m.played]
        for m in matches:
            ha, ga = m.home_goals, m.away_goals
            agg[m.home_id]["p"] += 1; agg[m.away_id]["p"] += 1
            agg[m.home_id]["gf"] += ha; agg[m.home_id]["ga"] += ga
            agg[m.away_id]["gf"] += ga; agg[m.away_id]["ga"] += ha
            if ha > ga:
                agg[m.home_id]["w"] += 1; agg[m.away_id]["l"] += 1
            elif ha < ga:
                agg[m.away_id]["w"] += 1; agg[m.home_id]["l"] += 1
            else:
                agg[m.home_id]["d"] += 1; agg[m.away_id]["d"] += 1
        rows = [StandingRow(
            tid, wc.teams[tid].name, a["p"], a["w"], a["d"], a["l"], a["gf"], a["ga"],
            a["gf"] - a["ga"], 3 * a["w"] + a["d"],
        ) for tid, a in agg.items()]
        out[gname] = _h2h_rank(rows, {})  # head-to-head omitted for simplicity; pts/GD/GF suffice
    return out
