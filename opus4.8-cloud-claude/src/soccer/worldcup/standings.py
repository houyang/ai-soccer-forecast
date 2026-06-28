"""Compute final group standings from played group matches.

The API's ``standings.rank`` field is unreliable for this dataset (it lists
fourth-placed teams as advancing), so ranks are derived here from match goals
using the FIFA tiebreaker order: points, goal difference, goals for, then
head-to-head among the teams still tied, then team id as a deterministic
stand-in for the drawing of lots.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

from soccer.worldcup.entities import WcMatch, WorldCup


@dataclass(frozen=True)
class StandingRow:
    team_id: int
    group: str
    played: int
    won: int
    drawn: int
    lost: int
    gf: int
    ga: int
    points: int
    rank: int = 0

    @property
    def gd(self) -> int:
        return self.gf - self.ga


def _group_matches(wc: WorldCup) -> dict[str, list[WcMatch]]:
    out: dict[str, list[WcMatch]] = defaultdict(list)
    for m in wc.matches:
        if m.group and m.played:
            out[m.group].append(m)
    return out


def _raw_rows(group: str, matches: list[WcMatch]) -> dict[int, StandingRow]:
    acc: dict[int, dict[str, int]] = defaultdict(
        lambda: {"played": 0, "won": 0, "drawn": 0, "lost": 0, "gf": 0, "ga": 0, "points": 0}
    )
    for m in matches:
        assert m.home_goals is not None and m.away_goals is not None
        hg, ag = m.home_goals, m.away_goals
        for tid, gf, ga in ((m.home_id, hg, ag), (m.away_id, ag, hg)):
            r = acc[tid]
            r["played"] += 1
            r["gf"] += gf
            r["ga"] += ga
        if hg > ag:
            acc[m.home_id]["won"] += 1
            acc[m.home_id]["points"] += 3
            acc[m.away_id]["lost"] += 1
        elif hg < ag:
            acc[m.away_id]["won"] += 1
            acc[m.away_id]["points"] += 3
            acc[m.home_id]["lost"] += 1
        else:
            acc[m.home_id]["drawn"] += 1
            acc[m.away_id]["drawn"] += 1
            acc[m.home_id]["points"] += 1
            acc[m.away_id]["points"] += 1
    return {tid: StandingRow(team_id=tid, group=group, **vals) for tid, vals in acc.items()}


def _head_to_head_key(tied: list[int], matches: list[WcMatch]) -> dict[int, tuple[int, int, int]]:
    pts: dict[int, int] = defaultdict(int)
    gf: dict[int, int] = defaultdict(int)
    ga: dict[int, int] = defaultdict(int)
    members = set(tied)
    for m in matches:
        if m.home_id not in members or m.away_id not in members:
            continue
        assert m.home_goals is not None and m.away_goals is not None
        hg, ag = m.home_goals, m.away_goals
        gf[m.home_id] += hg
        ga[m.home_id] += ag
        gf[m.away_id] += ag
        ga[m.away_id] += hg
        if hg > ag:
            pts[m.home_id] += 3
        elif hg < ag:
            pts[m.away_id] += 3
        else:
            pts[m.home_id] += 1
            pts[m.away_id] += 1
    return {tid: (pts[tid], gf[tid] - ga[tid], gf[tid]) for tid in tied}


def _rank_rows(rows: list[StandingRow], matches: list[WcMatch]) -> list[StandingRow]:
    def primary(r: StandingRow) -> tuple[int, int, int]:
        return (r.points, r.gd, r.gf)

    ordered = sorted(rows, key=primary, reverse=True)
    resolved: list[StandingRow] = []
    i = 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and primary(ordered[j]) == primary(ordered[i]):
            j += 1
        cluster = ordered[i:j]
        if len(cluster) > 1:
            h2h = _head_to_head_key([r.team_id for r in cluster], matches)
            cluster.sort(key=lambda r: (h2h[r.team_id], -r.team_id), reverse=True)
        resolved.extend(cluster)
        i = j
    return [replace(r, rank=k) for k, r in enumerate(resolved, start=1)]


def group_tables(wc: WorldCup) -> dict[str, list[StandingRow]]:
    by_group = _group_matches(wc)
    out: dict[str, list[StandingRow]] = {}
    for group in sorted(by_group):
        matches = by_group[group]
        rows = list(_raw_rows(group, matches).values())
        out[group] = _rank_rows(rows, matches)
    return out


def team_labels(wc: WorldCup) -> dict[int, str]:
    labels: dict[int, str] = {}
    for group, rows in group_tables(wc).items():
        letter = group.split()[-1]
        for row in rows:
            labels[row.team_id] = f"{row.rank}{letter}"
    return labels
