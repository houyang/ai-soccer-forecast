# soccer_agent/worldcup/lineup.py
"""Project each team's formation, starting XI, and subs for an upcoming match.

Precedence: a live-fetched recent lineup ("live") > an attached dataset lineup from an
earlier matchday ("prior") > a curated-formation + squad-rating projection ("projected").
"""
from __future__ import annotations

from dataclasses import dataclass

from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.ranking import Rankings

DEFAULT_FORMATION = "4-3-3"
SUB_COUNT = 7
_NEUTRAL = 50.0
_POS_GROUP = {"Goalkeeper": "GK", "Defender": "DEF", "Midfielder": "MID", "Attacker": "FWD"}

# Curated real coach formations (override the squad-derived default where well-known).
FORMATIONS: dict[str, str] = {
    "Argentina": "4-3-3", "France": "4-2-3-1", "Brazil": "4-3-3", "England": "4-2-3-1",
    "Spain": "4-3-3", "Germany": "4-2-3-1", "Portugal": "4-3-3", "Netherlands": "3-4-3",
    "Belgium": "3-4-3", "Croatia": "4-3-3", "Mexico": "4-1-4-1", "USA": "4-2-3-1",
    "Norway": "4-4-2", "Morocco": "4-3-3", "Japan": "4-2-3-1", "Colombia": "4-2-3-1",
    "Uruguay": "4-3-3", "Switzerland": "3-4-2-1", "Senegal": "4-3-3", "Ecuador": "4-2-3-1",
    "Australia": "4-4-2", "South Korea": "4-2-3-1", "Iran": "4-3-3", "Saudi Arabia": "4-2-3-1",
    "Canada": "4-4-2", "Türkiye": "4-2-3-1", "Austria": "4-2-3-1", "Sweden": "4-3-3",
    "Czech Republic": "4-2-3-1", "Ivory Coast": "4-3-3", "Egypt": "4-2-3-1",
    "Scotland": "5-3-2", "Paraguay": "4-3-3", "Ghana": "4-2-3-1", "Tunisia": "4-3-3",
    "Algeria": "4-2-3-1", "Congo DR": "4-3-3", "Iraq": "4-3-3", "Jordan": "4-4-2",
    "Qatar": "3-5-2", "Uzbekistan": "4-2-3-1", "Panama": "4-4-2", "Cape Verde Islands": "4-3-3",
    "Curaçao": "4-3-3", "Haiti": "4-4-2", "New Zealand": "4-4-2", "Bosnia & Herzegovina": "4-2-3-1",
    "South Africa": "4-3-3",
}


@dataclass(frozen=True)
class ProjectedLineup:
    team_id: int
    formation: str
    start_ids: tuple[int, ...]
    sub_ids: tuple[int, ...]
    source: str  # "live" | "prior" | "projected"
    source_matchday: int | None = None


def formation_slots(formation: str) -> tuple[int, int, int]:
    """(defenders, midfielders, forwards) from e.g. '4-2-3-1'. Falls back to 4-3-3."""
    try:
        nums = [int(part) for part in formation.split("-")]
    except ValueError:
        nums = []
    if len(nums) < 2:
        return (4, 3, 3)
    return (nums[0], sum(nums[1:-1]), nums[-1])


def _group(pos: str) -> str:
    return _POS_GROUP.get(pos, "MID")


def _curated_or_default(wc: WorldCup, team_id: int) -> str:
    return FORMATIONS.get(wc.teams[team_id].name, DEFAULT_FORMATION)


def _project_xi(wc: WorldCup, rankings: Rankings, team_id: int, formation: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    squad = sorted(
        wc.squad(team_id),
        key=lambda p: rankings.players.get(p.id, _NEUTRAL),
        reverse=True,
    )
    d, m, f = formation_slots(formation)
    need = {"GK": 1, "DEF": d, "MID": m, "FWD": f}
    by_group: dict[str, list] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in squad:
        by_group[_group(p.position)].append(p)

    chosen: list[int] = []
    chosen_set: set[int] = set()
    for group, count in need.items():
        for p in by_group[group][:count]:
            chosen.append(p.id)
            chosen_set.add(p.id)
    # Backfill to 11 from best remaining if a group is short.
    for p in squad:
        if len(chosen) >= 11:
            break
        if p.id not in chosen_set:
            chosen.append(p.id)
            chosen_set.add(p.id)

    start = tuple(chosen[:11])
    start_set = set(start)
    subs = tuple(p.id for p in squad if p.id not in start_set)[:SUB_COUNT]
    return start, subs


def project_lineup(
    wc: WorldCup,
    rankings: Rankings,
    team_id: int,
    fixture_id: int,
    fetcher=None,
) -> ProjectedLineup:
    # 1) Live-fetched recent lineup (most recent played WC match), if a fetcher is provided.
    if fetcher is not None:
        live = fetcher.recent_team_lineup(wc, team_id)
        if live is not None and len(live.start_ids) >= 11:
            # Find the matchday of that recent fixture for provenance.
            md = next((m.matchday for m in wc.matches if m.fixture_id == live.fixture_id), None)
            return ProjectedLineup(team_id, live.formation, live.start_ids[:11], live.sub_ids[:SUB_COUNT], "live", md)

    # 2) Dataset-attached prior lineup from an earlier matchday (dataset lineups are usually empty).
    target = next((m for m in wc.matches if m.fixture_id == fixture_id), None)
    target_md = target.matchday if target else 99
    prior = None
    prior_md = -1
    for lu in wc.lineups:
        if lu.team_id != team_id:
            continue
        md = next((m.matchday for m in wc.matches if m.fixture_id == lu.fixture_id), -1)
        if md < target_md and md > prior_md:
            prior, prior_md = lu, md
    if prior is not None and len(prior.start_ids) >= 11:
        return ProjectedLineup(team_id, prior.formation, prior.start_ids[:11], prior.sub_ids[:SUB_COUNT], "prior", prior_md)

    # 3) Curated formation + squad-rating projection.
    formation = _curated_or_default(wc, team_id)
    start, subs = _project_xi(wc, rankings, team_id, formation)
    return ProjectedLineup(team_id, formation, start, subs, "projected", None)
