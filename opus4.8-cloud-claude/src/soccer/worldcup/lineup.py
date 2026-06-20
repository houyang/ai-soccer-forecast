"""Project the most likely starting XI, subs, and formation for an upcoming match.

Precedence: an official lineup already attached to this fixture ("confirmed"); else the
team's most recent lineup from an earlier matchday in this tournament ("prior"); else a
squad-based projection using the coach-preferred formation ("projected"). The result feeds
both the printed card and the lineup-aware prediction, so it carries a ``source`` provenance
label rather than silently guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from soccer.worldcup.entities import Lineup, Player, WorldCup
from soccer.worldcup.ranking import Rankings

DEFAULT_FORMATION = "4-3-3"
SUB_COUNT = 7
_NEUTRAL = 50.0
_POSITION_GROUP = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Attacker": "FWD",
    "G": "GK",
    "D": "DEF",
    "M": "MID",
    "F": "FWD",
}


@dataclass(frozen=True)
class ProjectedLineup:
    team_id: int
    formation: str
    start_ids: tuple[int, ...]
    sub_ids: tuple[int, ...]
    source: str  # "confirmed" | "prior" | "projected"
    source_matchday: int | None = None


def formation_slots(formation: str) -> tuple[int, int, int]:
    """Return (defenders, midfielders, forwards) from e.g. '4-3-3' or '4-2-3-1'.

    The goalkeeper is implicit (always 1). Unparseable input falls back to a 4-3-3 shape.
    """
    try:
        nums = [int(part) for part in formation.split("-")]
    except ValueError:
        nums = []
    if len(nums) < 2:
        return (4, 3, 3)
    defenders = nums[0]
    forwards = nums[-1]
    midfielders = sum(nums[1:-1])
    return (defenders, midfielders, forwards)


def _position_group(position: str) -> str:
    return _POSITION_GROUP.get(position, "MID")


def preferred_formation(wc: WorldCup, team_id: int) -> str:
    counts: dict[str, int] = {}
    for lineup in wc.lineups:
        if lineup.team_id == team_id and lineup.formation:
            counts[lineup.formation] = counts.get(lineup.formation, 0) + 1
    if not counts:
        return DEFAULT_FORMATION
    return max(counts, key=lambda formation: counts[formation])


def _project_xi(
    wc: WorldCup, rankings: Rankings, team_id: int, formation: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    squad: list[Player] = sorted(
        wc.squad(team_id),
        key=lambda p: rankings.players.get(p.id, _NEUTRAL),
        reverse=True,
    )
    defenders, midfielders, forwards = formation_slots(formation)
    need = {"GK": 1, "DEF": defenders, "MID": midfielders, "FWD": forwards}
    by_group: dict[str, list[Player]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for player in squad:
        by_group[_position_group(player.position)].append(player)

    chosen: list[int] = []
    chosen_ids: set[int] = set()
    for group, count in need.items():
        for player in by_group[group][:count]:
            chosen.append(player.id)
            chosen_ids.add(player.id)
    # Backfill to 11 from the best remaining players when a position group is short.
    if len(chosen) < 11:
        for player in squad:
            if player.id not in chosen_ids:
                chosen.append(player.id)
                chosen_ids.add(player.id)
                if len(chosen) >= 11:
                    break

    start_ids = tuple(chosen[:11])
    start_set = set(start_ids)
    sub_ids = tuple(p.id for p in squad if p.id not in start_set)[:SUB_COUNT]
    return start_ids, sub_ids


def project_lineup(
    wc: WorldCup, rankings: Rankings, team_id: int, fixture_id: int
) -> ProjectedLineup:
    target = next((m for m in wc.matches if m.fixture_id == fixture_id), None)
    if target is None:
        raise ValueError(f"fixture {fixture_id} not found in dataset")

    for lineup in wc.lineups:
        if lineup.fixture_id == fixture_id and lineup.team_id == team_id:
            return ProjectedLineup(
                team_id=team_id,
                formation=lineup.formation or DEFAULT_FORMATION,
                start_ids=lineup.start_ids,
                sub_ids=lineup.sub_ids,
                source="confirmed",
                source_matchday=None,
            )

    matchday_by_fixture = {m.fixture_id: m.matchday for m in wc.matches}
    prior: Lineup | None = None
    prior_matchday = -1
    for lineup in wc.lineups:
        if lineup.team_id != team_id:
            continue
        matchday = matchday_by_fixture.get(lineup.fixture_id, -1)
        if matchday < target.matchday and matchday > prior_matchday:
            prior, prior_matchday = lineup, matchday
    if prior is not None:
        return ProjectedLineup(
            team_id=team_id,
            formation=prior.formation or DEFAULT_FORMATION,
            start_ids=prior.start_ids,
            sub_ids=prior.sub_ids,
            source="prior",
            source_matchday=prior_matchday,
        )

    formation = preferred_formation(wc, team_id)
    start_ids, sub_ids = _project_xi(wc, rankings, team_id, formation)
    return ProjectedLineup(
        team_id=team_id,
        formation=formation,
        start_ids=start_ids,
        sub_ids=sub_ids,
        source="projected",
        source_matchday=None,
    )
