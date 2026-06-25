"""Post-matchday adjustments derived only from matches a team has actually played.

Each played match yields a bounded rating delta (momentum from the scoreline vs the
pre-tournament line, plus a lineup-quality term from who actually started) and a small
formation-based lambda lean. A team with no played match gets a zero adjustment, so the
prediction layer degrades exactly to the pre-tournament baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from soccer.worldcup.entities import Lineup, WcMatch, WorldCup
from soccer.worldcup.predict import SUPREMACY_PER_10, _effective_rating
from soccer.worldcup.ranking import Rankings

# Momentum: rating points per goal of over-/under-performance vs the pre-tournament line.
K_MOM = 0.8
CAP_MOM = 4.0
# Lineup: rating points per point of (starting-XI quality - squad-core quality).
K_LU = 0.15
CAP_LU = 3.0
# One match must not swamp pedigree, so the combined rating delta is capped.
CAP_TOTAL = 5.0
# Formation lean (goals): per forward above 3, per defender above 4.
FORM_ATTACK = 0.06
FORM_DEFENSE = 0.05
_SQUAD_CORE = 16  # mirrors ranking._SQUAD_CORE: strongest N players define squad quality
_NEUTRAL = 50.0


@dataclass(frozen=True)
class TeamAdjustment:
    rating_delta: float = 0.0
    momentum: float = 0.0
    lineup: float = 0.0
    attack_lean: float = 0.0
    defense_lean: float = 0.0


class _LineupLike(Protocol):
    @property
    def formation(self) -> str: ...

    @property
    def start_ids(self) -> tuple[int, ...]: ...


def _clamp(value: float, cap: float) -> float:
    return max(-cap, min(cap, value))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def base_supremacy(wc: WorldCup, rankings: Rankings, match: WcMatch) -> float:
    """Pre-tournament home-oriented goal supremacy with no post-match adjustments."""
    base_h = rankings.teams.get(match.home_id, _NEUTRAL)
    base_a = rankings.teams.get(match.away_id, _NEUTRAL)
    eff_h = _effective_rating(wc, match.home_id, base_h, is_home=True, venue=match.venue)
    eff_a = _effective_rating(wc, match.away_id, base_a, is_home=False, venue=match.venue)
    return SUPREMACY_PER_10 * (eff_h - eff_a) / 10.0


def parse_formation(formation: str) -> tuple[int, int] | None:
    """Return (defenders, forwards) from e.g. '4-3-3'; None if unparseable."""
    parts = formation.split("-")
    if len(parts) < 2:
        return None
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    return nums[0], nums[-1]


def _momentum(wc: WorldCup, rankings: Rankings, team_id: int, played: list[WcMatch]) -> float:
    deltas: list[float] = []
    for match in played:
        is_home = match.home_id == team_id
        supremacy = base_supremacy(wc, rankings, match)
        expected = supremacy if is_home else -supremacy
        assert match.home_goals is not None and match.away_goals is not None
        goals_for = match.home_goals if is_home else match.away_goals
        goals_against = match.away_goals if is_home else match.home_goals
        actual = float(goals_for - goals_against)
        deltas.append(_clamp(K_MOM * (actual - expected), CAP_MOM))
    return _mean(deltas)


def _lineup_delta(
    wc: WorldCup, rankings: Rankings, team_id: int, lineup: _LineupLike | None
) -> float:
    if lineup is None or not lineup.start_ids:
        return 0.0
    team = wc.teams[team_id]
    squad = sorted((rankings.players.get(pid, _NEUTRAL) for pid in team.player_ids), reverse=True)
    squad_core = _mean(squad[:_SQUAD_CORE]) if squad else _NEUTRAL
    xi = _mean([rankings.players.get(pid, _NEUTRAL) for pid in lineup.start_ids])
    return _clamp(K_LU * (xi - squad_core), CAP_LU)


def _formation_lean(lineup: _LineupLike | None) -> tuple[float, float]:
    if lineup is None:
        return 0.0, 0.0
    parsed = parse_formation(lineup.formation)
    if parsed is None:
        return 0.0, 0.0
    defenders, forwards = parsed
    return FORM_ATTACK * (forwards - 3), FORM_DEFENSE * (defenders - 4)


def compute_adjustments(wc: WorldCup, rankings: Rankings) -> dict[int, TeamAdjustment]:
    played_by_team: dict[int, list[WcMatch]] = {}
    for match in wc.matches:
        if match.played:
            played_by_team.setdefault(match.home_id, []).append(match)
            played_by_team.setdefault(match.away_id, []).append(match)
    # Most recent lineup per team (lineups are appended in fixture order by refresh_live).
    lineup_by_team: dict[int, Lineup] = {lu.team_id: lu for lu in wc.lineups}

    out: dict[int, TeamAdjustment] = {}
    for team_id, played in played_by_team.items():
        lineup = lineup_by_team.get(team_id)
        momentum = _momentum(wc, rankings, team_id, played)
        lineup_delta = _lineup_delta(wc, rankings, team_id, lineup)
        attack_lean, defense_lean = _formation_lean(lineup)
        out[team_id] = TeamAdjustment(
            rating_delta=_clamp(momentum + lineup_delta, CAP_TOTAL),
            momentum=momentum,
            lineup=lineup_delta,
            attack_lean=attack_lean,
            defense_lean=defense_lean,
        )
    return out


def adjustment_for_match(
    wc: WorldCup, rankings: Rankings, team_id: int, lineup: _LineupLike | None
) -> TeamAdjustment:
    """Adjustment for an upcoming match: momentum from played games plus this match's XI.

    Unlike :func:`compute_adjustments` (which uses each team's last finished lineup), this is
    driven by the confirmed-or-projected lineup for the specific fixture being previewed.
    """
    played = [m for m in wc.matches if m.played and team_id in (m.home_id, m.away_id)]
    momentum = _momentum(wc, rankings, team_id, played)
    lineup_delta = _lineup_delta(wc, rankings, team_id, lineup)
    attack_lean, defense_lean = _formation_lean(lineup)
    return TeamAdjustment(
        rating_delta=_clamp(momentum + lineup_delta, CAP_TOTAL),
        momentum=momentum,
        lineup=lineup_delta,
        attack_lean=attack_lean,
        defense_lean=defense_lean,
    )
