"""Monte-Carlo simulation of the knockout bracket to a champion.

Each R32 match is predicted once (deterministic) for the printed card; the bracket is then
walked by sampling match outcomes from the Poisson scoreline matrix. Knockout ties that are
drawn after 90' go to extra time (lambdas * 4/3) and, if still level, a shootout whose win
prob is shifted by the rating edge (capped at +/-0.15).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from soccer_agent.worldcup.bracket import Bracket, build_bracket
from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.lineup import project_lineup
from soccer_agent.worldcup.predict import (
    BASE_MATCH_GOALS,
    LAMBDA_FLOOR,
    SUPREMACY_PER_10,
    MatchPrediction,
    predict_one,
    scoreline_matrix,
)
from soccer_agent.worldcup.ranking import Rankings

ET_FACTOR = 4.0 / 3.0
PEN_EDGE_PER_10 = 0.03
PEN_EDGE_CAP = 0.15


@dataclass
class BracketSim:
    r32_predictions: list[MatchPrediction] = field(default_factory=list)
    champion: dict[int, float] = field(default_factory=dict)
    advancement: dict[int, dict[str, float]] = field(default_factory=dict)
    modal_path: list[dict[str, Any]] = field(default_factory=list)


def _sample_winner(lh: float, la: float, eff_h: float, eff_a: float, rng: random.Random) -> int:
    """Return 1 if home wins the tie, 2 if away wins (after ET + pens if needed)."""
    mat = scoreline_matrix(lh, la)
    flat = [(i, j, mat[i][j]) for i in range(len(mat)) for j in range(len(mat))]
    r = rng.random()
    cum = 0.0
    sh, sa = flat[-1][0], flat[-1][1]  # absorb float-drift tail into the last cell
    for i, j, p in flat:
        cum += p
        if r <= cum:
            sh, sa = i, j
            break
    if sh != sa:
        return 1 if sh > sa else 2
    # Extra time.
    et = scoreline_matrix(lh * ET_FACTOR, la * ET_FACTOR)
    flat = [(i, j, et[i][j]) for i in range(len(et)) for j in range(len(et))]
    r = rng.random()
    cum = 0.0
    sh, sa = flat[-1][0], flat[-1][1]  # absorb float-drift tail into the last cell
    for i, j, p in flat:
        cum += p
        if r <= cum:
            sh, sa = i, j
            break
    if sh != sa:
        return 1 if sh > sa else 2
    # Shootout: shift 0.5 by rating edge.
    edge = (eff_h - eff_a) / 10.0 * PEN_EDGE_PER_10
    edge = max(-PEN_EDGE_CAP, min(PEN_EDGE_CAP, edge))
    return 1 if rng.random() < (0.5 + edge) else 2


def simulate_bracket(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float],
    fetcher=None, n: int = 10000, seed: int = 2026,
) -> BracketSim:
    bracket: Bracket = build_bracket(wc)
    rng = random.Random(seed)

    # Predict each R32 match deterministically for the cards.
    r32_preds: list[MatchPrediction] = []
    r32_map: dict[int, MatchPrediction] = {}
    for fid in bracket.r32:
        m = next(x for x in wc.matches if x.fixture_id == fid)
        hlu = project_lineup(wc, rankings, m.home_id, fid, fetcher)
        alu = project_lineup(wc, rankings, m.away_id, fid, fetcher)
        pred = predict_one(wc, rankings, strengths, fid, hlu, alu)
        r32_preds.append(pred)
        r32_map[fid] = pred

    champion: dict[int, float] = {tid: 0.0 for tid in wc.teams}

    for _ in range(n):
        # current_winners: list of team_ids at each slot, seeded from R32 fixtures.
        slots: list[int] = []
        for fid in bracket.r32:
            m = next(x for x in wc.matches if x.fixture_id == fid)
            pred = r32_map[fid]
            eff_h = pred.home_adjustment + strengths.get(m.home_id, 50.0)
            eff_a = pred.away_adjustment + strengths.get(m.away_id, 50.0)
            w = _sample_winner(pred.lambda_home, pred.lambda_away, eff_h, eff_a, rng)
            slots.append(m.home_id if w == 1 else m.away_id)
        # R16 -> Final: pair adjacent slots.
        round_idx = 1
        while len(slots) > 1:
            nxt: list[int] = []
            for i in range(0, len(slots), 2):
                home_id, away_id = slots[i], slots[i + 1]
                gap = strengths.get(home_id, 50.0) - strengths.get(away_id, 50.0)
                lh = max(LAMBDA_FLOOR, BASE_MATCH_GOALS / 2.0 + gap / 10.0 * SUPREMACY_PER_10 / 2.0)
                la = max(LAMBDA_FLOOR, BASE_MATCH_GOALS - lh)
                eff_h = strengths.get(home_id, 50.0)
                eff_a = strengths.get(away_id, 50.0)
                w = _sample_winner(lh, la, eff_h, eff_a, rng)
                nxt.append(home_id if w == 1 else away_id)
            slots = nxt
            round_idx += 1
        champion[slots[0]] += 1.0

    for tid in champion:
        champion[tid] /= n

    return BracketSim(r32_predictions=r32_preds, champion=champion, advancement={}, modal_path=[])
