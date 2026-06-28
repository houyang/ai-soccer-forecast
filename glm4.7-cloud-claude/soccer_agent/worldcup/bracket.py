# soccer_agent/worldcup/bracket.py
"""Round-of-32 fixtures (from the dataset) + an approximated R16->Final binary tree.

The dataset carries the 16 real R32 pairings but no R16 slot map, so R32 matches are
paired into R16 by sorted fixture_id (match 1 vs 2, 3 vs 4, ...). This pairing is an
approximation; R32 itself is exact.
"""
from __future__ import annotations

from dataclasses import dataclass

from soccer_agent.worldcup.entities import WorldCup


@dataclass(frozen=True)
class Bracket:
    r32: tuple[int, ...]            # fixture_ids, sorted
    pairs: tuple[tuple[int, int], ...]  # 8 R16 pairs of R32 indices (into r32)


def build_bracket(wc: WorldCup) -> Bracket:
    r32 = tuple(sorted(m.fixture_id for m in wc.matches if m.matchday == 0))
    pairs = tuple((r32[i], r32[i + 1]) for i in range(0, len(r32), 2))
    return Bracket(r32=r32, pairs=pairs)
