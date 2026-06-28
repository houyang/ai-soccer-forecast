# tests/worldcup/test_bracket.py
from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest

from soccer.worldcup.bracket import BracketError, build_bracket, round_name_for
from soccer.worldcup.entities import WcMatch, WorldCup


def _r32(fid: int, h: int, a: int) -> WcMatch:
    return WcMatch(
        fixture_id=fid,
        matchday=0,
        group="",
        home_id=h,
        away_id=a,
        kickoff=datetime(2026, 6, 28, tzinfo=UTC),
        venue="SoFi",
        home_goals=None,
        away_goals=None,
        round_name="Round of 32",
    )


class _WC:
    def __init__(self, matches: tuple[WcMatch, ...]) -> None:
        self.matches = matches


def _full_labels() -> dict[int, str]:
    # 32 teams: ids 1..32 mapped to the 32 advancing slot labels.
    slots = []
    for letter in "ABCDEFGHIJKL":
        slots += [f"1{letter}", f"2{letter}"]
    # eight third-place qualifiers (any 8 distinct groups) to fill the 8 third slots
    slots += [f"3{c}" for c in "CDEFGHIJ"][:8]
    return {i + 1: slots[i] for i in range(32)}


def test_round_name_for_covers_all_rounds() -> None:
    assert round_name_for(73) == "Round of 32"
    assert round_name_for(90) == "Round of 16"
    assert round_name_for(99) == "Quarter-final"
    assert round_name_for(101) == "Semi-final"
    assert round_name_for(103) == "Third-place play-off"
    assert round_name_for(104) == "Final"


def test_build_bracket_maps_double_anchor_slot() -> None:
    labels = _full_labels()
    inv = {v: k for k, v in labels.items()}
    # Slot 73 anchors = {"2A","2B"}; build one R32 fixture for it (+ fillers for the rest).
    fixtures = []
    # one fixture per slot using that slot's anchors / a third filler
    from soccer.worldcup.bracket import R32_ANCHORS

    used_third = iter([f"3{c}" for c in "CDEFGHIJ"])
    fid = 1000
    for _match_no, anchors in R32_ANCHORS.items():
        anchor_list = list(anchors)
        if len(anchor_list) == 2:
            h, a = inv[anchor_list[0]], inv[anchor_list[1]]
        else:
            h = inv[anchor_list[0]]
            a = inv[next(used_third)]
        fixtures.append(_r32(fid, h, a))
        fid += 1
    bracket = build_bracket(cast(WorldCup, _WC(tuple(fixtures))), labels)
    assert set(bracket) == set(range(73, 105))
    # 73 is an R32 tie with concrete ids
    assert bracket[73].home_id is not None
    # 89 is the first R16 tie wired to winners of 74 and 77
    assert (bracket[89].home_src, bracket[89].away_src) == ("W74", "W77")
    assert bracket[104].home_src == "W101" and bracket[104].away_src == "W102"
    assert bracket[103].home_src == "L101" and bracket[103].away_src == "L102"


def test_build_bracket_raises_on_unmatched_fixture() -> None:
    labels = {1: "1A", 2: "1A"}  # impossible duplicate, matches no slot
    fixtures = (_r32(1, 1, 2),)
    with pytest.raises(BracketError):
        build_bracket(cast(WorldCup, _WC(fixtures)), labels)
