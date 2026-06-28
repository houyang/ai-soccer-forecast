"""Official FIFA-2026 knockout bracket: live R32 leaves + fixed downstream tree.

The API provides only the 16 concrete Round-of-32 fixtures; the Round of 16
through the Final are synthesized from the published bracket (matches 73-104,
Wikipedia "2026 FIFA World Cup knockout stage", verified 2026-06-28). Each R32
slot is keyed by its fixed winner/runner-up "anchor" label(s); because every
``(group, rank)`` anchor is unique, a live fixture maps to exactly one slot.
"""

from __future__ import annotations

from dataclasses import dataclass

from soccer.worldcup.entities import WcMatch, WorldCup


class BracketError(Exception):
    """Raised when the live R32 draw cannot be mapped onto the official slots."""


# match_no -> the fixed winner/runner-up labels that identify the slot.
# Slots with two anchors are fully fixed; one-anchor slots take a third-placed team.
R32_ANCHORS: dict[int, frozenset[str]] = {
    73: frozenset({"2A", "2B"}),
    74: frozenset({"1E"}),
    75: frozenset({"1F", "2C"}),
    76: frozenset({"1C", "2F"}),
    77: frozenset({"1I"}),
    78: frozenset({"2E", "2I"}),
    79: frozenset({"1A"}),
    80: frozenset({"1L"}),
    81: frozenset({"1D"}),
    82: frozenset({"1G"}),
    83: frozenset({"2K", "2L"}),
    84: frozenset({"1H", "2J"}),
    85: frozenset({"1B"}),
    86: frozenset({"1J", "2H"}),
    87: frozenset({"1K"}),
    88: frozenset({"2D", "2G"}),
}

# match_no -> (home source, away source) for the synthesized rounds.
KNOCKOUT_EDGES: dict[int, tuple[str, str]] = {
    89: ("W74", "W77"),
    90: ("W73", "W75"),
    91: ("W76", "W78"),
    92: ("W79", "W80"),
    93: ("W83", "W84"),
    94: ("W81", "W82"),
    95: ("W86", "W88"),
    96: ("W85", "W87"),
    97: ("W89", "W90"),
    98: ("W93", "W94"),
    99: ("W91", "W92"),
    100: ("W95", "W96"),
    101: ("W97", "W98"),
    102: ("W99", "W100"),
    103: ("L101", "L102"),
    104: ("W101", "W102"),
}


@dataclass(frozen=True)
class BracketTie:
    match_no: int
    round_name: str
    home_src: str = ""
    away_src: str = ""
    fixture_id: int | None = None
    home_id: int | None = None
    away_id: int | None = None
    venue: str = ""


def round_name_for(match_no: int) -> str:
    if 73 <= match_no <= 88:
        return "Round of 32"
    if 89 <= match_no <= 96:
        return "Round of 16"
    if 97 <= match_no <= 100:
        return "Quarter-final"
    if 101 <= match_no <= 102:
        return "Semi-final"
    if match_no == 103:
        return "Third-place play-off"
    if match_no == 104:
        return "Final"
    raise BracketError(f"no round for match {match_no}")


def _match_r32(fixture: WcMatch, labels: dict[int, str]) -> int:
    present = {labels.get(fixture.home_id, ""), labels.get(fixture.away_id, "")}
    matched = [no for no, anchors in R32_ANCHORS.items() if anchors <= present]
    if len(matched) != 1:
        raise BracketError(
            f"R32 fixture {fixture.fixture_id} (labels {sorted(present)}) "
            f"matched slots {matched}; expected exactly one"
        )
    return matched[0]


def build_bracket(wc: WorldCup, labels: dict[int, str]) -> dict[int, BracketTie]:
    r32 = [m for m in wc.matches if m.round_name == "Round of 32"]
    if len(r32) != 16:
        raise BracketError(f"expected 16 Round of 32 fixtures, found {len(r32)}")
    ties: dict[int, BracketTie] = {}
    for fixture in r32:
        no = _match_r32(fixture, labels)
        if no in ties:
            raise BracketError(f"two fixtures mapped to slot {no}")
        ties[no] = BracketTie(
            match_no=no,
            round_name="Round of 32",
            fixture_id=fixture.fixture_id,
            home_id=fixture.home_id,
            away_id=fixture.away_id,
            venue=fixture.venue,
        )
    if len(ties) != 16:
        raise BracketError("R32 slots not fully populated")
    for no, (home_src, away_src) in KNOCKOUT_EDGES.items():
        ties[no] = BracketTie(
            match_no=no,
            round_name=round_name_for(no),
            home_src=home_src,
            away_src=away_src,
        )
    return ties
