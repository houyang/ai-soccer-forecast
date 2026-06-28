"""Walk the knockout bracket: a modal headline bracket and Monte-Carlo odds.

The modal pass advances each tie's more-likely side to a single champion. The
Monte-Carlo pass (Task 7) samples each tie to produce advancement/title odds.
Both reuse the knockout match model in :mod:`soccer.worldcup.predict`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from soccer.worldcup.bracket import BracketTie
from soccer.worldcup.entities import WorldCup
from soccer.worldcup.predict import KnockoutPrediction, predict_knockout
from soccer.worldcup.ranking import Rankings


@dataclass(frozen=True)
class Podium:
    champion_id: int
    champion_name: str
    runner_up_id: int
    runner_up_name: str
    third_id: int
    third_name: str
    fourth_id: int
    fourth_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "champion": {"id": self.champion_id, "name": self.champion_name},
            "runner_up": {"id": self.runner_up_id, "name": self.runner_up_name},
            "third": {"id": self.third_id, "name": self.third_name},
            "fourth": {"id": self.fourth_id, "name": self.fourth_name},
        }


def _resolve_src(src: str, winners: dict[int, int], losers: dict[int, int]) -> int:
    kind, ref = src[0], int(src[1:])
    return winners[ref] if kind == "W" else losers[ref]


def _teams_for(tie: BracketTie, winners: dict[int, int], losers: dict[int, int]) -> tuple[int, int]:
    if tie.home_id is not None and tie.away_id is not None:
        return tie.home_id, tie.away_id
    return (
        _resolve_src(tie.home_src, winners, losers),
        _resolve_src(tie.away_src, winners, losers),
    )


def run_modal_bracket(
    wc: WorldCup, rankings: Rankings, ties: dict[int, BracketTie]
) -> tuple[list[KnockoutPrediction], Podium]:
    winners: dict[int, int] = {}
    losers: dict[int, int] = {}
    preds: list[KnockoutPrediction] = []
    for no in sorted(ties):
        tie = ties[no]
        home_id, away_id = _teams_for(tie, winners, losers)
        pred = predict_knockout(
            wc,
            rankings,
            home_id,
            away_id,
            match_no=no,
            round_name=tie.round_name,
            venue=tie.venue,
        )
        preds.append(pred)
        if pred.p_home_advance >= 0.5:
            winners[no], losers[no] = home_id, away_id
        else:
            winners[no], losers[no] = away_id, home_id
    champ = winners[104]
    runner = losers[104]
    third = winners[103]
    fourth = losers[103]
    podium = Podium(
        champion_id=champ,
        champion_name=wc.teams[champ].name,
        runner_up_id=runner,
        runner_up_name=wc.teams[runner].name,
        third_id=third,
        third_name=wc.teams[third].name,
        fourth_id=fourth,
        fourth_name=wc.teams[fourth].name,
    )
    return preds, podium
