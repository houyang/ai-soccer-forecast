"""Deterministic modal-bracket forecast: predict every match of every round to the Final.

R32 uses the real fixtures; later rounds pair winners in a balanced binary bracket
(R16[i] = winner(R32[2i]) vs winner(R32[2i+1]), etc.). Knockout ties drawn on the modal
scoreline go to extra time/penalties: the advancing side is argmax(p_home, p_away)
(tiebreak: higher recalibrated strength).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

from soccer_agent.worldcup.bracket import build_bracket
from soccer_agent.worldcup.entities import WorldCup
from soccer_agent.worldcup.lineup import project_lineup
from soccer_agent.worldcup.predict import MatchPrediction, predict_match
from soccer_agent.worldcup.ranking import Rankings

# Projected round windows, as offsets from the latest R32 kickoff.
_ROUND_OFFSET_DAYS = {"R16": 3, "QF": 6, "SF": 10, "Final": 14, "3rd": 13}
_ROUND_VENUE = {
    "R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-Final",
    "SF": "Semi-Final", "Final": "Final", "3rd": "Third-Place Play-off",
}


@dataclass
class BracketMatch:
    round_name: str
    match_no: int
    home_id: Optional[int]
    away_id: Optional[int]
    prediction: Optional[MatchPrediction]
    advancing_id: Optional[int]
    expected_extra_time: bool
    kickoff: Optional[datetime]
    venue: str
    home_name: Optional[str] = None
    away_name: Optional[str] = None
    advancing_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "round": self.round_name, "match_no": self.match_no,
            "home": self.home_name, "away": self.away_name,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "advancing": self.advancing_name,
            "expected_extra_time": self.expected_extra_time,
            "kickoff": self.kickoff.isoformat() if self.kickoff else None,
            "venue": self.venue,
        }


@dataclass
class BracketForecast:
    rounds: dict[str, list[BracketMatch]] = field(default_factory=dict)
    champion_id: Optional[int] = None
    runner_up_id: Optional[int] = None
    third_place_id: Optional[int] = None
    wc: Optional[WorldCup] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rounds": {rnd: [bm.to_dict() for bm in matches] for rnd, matches in self.rounds.items()},
            "champion": self._name(self.champion_id),
            "runner_up": self._name(self.runner_up_id),
            "third_place": self._name(self.third_place_id),
        }

    def _name(self, tid: Optional[int]) -> Optional[str]:
        return self.wc.teams[tid].name if (tid is not None and self.wc and tid in self.wc.teams) else None


def _advance(pred: MatchPrediction, strengths: dict[int, float]) -> tuple[int, bool]:
    """Return (advancing_id, expected_extra_time) for a knockout tie."""
    et = pred.score_home == pred.score_away  # modal score drawn -> ET/pen
    if pred.p_home >= pred.p_away:
        adv = pred.home_id
    else:
        adv = pred.away_id
    # tiebreak by strength if probs are effectively equal
    if abs(pred.p_home - pred.p_away) < 1e-9:
        adv = pred.home_id if strengths.get(pred.home_id, 50.0) >= strengths.get(pred.away_id, 50.0) else pred.away_id
    return adv, et


def forecast_bracket(
    wc: WorldCup, rankings: Rankings, strengths: dict[int, float], forms, fetcher=None,
) -> BracketForecast:
    bracket = build_bracket(wc)
    r32_fixtures = [next(x for x in wc.matches if x.fixture_id == fid) for fid in bracket.r32]
    base_ko = max(m.kickoff for m in r32_fixtures)
    out = BracketForecast(wc=wc, rounds={"R32": [], "R16": [], "QF": [], "SF": [], "Final": [], "3rd": []})

    def name(tid: int) -> str:
        return wc.teams[tid].name

    def predict_pair(home_id: int, away_id: int, round_name: str, match_no: int, kickoff, venue: str) -> BracketMatch:
        hlu = project_lineup(wc, rankings, home_id, 0, fetcher)
        alu = project_lineup(wc, rankings, away_id, 0, fetcher)
        pred = predict_match(wc, rankings, strengths, forms, home_id, away_id, hlu, alu,
                             kickoff=kickoff, venue=venue, group=round_name, round_name=round_name)
        adv, et = _advance(pred, strengths)
        return BracketMatch(round_name, match_no, home_id, away_id, pred, adv, et, kickoff, venue,
                            home_name=name(home_id), away_name=name(away_id), advancing_name=name(adv))

    # R32: real fixtures.
    for i, m in enumerate(r32_fixtures, start=1):
        hlu = project_lineup(wc, rankings, m.home_id, m.fixture_id, fetcher)
        alu = project_lineup(wc, rankings, m.away_id, m.fixture_id, fetcher)
        pred = predict_match(wc, rankings, strengths, forms, m.home_id, m.away_id, hlu, alu,
                             fixture_id=m.fixture_id, kickoff=m.kickoff, venue=m.venue,
                             group=m.group or "R32", matchday=m.matchday, round_name="Round of 32")
        adv, et = _advance(pred, strengths)
        out.rounds["R32"].append(BracketMatch("R32", i, m.home_id, m.away_id, pred, adv, et, m.kickoff, m.venue,
                                              home_name=name(m.home_id), away_name=name(m.away_id), advancing_name=name(adv)))

    # Walk the tree: winners feed the next round.
    winners = [bm.advancing_id for bm in out.rounds["R32"]]  # 16
    losers_sf: list[int] = []

    def walk(slot_teams: list[int], round_name: str) -> list[int]:
        offset = _ROUND_OFFSET_DAYS[round_name]
        ko = base_ko + timedelta(days=offset)
        venue = _ROUND_VENUE[round_name]
        nxt: list[int] = []
        for i in range(0, len(slot_teams), 2):
            match_no = i // 2 + 1
            bm = predict_pair(slot_teams[i], slot_teams[i + 1], round_name, match_no, ko, venue)
            out.rounds[round_name].append(bm)
            nxt.append(bm.advancing_id)
            if round_name == "SF":
                loser = slot_teams[i] if bm.advancing_id == slot_teams[i + 1] else slot_teams[i + 1]
                losers_sf.append(loser)
        return nxt

    r16_winners = walk(winners, "R16")          # 8 -> 4
    qf_winners = walk(r16_winners, "QF")        # 4 -> 2
    sf_winners = walk(qf_winners, "SF")         # 2 -> 1 (SF winners), losers_sf filled
    final_bm = predict_pair(sf_winners[0], sf_winners[1], "Final", 1,
                            base_ko + timedelta(days=_ROUND_OFFSET_DAYS["Final"]), _ROUND_VENUE["Final"])
    out.rounds["Final"].append(final_bm)
    out.champion_id = final_bm.advancing_id
    out.runner_up_id = final_bm.away_id if final_bm.advancing_id == final_bm.home_id else final_bm.home_id
    third_bm = predict_pair(losers_sf[0], losers_sf[1], "3rd", 1,
                            base_ko + timedelta(days=_ROUND_OFFSET_DAYS["3rd"]), _ROUND_VENUE["3rd"])
    out.rounds["3rd"].append(third_bm)
    out.third_place_id = third_bm.advancing_id
    return out
