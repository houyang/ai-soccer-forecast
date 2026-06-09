# src/soccer/tools/odds.py
from __future__ import annotations

from typing import Protocol

from soccer.models import MatchRef, OddsSnapshot
from soccer.tools.fixtures import FixtureStore


class OddsProvider(Protocol):
    def get_odds(self, match: MatchRef) -> OddsSnapshot: ...


class FixtureOddsProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_odds(self, match: MatchRef) -> OddsSnapshot:
        raw = self._store.get("odds", match.id)
        return OddsSnapshot(
            bookmaker=raw["bookmaker"],
            home=raw["home"],
            draw=raw["draw"],
            away=raw["away"],
            as_of=match.kickoff,
            source="fixture",
        )
