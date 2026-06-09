# src/soccer/tools/head_to_head.py
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from soccer.models import H2HRecord, PastMeeting
from soccer.tools.fixtures import FixtureStore


class H2HProvider(Protocol):
    def get_h2h(self, home: str, away: str) -> H2HRecord: ...


class FixtureH2HProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_h2h(self, home: str, away: str) -> H2HRecord:
        raw = self._store.get("h2h", f"{home}|{away}")
        meetings = tuple(
            PastMeeting(
                date=datetime.fromisoformat(m["date"]),
                home=m["home"],
                away=m["away"],
                home_goals=m["home_goals"],
                away_goals=m["away_goals"],
            )
            for m in raw["meetings"]
        )
        return H2HRecord(
            home=raw["home"],
            away=raw["away"],
            meetings=meetings,
            home_wins=raw["home_wins"],
            draws=raw["draws"],
            away_wins=raw["away_wins"],
            source="fixture",
        )
