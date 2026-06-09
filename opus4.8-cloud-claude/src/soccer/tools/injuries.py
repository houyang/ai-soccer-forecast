# src/soccer/tools/injuries.py
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from soccer.models import InjuryReport, PlayerStatus
from soccer.tools.fixtures import FixtureStore


class InjuryProvider(Protocol):
    def get_injuries(self, team: str, as_of: datetime) -> InjuryReport: ...


def _players(items: list[dict[str, str]]) -> tuple[PlayerStatus, ...]:
    return tuple(
        PlayerStatus(name=i["name"], status=i["status"], reason=i["reason"]) for i in items
    )


class FixtureInjuryProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_injuries(self, team: str, as_of: datetime) -> InjuryReport:
        raw = self._store.get("injuries", team)
        return InjuryReport(
            team=raw["team"],
            out=_players(raw["out"]),
            doubtful=_players(raw["doubtful"]),
            as_of=as_of,
            source="fixture",
        )
