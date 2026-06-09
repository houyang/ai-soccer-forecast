# src/soccer/tools/results.py
from __future__ import annotations

from typing import Protocol

from soccer.models import MatchRef, MatchResult
from soccer.tools.base import MissingFixtureKey
from soccer.tools.fixtures import FixtureStore


class ResultProvider(Protocol):
    def get_result(self, match: MatchRef) -> MatchResult | None: ...


class FixtureResultProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_result(self, match: MatchRef) -> MatchResult | None:
        try:
            raw = self._store.get("results", match.id)
        except MissingFixtureKey:
            return None
        return MatchResult(
            match_id=match.id,
            home_goals=raw["home_goals"],
            away_goals=raw["away_goals"],
            status=raw["status"],
            source="fixture",
        )
