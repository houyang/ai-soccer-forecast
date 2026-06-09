# src/soccer/tools/venue.py
from __future__ import annotations

from typing import Protocol

from soccer.models import VenueInfo
from soccer.tools.fixtures import FixtureStore


class VenueProvider(Protocol):
    def get_venue(self, venue_id: str) -> VenueInfo: ...


class FixtureVenueProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_venue(self, venue_id: str) -> VenueInfo:
        raw = self._store.get("venue", venue_id)
        return VenueInfo(
            venue_id=raw["venue_id"],
            name=raw["name"],
            city=raw["city"],
            surface=raw["surface"],
            capacity=raw["capacity"],
            altitude_m=raw["altitude_m"],
            home_advantage_hint=raw["home_advantage_hint"],
            source="fixture",
        )
