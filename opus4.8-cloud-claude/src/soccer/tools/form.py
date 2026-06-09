# src/soccer/tools/form.py
from __future__ import annotations

from datetime import datetime
from typing import Protocol

from soccer.models import MatchOutcome, TeamForm
from soccer.tools.fixtures import FixtureStore


class FormProvider(Protocol):
    def get_form(self, team: str, as_of: datetime) -> TeamForm: ...


class FixtureFormProvider:
    def __init__(self, store: FixtureStore) -> None:
        self._store = store

    def get_form(self, team: str, as_of: datetime) -> TeamForm:
        raw = self._store.get("form", team)
        return TeamForm(
            team=raw["team"],
            last_n=tuple(MatchOutcome(x) for x in raw["last_n"]),
            gf=raw["gf"],
            ga=raw["ga"],
            points=raw["points"],
            streak=raw["streak"],
            as_of=as_of,
            source="fixture",
        )
