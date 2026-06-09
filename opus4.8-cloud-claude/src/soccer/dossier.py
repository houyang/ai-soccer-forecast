from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TypeVar

from soccer.models import MatchDossier, MatchRef
from soccer.registry import ToolRegistry
from soccer.tools.base import ToolError

T = TypeVar("T")


def _try(missing: list[str], name: str, fn: Callable[[], T]) -> T | None:
    try:
        return fn()
    except ToolError:
        missing.append(name)
        return None


def build_dossier(match: MatchRef, registry: ToolRegistry) -> MatchDossier:
    missing: list[str] = []
    form = {
        "home": _try(
            missing, "form:home", lambda: registry.form.get_form(match.home, match.kickoff)
        ),
        "away": _try(
            missing, "form:away", lambda: registry.form.get_form(match.away, match.kickoff)
        ),
    }
    injuries = {
        "home": _try(
            missing,
            "injuries:home",
            lambda: registry.injuries.get_injuries(match.home, match.kickoff),
        ),
        "away": _try(
            missing,
            "injuries:away",
            lambda: registry.injuries.get_injuries(match.away, match.kickoff),
        ),
    }
    h2h = _try(missing, "h2h", lambda: registry.h2h.get_h2h(match.home, match.away))
    weather = _try(
        missing, "weather", lambda: registry.weather.get_weather(match.venue_id, match.kickoff)
    )
    venue = _try(missing, "venue", lambda: registry.venue.get_venue(match.venue_id))
    odds = _try(missing, "odds", lambda: registry.odds.get_odds(match))
    return MatchDossier(
        match=match,
        form=form,
        injuries=injuries,
        h2h=h2h,
        weather=weather,
        venue=venue,
        odds=odds,
        missing=tuple(missing),
    )


def dossier_digest(dossier: MatchDossier) -> str:
    odds = dossier.odds
    parts = [
        dossier.match.id,
        f"form={[f.points if f else None for f in dossier.form.values()]}",
        f"odds={None if odds is None else (odds.home, odds.draw, odds.away)}",
        f"missing={sorted(dossier.missing)}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
