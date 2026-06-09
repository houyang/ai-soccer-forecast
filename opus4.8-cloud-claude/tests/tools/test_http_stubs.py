# tests/tools/test_http_stubs.py
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from soccer.models import MatchRef
from soccer.tools.http_stubs import (
    HttpFormProvider,
    HttpH2HProvider,
    HttpInjuryProvider,
    HttpOddsProvider,
    HttpResultProvider,
    HttpVenueProvider,
    HttpWeatherProvider,
)

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=UTC)
REF = MatchRef(
    id="m1",
    competition="UCL",
    home="A",
    away="B",
    kickoff=KICK,
    venue_id="v1",
    season="2025-26",
)


@pytest.mark.parametrize(
    "invoke",
    [
        lambda: HttpFormProvider(base_url="https://example.test").get_form("A", KICK),
        lambda: HttpInjuryProvider(base_url="https://example.test").get_injuries("A", KICK),
        lambda: HttpH2HProvider(base_url="https://example.test").get_h2h("A", "B"),
        lambda: HttpWeatherProvider(base_url="https://example.test").get_weather("v1", KICK),
        lambda: HttpVenueProvider(base_url="https://example.test").get_venue("v1"),
        lambda: HttpOddsProvider(base_url="https://example.test").get_odds(REF),
        lambda: HttpResultProvider(base_url="https://example.test").get_result(REF),
    ],
)
def test_http_provider_not_implemented(invoke: Callable[[], object]) -> None:
    with pytest.raises(NotImplementedError):
        invoke()
