from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from soccer.worldcup.apifootball import ApiFootballClient, ApiFootballError
from soccer.worldcup.cache import JsonCache


def _page(response: list[dict[str, Any]], current: int, total: int) -> str:
    return json.dumps(
        {"response": response, "errors": [], "paging": {"current": current, "total": total}}
    )


class RecordingTransport:
    """Fake HTTP transport: serves canned bodies by URL and records every call."""

    def __init__(self, by_url: dict[str, tuple[int, str]]) -> None:
        self._by_url = by_url
        self.calls: list[str] = []

    def __call__(self, url: str, headers: Mapping[str, str]) -> tuple[int, str]:
        self.calls.append(url)
        if url not in self._by_url:
            raise AssertionError(f"unexpected url {url}")
        return self._by_url[url]


def test_requires_key() -> None:
    with pytest.raises(ApiFootballError):
        ApiFootballClient("")


def test_single_page_omits_page_param() -> None:
    base = "https://api.test"
    transport = RecordingTransport({f"{base}/teams?league=1": (200, _page([{"id": 7}], 1, 1))})
    client = ApiFootballClient("k", base_url=base, transport=transport)
    assert client.get("teams", {"league": 1}) == [{"id": 7}]
    assert transport.calls == [f"{base}/teams?league=1"]  # no &page=1


def test_paging_concatenates_and_sends_page_from_two() -> None:
    base = "https://api.test"
    transport = RecordingTransport(
        {
            f"{base}/players?team=5": (200, _page([{"id": 1}], 1, 2)),
            f"{base}/players?team=5&page=2": (200, _page([{"id": 2}], 2, 2)),
        }
    )
    client = ApiFootballClient("k", base_url=base, transport=transport)
    assert client.get("players", {"team": 5}) == [{"id": 1}, {"id": 2}]
    assert transport.calls[1].endswith("&page=2")


def test_http_error_raises() -> None:
    base = "https://api.test"
    transport = RecordingTransport({f"{base}/teams": (500, "boom")})
    client = ApiFootballClient("k", base_url=base, transport=transport)
    with pytest.raises(ApiFootballError, match="HTTP 500"):
        client.get("teams")


def test_api_errors_field_raises() -> None:
    base = "https://api.test"
    body = json.dumps({"response": [], "errors": {"token": "bad"}, "paging": {}})
    transport = RecordingTransport({f"{base}/teams": (200, body)})
    client = ApiFootballClient("k", base_url=base, transport=transport)
    with pytest.raises(ApiFootballError, match="API errors"):
        client.get("teams")


def test_force_refresh_bypasses_cache_read_but_stores(tmp_path: Path) -> None:
    base = "https://api.test"
    transport = RecordingTransport(
        {f"{base}/fixtures?league=1": (200, _page([{"id": 7}], 1, 1))}
    )
    cache = JsonCache(tmp_path)
    client = ApiFootballClient("k", base_url=base, transport=transport, cache=cache)
    client.get("fixtures", {"league": 1})  # warms the cache
    client.get("fixtures", {"league": 1}, force_refresh=True)  # ignores cache, refetches
    assert len(transport.calls) == 2
    assert cache.has("fixtures?league=1&page=1")


def test_cache_prevents_second_network_call(tmp_path: Path) -> None:
    base = "https://api.test"
    transport = RecordingTransport({f"{base}/teams?league=1": (200, _page([{"id": 7}], 1, 1))})
    cache = JsonCache(tmp_path)
    client = ApiFootballClient("k", base_url=base, transport=transport, cache=cache)
    first = client.get("teams", {"league": 1})
    second = client.get("teams", {"league": 1})
    assert first == second == [{"id": 7}]
    assert len(transport.calls) == 1  # second call served from cache
