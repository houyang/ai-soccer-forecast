"""Client for API-Sports v3 (api-football).

The HTTP transport is injected (``HttpGet``) so the client is fully testable without a
network: tests pass a fake transport. The production default wraps stdlib
``urllib.request`` -- no third-party dependency. Responses are cached page-by-page via an
optional :class:`~soccer.worldcup.cache.JsonCache`, so a completed ingest costs nothing to
replay and the offline ranking/prediction paths read straight from the cache.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from soccer.worldcup.cache import JsonCache

# (url, headers) -> (status_code, body_text)
HttpGet = Callable[[str, Mapping[str, str]], tuple[int, str]]

_MAX_PAGES = 50


class ApiFootballError(Exception):
    """Raised on HTTP, quota, or payload errors that prevent returning data."""


def _default_ssl_context() -> ssl.SSLContext:
    # Verify against the certifi CA bundle: the macOS/Python.org build does not read the
    # system keychain, so the stdlib default context fails to find an issuer. TLS
    # verification stays ON -- we never disable it.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def urllib_transport(timeout: float = 30.0, context: ssl.SSLContext | None = None) -> HttpGet:
    ssl_context = context or _default_ssl_context()

    def _get(url: str, headers: Mapping[str, str]) -> tuple[int, str]:
        request = urllib.request.Request(url, headers=dict(headers), method="GET")
        try:
            with urllib.request.urlopen(  # noqa: S310
                request, timeout=timeout, context=ssl_context
            ) as response:
                body: str = response.read().decode("utf-8")
                status: int = response.status
                return status, body
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            raise ApiFootballError(f"network error for {url}: {exc.reason}") from exc

    return _get


class ApiFootballClient:
    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://v3.football.api-sports.io",
        transport: HttpGet | None = None,
        cache: JsonCache | None = None,
        sleep: Callable[[float], None] = time.sleep,
        throttle_seconds: float = 0.0,
    ) -> None:
        if not api_key:
            raise ApiFootballError("API-Football key is required")
        self._key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport or urllib_transport()
        self._cache = cache
        self._sleep = sleep
        self._throttle = throttle_seconds

    def _headers(self) -> dict[str, str]:
        return {"x-apisports-key": self._key, "Accept": "application/json"}

    def _cache_key(self, path: str, params: Mapping[str, Any], page: int) -> str:
        query = "&".join(f"{k}={params[k]}" for k in sorted(params))
        return f"{path}?{query}&page={page}"

    def _fetch_page(self, path: str, params: Mapping[str, Any], page: int) -> dict[str, Any]:
        key = self._cache_key(path, params, page)
        if self._cache is not None:
            cached = self._cache.load(key)
            if cached is not None:
                return dict(cached)
        # This API rejects an explicit page=1; only paginate from page 2 onward.
        query = dict(params) if page == 1 else {**params, "page": page}
        url = f"{self._base_url}/{path}"
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        if self._throttle:
            self._sleep(self._throttle)
        status, body = self._transport(url, self._headers())
        if status != 200:
            raise ApiFootballError(f"{path} returned HTTP {status}: {body[:200]}")
        try:
            payload: dict[str, Any] = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiFootballError(f"{path} returned non-JSON body") from exc
        errors = payload.get("errors")
        if errors:
            raise ApiFootballError(f"{path} API errors: {errors}")
        if self._cache is not None:
            self._cache.store(key, payload)
        return payload

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return the concatenated ``response`` array across all pages."""
        params = dict(params or {})
        out: list[dict[str, Any]] = []
        page = 1
        while page <= _MAX_PAGES:
            payload = self._fetch_page(path, params, page)
            out.extend(payload.get("response", []))
            paging = payload.get("paging") or {}
            current = int(paging.get("current", page))
            total = int(paging.get("total", current))
            if current >= total:
                break
            page = current + 1
        return out
