from __future__ import annotations

from pathlib import Path

from soccer.worldcup.cache import JsonCache


def test_store_and_load_round_trip(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path)
    payload = {"response": [{"a": 1}], "paging": {"current": 1, "total": 1}}
    cache.store("teams?league=1&page=1", payload)
    assert cache.has("teams?league=1&page=1")
    assert cache.load("teams?league=1&page=1") == payload


def test_missing_key_returns_none(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path)
    assert cache.load("nope") is None
    assert cache.has("nope") is False


def test_long_key_is_hashed_into_safe_filename(tmp_path: Path) -> None:
    cache = JsonCache(tmp_path)
    key = "players?" + "x=1&" * 100
    cache.store(key, {"ok": True})
    assert cache.load(key) == {"ok": True}
    # exactly one file written, name within filesystem limits
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert len(files[0].name) <= 130
