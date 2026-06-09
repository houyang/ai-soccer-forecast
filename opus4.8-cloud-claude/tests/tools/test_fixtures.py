# tests/tools/test_fixtures.py
import json
from pathlib import Path

import pytest

from soccer.tools.base import MissingFixtureKey, ToolError
from soccer.tools.fixtures import FixtureStore


def test_fixture_store_reads_section(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"form": {"A": {"team": "A"}}}))
    store = FixtureStore(path)
    assert store.get("form", "A") == {"team": "A"}


def test_fixture_store_missing_key_raises_toolerror(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"form": {}}))
    store = FixtureStore(path)
    with pytest.raises(ToolError):
        store.get("form", "Z")


def test_fixture_store_missing_key_raises_missingfixturekey(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"form": {}}))
    store = FixtureStore(path)
    with pytest.raises(MissingFixtureKey) as excinfo:
        store.get("form", "Z")
    assert isinstance(excinfo.value, ToolError)


def test_fixture_store_missing_section_raises_toolerror_not_missingkey(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"form": {"A": {"team": "A"}}}))
    store = FixtureStore(path)
    with pytest.raises(ToolError) as excinfo:
        store.get("results", "A")
    assert not isinstance(excinfo.value, MissingFixtureKey)


def test_fixture_store_non_dict_section_raises_toolerror(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"form": [1, 2, 3]}))
    store = FixtureStore(path)
    with pytest.raises(ToolError) as excinfo:
        store.get("form", "A")
    assert not isinstance(excinfo.value, MissingFixtureKey)
