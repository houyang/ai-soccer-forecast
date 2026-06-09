# tests/test_registry.py
import json
from pathlib import Path

from soccer.registry import build_fixture_registry


def test_build_fixture_registry_exposes_providers(tmp_path: Path) -> None:
    payload: dict[str, dict[str, object]] = {
        "form": {},
        "injuries": {},
        "h2h": {},
        "weather": {},
        "venue": {},
        "odds": {},
        "results": {},
    }
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    reg = build_fixture_registry(path)
    assert reg.form is not None and reg.results is not None


def test_registry_tool_view_lists_capabilities(tmp_path: Path) -> None:
    payload: dict[str, dict[str, object]] = {
        "form": {},
        "injuries": {},
        "h2h": {},
        "weather": {},
        "venue": {},
        "odds": {},
        "results": {},
    }
    path = tmp_path / "f.json"
    path.write_text(json.dumps(payload))
    reg = build_fixture_registry(path)
    tools = reg.as_tools()
    names = {t.name for t in tools}
    assert {
        "form",
        "injuries",
        "h2h",
        "weather",
        "venue",
        "odds",
        "results",
    } <= names
    form_tool = next(t for t in tools if t.name == "form")
    assert form_tool.call == reg.form.get_form
