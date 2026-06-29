# tests/test_worldcup_cli.py
import json

from soccer_agent.worldcup.cli import main
from soccer_agent.worldcup.dataset import load_worldcup


def test_predict_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = main(["predict"])
    assert code == 0
    md = tmp_path / "predictions" / "worldcup-2026-predictions-after1st-group.md"
    js = tmp_path / "predictions" / "worldcup-2026-predictions-after1st-group.json"
    assert md.exists() and js.exists()
    data = json.loads(js.read_text())
    assert "standings" in data and "r32" in data and "bracket" in data
    assert len(data["r32"]) == 16


def test_card_writes_pdf_and_json(tmp_path, monkeypatch):
    import pytest
    pytest.importorskip("reportlab")
    wc = load_worldcup()
    m = next(m for m in wc.matches if m.matchday == 0)  # first real R32 fixture
    home = wc.teams[m.home_id].name
    away = wc.teams[m.away_id].name
    monkeypatch.chdir(tmp_path)
    code = main(["card", home, away])
    assert code == 0
    pdfs = list((tmp_path / "predictions").glob("*.pdf"))
    assert pdfs, "expected a PDF card"
