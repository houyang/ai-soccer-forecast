# tests/test_worldcup_cli.py
import json

from soccer_agent.worldcup.cli import main
from soccer_agent.worldcup.dataset import load_worldcup


def test_predict_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
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
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    code = main(["card", home, away])
    assert code == 0
    pdfs = list((tmp_path / "predictions").glob("*.pdf"))
    assert pdfs, "expected a PDF card"


def test_bracket_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    code = main(["bracket"])
    assert code == 0
    md = tmp_path / "predictions" / "worldcup-2026-knockout-bracket.md"
    js = tmp_path / "predictions" / "worldcup-2026-knockout-bracket.json"
    assert md.exists() and js.exists()
    import json
    d = json.loads(js.read_text())
    assert "rounds" in d and "champion" in d
    assert len(d["rounds"]["R32"]) == 16 and len(d["rounds"]["Final"]) == 1
    cards = list((tmp_path / "predictions" / "bracket-cards").glob("*.pdf"))
    # Every round R32 through the Final + 3rd place has a card (16+8+4+2+1+1 = 32).
    assert len(cards) == 32, f"expected 32 round cards, got {len(cards)}"
    assert any(p.name.startswith("R32-") for p in cards), "expected R32 cards"
    assert any(p.name.startswith("Final-") for p in cards), "expected a Final card"
