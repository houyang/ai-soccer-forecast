# tests/test_worldcup_dataset.py
import json
from pathlib import Path


def test_dataset_present_and_well_formed():
    path = Path(__file__).resolve().parents[1] / "data" / "worldcup-2026.json"
    assert path.exists(), "data/worldcup-2026.json must be copied in"
    data = json.loads(path.read_text())
    for key in ("teams", "players", "coaches", "matches"):
        assert key in data and len(data[key]) > 0
    assert len(data["teams"]) == 48
    assert len(data["matches"]) == 88
