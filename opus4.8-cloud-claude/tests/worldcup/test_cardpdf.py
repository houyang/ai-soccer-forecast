from __future__ import annotations

from pathlib import Path

import pytest

from soccer.worldcup.card import build_card
from soccer.worldcup.cardpdf import render_card_pdf
from soccer.worldcup.entities import WorldCup
from soccer.worldcup.ranking import rank_all


def test_render_card_pdf_writes_a_pdf_file(tmp_path: Path, sample_world_cup: WorldCup) -> None:
    pytest.importorskip("reportlab")
    card = build_card(sample_world_cup, rank_all(sample_world_cup), 9001)
    out = tmp_path / "card-9001.pdf"
    render_card_pdf(card, out)
    data = out.read_bytes()
    assert data[:4] == b"%PDF"
    assert len(data) > 500
