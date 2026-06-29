# tests/test_worldcup_card.py
import json

import pytest

from soccer_agent.worldcup.card import build_card
from soccer_agent.worldcup.dataset import load_worldcup
from soccer_agent.worldcup.form import compute_forms, recalibrated_strength
from soccer_agent.worldcup.ranking import rank_all


def _setup():
    wc = load_worldcup()
    r = rank_all(wc)
    f = compute_forms(wc)
    return wc, r, recalibrated_strength(wc, r, f)


def test_build_card_structure():
    wc, r, s = _setup()
    m = next(m for m in wc.matches if m.matchday == 0)
    card = build_card(wc, r, s, m.home_id, m.away_id, fixture_id=m.fixture_id)
    assert card.home.name and card.away.name
    assert len(card.home.starters) == 11
    assert len(card.home.subs) == 7
    assert card.home.coach_name is not None
    assert card.prediction is not None
    d = card.to_dict()
    json.dumps(d)  # serializable


def test_render_card_pdf_skips_without_reportlab(tmp_path):
    pytest.importorskip("reportlab")  # skip if not installed
    from soccer_agent.worldcup.cardpdf import render_card_pdf
    wc, r, s = _setup()
    m = next(m for m in wc.matches if m.matchday == 0)
    card = build_card(wc, r, s, m.home_id, m.away_id, fixture_id=m.fixture_id)
    out = tmp_path / "x.pdf"
    render_card_pdf(card, out)
    assert out.exists() and out.stat().st_size > 100
