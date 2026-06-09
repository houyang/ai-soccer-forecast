# tests/test_models_dossier.py
from datetime import UTC, datetime

import pytest

from soccer.models import (
    MatchDossier,
    MatchOutcome,
    MatchRef,
    OddsSnapshot,
    Outcome,
    TeamForm,
)

KICK = datetime(2026, 4, 1, 19, 0, tzinfo=UTC)


def test_odds_implied_probs_normalised() -> None:
    odds = OddsSnapshot(bookmaker="b", home=2.0, draw=4.0, away=4.0, as_of=KICK, source="fixture")
    p = odds.implied_probs
    assert sum(p.values()) == 0.0 + 1.0  # normalised to 1
    assert p[Outcome.HOME] > p[Outcome.DRAW]


def test_odds_rejects_nonpositive_decimal_odds() -> None:
    with pytest.raises(ValueError):
        OddsSnapshot(bookmaker="b", home=0.0, draw=4.0, away=4.0, as_of=KICK, source="fixture")


def test_dossier_holds_optional_pieces() -> None:
    ref = MatchRef(
        id="m1",
        competition="UCL",
        home="A",
        away="B",
        kickoff=KICK,
        venue_id="v1",
        season="2025-26",
    )
    form = TeamForm(
        team="A",
        last_n=(MatchOutcome.WIN,),
        gf=3,
        ga=1,
        points=3,
        streak="W1",
        as_of=KICK,
        source="fixture",
    )
    d = MatchDossier(
        match=ref,
        form={"home": form, "away": None},
        injuries={"home": None, "away": None},
        h2h=None,
        weather=None,
        venue=None,
        odds=None,
        missing=("odds",),
    )
    home_form = d.form["home"]
    assert home_form is not None
    assert home_form.team == "A"
    assert "odds" in d.missing
