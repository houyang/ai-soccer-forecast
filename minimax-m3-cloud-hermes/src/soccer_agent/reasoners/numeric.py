"""Numeric reasoner.

Deterministic, fast, and interpretable. Weights chosen from a simple
logistic blend — these are not optimised; Phase 2 will tune them on
the 24/25 eval set.

Components (each scaled to ~0..1 then blended):

  elo_component        – Elo-based expected outcome
  form_component       – win rate differential over last N (home - away)
  h2h_component        – H2H win rate for the home team (last N meetings)
  injury_component     – count of "out" injuries (negative signal)
  market_component     – devigged market-implied probability for "home"
  weather_component    – neutral unless playability_risk == "high"

Blend formula:
    logit_home = w_elo * elo
              + w_form * form
              + w_h2h * (h2h - 0.5)
              + w_inj * injury
              + w_market * (market - 1/3)

logit_away = mirror.  draw is the residual that closes the simplex.

Confidence is derived from the gap between the top two probs.
"""

from __future__ import annotations

from ..models import (
    Factor,
    H2HOutput,
    InjuryOutput,
    MatchContext,
    OddsOutput,
    ReasonerOutput,
    WeatherOutput,
)
from .base import Reasoner, normalize_probs


# Default Elo (FIDE-ish: 1500). In a real system these would be loaded
# from a persistent ratings file; Phase 1 starts at the prior.
DEFAULT_ELO = 1500.0

# Blend weights — sum is not constrained; higher = more influence.
WEIGHTS = {
    "elo": 1.0,
    "form": 0.6,
    "h2h": 0.3,
    "injury": 0.4,
    "market": 1.2,
    "weather": 0.1,
}


def _elo_expected(rating_a: float, rating_b: float) -> float:
    """Probability that A beats B, standard Elo formula.

    Retained for backward compat — `run()` uses the full EloState
    when available, falling back to a single-rating call here.
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def _form_score(form) -> float:
    """Win rate (0..1) from FormEntry."""
    if form.played == 0:
        return 0.5
    return (form.won * 3 + form.drawn) / (form.played * 3)


def _h2h_score(h2h: H2HOutput) -> float:
    """Home-team win rate from H2H meetings (perspective: 'home' = home_team_id)."""
    n = h2h.home_wins + h2h.away_wins + h2h.draws
    if n == 0:
        return 0.5
    return h2h.home_wins / n


def _injury_score(inj: InjuryOutput) -> float:
    """Negative penalty: more 'out' injuries → lower score (max -0.5)."""
    n_out = sum(1 for x in (inj.home + inj.away) if x.status == "out")
    return max(-0.5, -0.1 * n_out)


def _market_score(odds: OddsOutput) -> float:
    """Devigged home probability; if vigged poorly, clip to (0.05, 0.95)."""
    p_home = odds.implied_probs.get("home", 1/3)
    return max(0.05, min(0.95, p_home))


def _weather_score(w: WeatherOutput) -> float:
    """0 = neutral; positive = helps home (e.g. wind at away end). For now neutral."""
    if w.playability_risk == "high":
        return -0.1  # high wind/snow hurts attacking football
    return 0.0


def _draw_residual(p_home: float, p_away: float, base_draw: float = 0.27) -> tuple[float, float, float]:
    """Form a 3-way simplex. base_draw is the draw prob when teams are even.

    Step 1: gap = |home - away|. Higher gap → lower draw.
    Step 2: draw = clamp(base_draw - gap * 0.3, 0.10, 0.40)
    Step 3: split (1 - draw) proportionally to p_home, p_away.
    """
    gap = abs(p_home - p_away)
    draw = max(0.10, min(0.40, base_draw - gap * 0.3))
    leftover = 1.0 - draw
    total = p_home + p_away
    if total <= 0:
        return leftover / 2, leftover / 2, draw
    return p_home / total * leftover, p_away / total * leftover, draw


def _confidence(probs: dict[str, float]) -> float:
    """Confidence = (top - second) in 0..1, then scaled by overall decisiveness."""
    sorted_p = sorted(probs.values(), reverse=True)
    margin = sorted_p[0] - sorted_p[1]
    # top-1 also matters: even with a small margin, a 90% favourite is more confident
    decisiveness = sorted_p[0]
    # Combine: 0.6 * margin + 0.4 * (decisiveness - 1/3)
    raw = 0.6 * margin + 0.4 * max(0.0, decisiveness - 1/3)
    return max(0.0, min(1.0, raw))


def _pick(probs: dict[str, float]) -> str:
    return max(probs, key=probs.get)  # type: ignore[arg-type]


def run(context: MatchContext) -> ReasonerOutput:
    """Compute a ReasonerOutput from the assembled context."""
    home_id = context.match.home.id
    away_id = context.match.away.id

    # Elo: prefer the full EloState if the caller provided one
    # (per-team home/away ratings, home advantage, form window).
    # Fall back to the historical 1500/1500 placeholder so the
    # reasoner is safe to run without any pre-computed state.
    elo_state = getattr(context, "elo_state", None)
    elo_p_home: float
    if elo_state is not None:
        from ..elo import predict_proba as _elo_predict
        p_h, p_a, _p_d = _elo_predict(elo_state, home_id, away_id)
        # Drop the draw residual from the 3-way probability; the
        # reasoner reapplies its own draw_residual below.
        total = p_h + p_a
        if total > 0:
            elo_p_home = p_h / total
        else:
            elo_p_home = 0.5
    else:
        elo_p_home = _elo_expected(DEFAULT_ELO, DEFAULT_ELO)

    form_sig = context.signals.get("form_recent")
    h2h_sig = context.signals.get("h2h_history")
    inj_sig = context.signals.get("injury_news")
    odds_sig = context.signals.get("odds_market")
    wx_sig = context.signals.get("weather_venue")

    factors: list[Factor] = []
    warnings: list[str] = []

    if form_sig and form_sig.ok and form_sig.data:
        from ..models import FormOutput  # local to avoid cycles
        f = FormOutput.model_validate(form_sig.data)
        form_home = _form_score(f.home)
        form_away = _form_score(f.away)
        form_diff = form_home - form_away
        factors.append(Factor(name="form_diff", value=form_diff, sign="positive" if form_diff > 0 else "negative", weight=WEIGHTS["form"]))
    else:
        form_diff = 0.0
        if form_sig:
            warnings.append("form_recent signal missing — using neutral")

    if h2h_sig and h2h_sig.ok and h2h_sig.data:
        h2h = H2HOutput.model_validate(h2h_sig.data)
        h2h_score = _h2h_score(h2h)
        factors.append(Factor(name="h2h_home_winrate", value=h2h_score, sign="positive" if h2h_score > 0.5 else "negative", weight=WEIGHTS["h2h"]))
    else:
        h2h_score = 0.5
        if h2h_sig:
            warnings.append("h2h_history signal missing — using neutral")

    if inj_sig and inj_sig.ok and inj_sig.data:
        inj = InjuryOutput.model_validate(inj_sig.data)
        inj_score = _injury_score(inj)
        factors.append(Factor(name="injury_penalty", value=inj_score, sign="negative" if inj_score < 0 else "neutral", weight=WEIGHTS["injury"]))
    else:
        inj_score = 0.0
        if inj_sig:
            warnings.append("injury_news signal missing — using neutral")

    if odds_sig and odds_sig.ok and odds_sig.data:
        odds = OddsOutput.model_validate(odds_sig.data)
        market_p_home = _market_score(odds)
        factors.append(Factor(name="market_p_home", value=market_p_home, sign="positive" if market_p_home > 0.5 else "negative", weight=WEIGHTS["market"]))
    else:
        market_p_home = 1/3
        if odds_sig:
            warnings.append("odds_market signal missing — falling back to Elo+form only")

    if wx_sig and wx_sig.ok and wx_sig.data:
        wx = WeatherOutput.model_validate(wx_sig.data)
        wx_score = _weather_score(wx)
        if wx_score != 0.0:
            factors.append(Factor(name="weather", value=wx_score, sign="negative", weight=WEIGHTS["weather"]))
    else:
        wx_score = 0.0

    # Core: Elo (already computed above using EloState or fallback)
    factors.append(Factor(name="elo_p_home", value=elo_p_home, sign="positive" if elo_p_home > 0.5 else "negative", weight=WEIGHTS["elo"]))

    # Blend into a "logit" for home win (centered at 0.5 for equal teams)
    blend = (
        WEIGHTS["elo"] * (elo_p_home - 0.5)
        + WEIGHTS["form"] * form_diff
        + WEIGHTS["h2h"] * (h2h_score - 0.5)
        + WEIGHTS["injury"] * inj_score
        + WEIGHTS["market"] * (market_p_home - 1/3)
        + WEIGHTS["weather"] * wx_score
    )

    # Sigmoid the blend
    import math
    p_home_raw = 1.0 / (1.0 + math.exp(-blend * 4))  # scale 4 makes weights more impactful
    p_away_raw = 1.0 - p_home_raw
    p_home, p_away, p_draw = _draw_residual(p_home_raw, p_away_raw)
    probs = normalize_probs({"home": p_home, "draw": p_draw, "away": p_away})

    pick = _pick(probs)
    confidence = _confidence(probs)
    rationale = _build_rationale(home_id, away_id, factors, probs, pick)

    return ReasonerOutput(
        reasoner="numeric",
        pick=pick,  # type: ignore[arg-type]
        probs=probs,
        confidence=confidence,
        rationale=rationale,
        factors=factors,
        warnings=warnings,
    )


def _build_rationale(home_id: str, away_id: str, factors: list[Factor], probs: dict[str, float], pick: str) -> str:
    """A one-sentence rationale. Kept deterministic; LLM produces the prose version."""
    if not factors:
        return f"Numeric baseline: no usable signals, defaulting to {pick} ({probs[pick]:.0%})."
    top = max(factors, key=lambda f: abs(f.value) * f.weight)
    return (
        f"Numeric baseline favours {pick} ({probs[pick]:.0%} vs "
        f"{probs['draw']:.0%} draw, {probs['away' if pick == 'home' else 'home']:.0%} other). "
        f"Strongest factor: {top.name}={top.value:+.2f} (weight {top.weight:.1f})."
    )


class NumericReasoner:
    name = "numeric"
    description = "Elo + form + H2H + injury + market blend (deterministic baseline)"
    version = "0.1.0"

    def run(self, context: MatchContext) -> ReasonerOutput:
        return run(context)


# A Reasoner (Protocol) — keep a module-level instance for the orchestrator.
DEFAULT = NumericReasoner()
