"""LLM reasoner.

Sends the assembled MatchContext to an LLM (OpenRouter / OpenAI / stub)
and parses the structured JSON response. Falls back to the numeric
reasoner's probs if the LLM is unavailable or its response is malformed,
so the orchestrator always has a valid pick.
"""

from __future__ import annotations

import json

from ..llm import LLMClient, LLMError, get_client
from ..models import (
    Factor,
    MatchContext,
    ReasonerOutput,
    ToolErrorPayload,
)
from .base import Reasoner, normalize_probs
from .numeric import NumericReasoner


SYSTEM_PROMPT = """\
You are a football match prediction analyst. Given a structured context
of the match (teams, kickoff, signals from form/injury/h2h/weather/odds),
return a JSON object with these fields:
  - pick:        one of "home", "draw", "away"
  - probs:       {"home": float, "draw": float, "away": float} summing to 1.0
  - confidence:  float in [0, 1]
  - rationale:   a 2-4 sentence plain-English explanation referencing the
                 key signals that drove your pick

Be specific. Reference the actual numbers (form strings, injury counts,
H2H win counts, bookmaker probs). Avoid hedging language. Output JSON only.
"""


def _build_user_prompt(ctx: MatchContext) -> str:
    """Render the MatchContext as a compact prompt-friendly dict."""
    payload = {
        "match_id": ctx.match.match_id,
        "home": {"id": ctx.match.home.id, "name": ctx.match.home.name},
        "away": {"id": ctx.match.away.id, "name": ctx.match.away.name},
        "kickoff": ctx.match.kickoff.isoformat(),
        "venue_id": ctx.match.venue_id,
        "competition": ctx.match.competition,
        "venue": (
            ctx.venue.model_dump(mode="json")
            if ctx.venue is not None else None
        ),
        "signals": {
            name: {"ok": sig.ok, "data": sig.data, "error": sig.error.model_dump() if sig.error else None}
            for name, sig in ctx.signals.items()
        },
    }
    return "Context:\n" + json.dumps(payload, indent=2, default=str)


def _coerce_probs(d: object) -> dict[str, float] | None:
    if not isinstance(d, dict):
        return None
    try:
        p = {k: float(d[k]) for k in ("home", "draw", "away")}
    except (KeyError, TypeError, ValueError):
        return None
    p = {k: max(0.0, v) for k, v in p.items()}
    s = sum(p.values())
    if s <= 0:
        return None
    return {k: v / s for k, v in p.items()}


def _coerce_pick(d: object, probs: dict[str, float]) -> str:
    if isinstance(d, str) and d in ("home", "draw", "away"):
        return d
    return max(probs, key=probs.get)  # type: ignore[arg-type]


def _coerce_confidence(d: object, probs: dict[str, float]) -> float:
    try:
        c = float(d)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        # Fallback: derive from top-1 - top-2 margin.
        sorted_p = sorted(probs.values(), reverse=True)
        c = sorted_p[0] - sorted_p[1]
    return max(0.0, min(1.0, c))


class LLMReasoner:
    name = "llm"
    description = "LLM-via-structured-JSON reasoner (OpenRouter / OpenAI / stub)"
    version = "0.1.0"

    def __init__(self, client: LLMClient | None = None,
                 fallback: Reasoner | None = None,
                 system_prompt: str | None = None):
        self._client: LLMClient | None = client
        self.fallback = fallback or NumericReasoner()
        # Allow the prompt-iteration sweep to override the default
        # system prompt per candidate. `None` means "use the default".
        self._system_prompt_override = system_prompt

    @property
    def client(self) -> LLMClient:
        if self._client is None:
            self._client = get_client()
        return self._client

    def run(self, context: MatchContext) -> ReasonerOutput:  # type: ignore[override]
        system = self._system_prompt_override or SYSTEM_PROMPT
        user = _build_user_prompt(context)
        warnings: list[str] = []
        try:
            result = self.client.complete(system, user)
        except LLMError as e:
            warnings.append(f"llm_error: {e}")
            fb = self.fallback.run(context)
            fb.warnings = list(fb.warnings) + warnings
            return fb

        if result.parsed is None:
            warnings.append("llm_unparseable")
            fb = self.fallback.run(context)
            fb.warnings = list(fb.warnings) + warnings
            return fb

        probs = _coerce_probs(result.parsed.get("probs"))
        if probs is None:
            warnings.append("llm_bad_probs")
            fb = self.fallback.run(context)
            fb.warnings = list(fb.warnings) + warnings
            return fb

        pick = _coerce_pick(result.parsed.get("pick"), probs)
        confidence = _coerce_confidence(result.parsed.get("confidence"), probs)
        rationale = result.parsed.get("rationale") or "LLM rationale missing."
        if not isinstance(rationale, str):
            rationale = str(rationale)

        return ReasonerOutput(
            reasoner=self.name,
            pick=pick,  # type: ignore[arg-type]
            probs=normalize_probs(probs),
            confidence=confidence,
            rationale=rationale,
            factors=[Factor(
                name=f"llm_{self.client.name}",
                value=confidence,
                sign="positive" if confidence > 0.3 else "neutral",
                weight=1.0,
            )],
            warnings=warnings,
        )
