from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from soccer.models import MatchDossier, MatchResult, Prediction
from soccer.reasoning.base import ReasonerError, ReasonResult
from soccer.reasoning.prompt import parse_reason_json, render_prompt

PostJson = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _urllib_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            result: dict[str, Any] = json.loads(resp.read().decode())
            return result
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ReasonerError(f"ollama request failed: {exc}") from exc


class OllamaReasoner:
    name = "ollama"

    def __init__(
        self,
        host: str,
        model: str,
        timeout: float,
        post_json: PostJson = _urllib_post,
    ) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._post = post_json

    def _chat(self, prompt: str, *, json_format: bool) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "seed": 7},
        }
        if json_format:
            payload["format"] = "json"
        data = self._post(f"{self._host}/api/chat", payload, self._timeout)
        try:
            return str(data["message"]["content"])
        except (KeyError, TypeError) as exc:
            raise ReasonerError(f"unexpected ollama response shape: {exc}") from exc

    def predict(self, dossier: MatchDossier) -> ReasonResult:
        content = self._chat(render_prompt(dossier), json_format=True)
        return parse_reason_json(content)

    def self_evaluate(self, prediction: Prediction, result: MatchResult) -> str:
        probs_summary = {k.value: round(v, 3) for k, v in prediction.probs.items()}
        prompt = (
            f"You predicted {prediction.pick.value} with probabilities "
            f"{probs_summary}. "
            f"The actual result was {result.outcome.value} "
            f"({result.home_goals}-{result.away_goals}). "
            "In 2-3 sentences, critique what your reasoning got wrong or right."
        )
        return self._chat(prompt, json_format=False)
