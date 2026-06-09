"""LLM client abstraction.

Implementations:
  - StubLLMClient:     deterministic, keyless; "leans home" with noise
  - OpenAICompatClient: single client for any OpenAI-shaped endpoint
                        (OpenAI, OpenRouter, Ollama, llama-server, vLLM,
                        LM Studio, etc.)
  - OpenAIClient:      thin subclass of OpenAICompatClient for back-compat
  - OpenRouterClient:  thin subclass of OpenAICompatClient for back-compat

A single swap is one env var (SOCCER_AGENT_LLM_PROVIDER):
  - stub        (default; no network)
  - openai      (api.openai.com)
  - openrouter  (api.openrouter.ai)
  - ollama      (http://127.0.0.1:11434/v1, local; model defaults to qwen2.5:0.5b)
  - openai-compat (custom; set SOCCER_AGENT_LLM_BASE_URL + SOCCER_AGENT_LLM_MODEL)
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable

import httpx


@runtime_checkable
class LLMClient(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str) -> "LLMResult": ...


class LLMError(RuntimeError):
    """Raised when the LLM provider returns an unusable response."""


class LLMResult:
    def __init__(self, raw: str, parsed: dict[str, Any] | None, model: str):
        self.raw = raw
        self.parsed = parsed
        self.model = model

    def __repr__(self) -> str:
        return f"LLMResult(model={self.model!r}, parsed={'yes' if self.parsed else 'no'})"


# -- stub ---------------------------------------------------------------------


class StubLLMClient:
    """Deterministic keyless client. Used in tests and as a fallback.

    The stub is *not* a no-op -- it parses the context and returns a
    reasonable-looking probs dict (a slight home bias plus factor
    weights from the numeric reasoner). This means the LLM phase of
    eval still has signal, which is what the eval harness needs to
    compare against the numeric baseline.
    """

    name = "stub"
    model = "stub-v0"

    def complete(self, system: str, user: str) -> LLMResult:
        text = user.lower()
        probs = {"home": 0.40, "draw": 0.28, "away": 0.32}
        if "home_team_id" in text or "favour home" in text or "home:" in text:
            home_hits = text.count("home")
            away_hits = text.count("away")
            if home_hits > away_hits:
                probs = {"home": 0.45, "draw": 0.27, "away": 0.28}
            elif away_hits > home_hits:
                probs = {"home": 0.30, "draw": 0.28, "away": 0.42}
        pick = max(probs, key=probs.get)  # type: ignore[arg-type]
        confidence = sorted(probs.values(), reverse=True)[0] - sorted(probs.values(), reverse=True)[1]
        parsed = {
            "pick": pick,
            "probs": probs,
            "confidence": round(max(0.0, min(1.0, confidence)), 3),
            "rationale": (
                "Stub LLM: balanced view based on assembled signals; "
                "leans on the team that appears more often in the context. "
                "Replace with a real provider to get LLM-style reasoning."
            ),
        }
        return LLMResult(raw=json.dumps(parsed), parsed=parsed, model=self.model)


# -- openai-compat (the real implementation) ---------------------------------


class OpenAICompatClient:
    """A single client that speaks the OpenAI /chat/completions wire format.

    Works against:
      - OpenAI:      set api_key from OPENAI_API_KEY
      - OpenRouter:  set api_key from SOCCER_AGENT_LLM_API_KEY
      - Ollama:      api_key is a placeholder; ollama ignores it
      - llama-server / vLLM / LM Studio:  api_key may be a placeholder

    Constructor args (all optional; most come from env):
      name:    short tag used in logs and for differentiation in the factory.
      base_url: e.g. https://api.openai.com/v1, http://127.0.0.1:11434/v1
      model:    e.g. gpt-4o-mini, anthropic/claude-3.5-sonnet, qwen2.5:0.5b
      api_key:  bearer token; ignored by ollama, required by openai/openrouter
      timeout_s: request timeout in seconds
    """

    # Subclasses (or callers via factory) override these:
    name: str = "openai-compat"
    default_model: str = "qwen2.5:0.5b"
    default_base_url: str = "http://127.0.0.1:11434/v1"
    api_key_env: str = "SOCCER_AGENT_LLM_API_KEY"  # for direct api-key lookup
    requires_api_key: bool = False  # ollama does not; openai/openrouter do
    api_key_placeholder: str = "ollama"

    def __init__(
        self,
        name: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 30.0,
    ):
        self.name = name or self.name
        # env vars take precedence over class defaults
        self.base_url = (
            base_url
            or os.environ.get("SOCCER_AGENT_LLM_BASE_URL")
            or self.default_base_url
        )
        self.model = (
            model
            or os.environ.get("SOCCER_AGENT_LLM_MODEL")
            or self.default_model
        )
        if api_key is not None:
            self.api_key = api_key
        else:
            self.api_key = os.environ.get(self.api_key_env, "") or os.environ.get("OPENAI_API_KEY", "")
            if not self.api_key:
                if self.requires_api_key:
                    raise LLMError(
                        f"{self.api_key_env} (or OPENAI_API_KEY) is required for {self.name}"
                    )
                self.api_key = self.api_key_placeholder
        self._timeout = timeout_s

    def complete(self, system: str, user: str) -> LLMResult:
        # Default max_tokens caps output so a rambling local model doesn't
        # burn the iteration loop. The reasoner only needs ~200-400 tokens
        # of structured JSON + a few sentences of rationale. OpenAI / OpenRouter
        # honor `max_tokens`; ollama also honors it (mapped to num_predict).
        # The cap can be overridden via SOCCER_AGENT_LLM_MAX_TOKENS.
        max_tokens_env = os.environ.get("SOCCER_AGENT_LLM_MAX_TOKENS")
        max_tokens = int(max_tokens_env) if max_tokens_env else 512
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }
        # response_format.json_object is an OpenAI-specific feature. ollama
        # silently ignores it for most models, but the model still behaves
        # better if we explicitly ask for JSON in the system prompt. Send
        # the field only for providers that support it.
        if self.name in ("openai", "openrouter", "openai-compat"):
            payload["response_format"] = {"type": "json_object"}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise LLMError(f"{self.name} http {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"{self.name} response missing content: {e}")
        # Try to parse as JSON; tolerate code fences.
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"{self.name} returned non-JSON content: {e}")
        return LLMResult(raw=content, parsed=parsed, model=self.model)


# -- thin back-compat subclasses --------------------------------------------


class OpenRouterClient(OpenAICompatClient):
    name = "openrouter"
    default_model = "anthropic/claude-3.5-sonnet"
    default_base_url = "https://openrouter.ai/api/v1"
    api_key_env = "SOCCER_AGENT_LLM_API_KEY"
    requires_api_key = True

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout_s: float = 30.0, base_url: str | None = None):
        super().__init__(
            name="openrouter",
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    def complete(self, system: str, user: str) -> LLMResult:
        # OpenRouter wants the HTTP-Referer / X-Title headers; the base
        # method doesn't set them. Wrap to add.
        # NOTE: we override only to inject the two extra headers; the
        # rest of the wire format is identical to OpenAICompatClient.
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "soccer-agent",
            "X-Title": "soccer-agent",
        }
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        with httpx.Client(timeout=self._timeout) as client:
            r = client.post(url, json=payload, headers=headers)
        if r.status_code >= 400:
            raise LLMError(f"openrouter http {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as e:
            raise LLMError(f"openrouter response missing content: {e}")
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"openrouter returned non-JSON content: {e}")
        return LLMResult(raw=content, parsed=parsed, model=self.model)


class OpenAIClient(OpenAICompatClient):
    name = "openai"
    default_model = "gpt-4o-mini"
    default_base_url = "https://api.openai.com/v1"
    api_key_env = "OPENAI_API_KEY"
    requires_api_key = True

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 timeout_s: float = 30.0):
        super().__init__(
            name="openai",
            base_url=None,  # use class default
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
        )


# -- factory -----------------------------------------------------------------


def get_client(provider: str | None = None) -> LLMClient:
    """Build the LLM client requested in env, falling back to the stub.

    Supported values for SOCCER_AGENT_LLM_PROVIDER:
      stub, openai, openrouter, ollama, openai-compat
    """
    provider = (provider or os.environ.get("SOCCER_AGENT_LLM_PROVIDER", "stub")).lower()
    if provider == "stub":
        return StubLLMClient()
    if provider == "openrouter":
        return OpenRouterClient()
    if provider == "openai":
        return OpenAIClient()
    if provider == "ollama":
        return OpenAICompatClient(name="ollama")
    if provider == "openai-compat":
        return OpenAICompatClient(name="openai-compat")
    raise LLMError(f"unknown SOCCER_AGENT_LLM_PROVIDER: {provider!r}")
