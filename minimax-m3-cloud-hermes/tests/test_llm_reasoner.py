"""Tests for the LLM client + LLM reasoner."""

from __future__ import annotations

import json as jsonlib
import os
from datetime import datetime

import pytest

import httpx

from soccer_agent.llm import (
    LLMError,
    OpenAIClient,
    OpenAICompatClient,
    OpenRouterClient,
    StubLLMClient,
    get_client,
)
from soccer_agent.models import (
    FormEntry,
    FormOutput,
    InjuryOutput,
    Match,
    MatchContext,
    OddsOutput,
    ReasonerOutput,
    Signal,
    Team,
)
from soccer_agent.reasoners.llm import LLMReasoner


def _ctx() -> MatchContext:
    return MatchContext(
        match=Match(
            match_id="m1",
            home=Team(id="man_city", name="Manchester City"),
            away=Team(id="real_madrid", name="Real Madrid"),
            kickoff=datetime(2025, 5, 30, 20, 0, 0),
            venue_id="puskas_arena",
            competition="UCL",
        ),
        signals={
            "form_recent": Signal(
                tool="form_recent", ok=True,
                data=FormOutput(
                    home=FormEntry(played=5, won=4, drawn=1, lost=0, gf=12, ga=3, points=13, last5_form_string="WWDLW"),
                    away=FormEntry(played=5, won=2, drawn=1, lost=2, gf=7, ga=8, points=7, last5_form_string="DLWLW"),
                ).model_dump(),
                source="fixture",
            ),
        },
    )


# -- stub client -------------------------------------------------------------


def test_stub_client_returns_parseable_json():
    c = StubLLMClient()
    res = c.complete("system", "user")
    assert res.parsed is not None
    for k in ("pick", "probs", "confidence", "rationale"):
        assert k in res.parsed


def test_stub_client_probs_sum_to_one():
    c = StubLLMClient()
    res = c.complete("system", "user with home: home: home: home: home: home:")
    probs = res.parsed["probs"]
    assert abs(sum(probs.values()) - 1.0) < 1e-9


# -- factory ----------------------------------------------------------------


def test_get_client_stub_default(monkeypatch):
    monkeypatch.delenv("SOCCER_AGENT_LLM_PROVIDER", raising=False)
    c = get_client()
    assert isinstance(c, StubLLMClient)


def test_get_client_stub_explicit(monkeypatch):
    c = get_client("stub")
    assert isinstance(c, StubLLMClient)


def test_get_client_openrouter_missing_key(monkeypatch):
    monkeypatch.setenv("SOCCER_AGENT_LLM_PROVIDER", "openrouter")
    monkeypatch.delenv("SOCCER_AGENT_LLM_API_KEY", raising=False)
    with pytest.raises(LLMError):
        get_client()


def test_get_client_openai_missing_key(monkeypatch):
    monkeypatch.setenv("SOCCER_AGENT_LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        get_client()


def test_get_client_unknown_provider(monkeypatch):
    with pytest.raises(LLMError):
        get_client("not_a_real_provider")


def test_openrouter_client_requires_key(monkeypatch):
    monkeypatch.delenv("SOCCER_AGENT_LLM_API_KEY", raising=False)
    with pytest.raises(LLMError):
        OpenRouterClient()


def test_openai_client_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        OpenAIClient()


# -- openai-compat (ollama / vllm / llama-server / lm-studio) ----------------


def test_openai_compat_client_ollama_defaults(monkeypatch):
    """When the user just sets PROVIDER=ollama, we should default to localhost:11434,
    the qwen2.5:0.5b model, and not require an API key.
    """
    monkeypatch.setenv("SOCCER_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("SOCCER_AGENT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("SOCCER_AGENT_LLM_MODEL", raising=False)
    monkeypatch.delenv("SOCCER_AGENT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = get_client()
    assert isinstance(c, OpenAICompatClient)
    assert c.name == "ollama"
    assert c.base_url == "http://127.0.0.1:11434/v1"
    assert c.model == "qwen2.5:0.5b"
    # Ollama ignores the API key, but the field is set to a placeholder.
    assert c.api_key == "ollama"


def test_openai_compat_client_ollama_respects_model_env(monkeypatch):
    monkeypatch.setenv("SOCCER_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("SOCCER_AGENT_LLM_MODEL", "qwen2.5:1.5b")
    monkeypatch.delenv("SOCCER_AGENT_LLM_BASE_URL", raising=False)
    c = get_client()
    assert c.model == "qwen2.5:1.5b"


def test_openai_compat_client_ollama_respects_base_url_env(monkeypatch):
    monkeypatch.setenv("SOCCER_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("SOCCER_AGENT_LLM_BASE_URL", "http://my-host:11434/v1")
    c = get_client()
    assert c.base_url == "http://my-host:11434/v1"


def test_openai_compat_client_arbitrary_base_url():
    """Smoke: build a client against a custom OpenAI-shaped endpoint with explicit args."""
    c = OpenAICompatClient(
        name="custom",
        base_url="http://localhost:8000/v1",
        model="qwen2.5:0.5b",
        api_key="placeholder",
    )
    assert c.name == "custom"
    assert c.base_url == "http://localhost:8000/v1"
    assert c.model == "qwen2.5:0.5b"


def test_openai_compat_client_sends_correct_request(monkeypatch):
    """Mock httpx and assert the wire format is OpenAI-shaped."""
    import json as jsonlib  # avoid shadowing the local `json` kwarg below
    captured: dict = {}

    def fake_post(self, url, json=None, headers=None, content=None, data=None):  # noqa: ARG001
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        req = httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": jsonlib.dumps(
                                {
                                    "pick": "home",
                                    "probs": {"home": 0.5, "draw": 0.3, "away": 0.2},
                                    "confidence": 0.3,
                                    "rationale": "test",
                                }
                            )
                        }
                    }
                ]
            },
            request=httpx.Request("POST", url),
        )
        return req

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    c = OpenAICompatClient(
        name="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5:0.5b",
        api_key="ollama",
    )
    res = c.complete("system prompt", "user prompt")
    # Wire-level assertions
    assert captured["url"] == "http://127.0.0.1:11434/v1/chat/completions"
    assert captured["json"]["model"] == "qwen2.5:0.5b"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    # Ollama clients should NOT send response_format (only OpenAI-shaped
    # providers honor it). The 0.5B model is more reliable when we don't
    # confuse it with JSON-mode hints.
    assert "response_format" not in captured["json"]
    assert captured["headers"]["Authorization"] == "Bearer ollama"
    # Result assertions
    assert res.parsed is not None
    assert res.parsed["pick"] == "home"
    assert res.model == "qwen2.5:0.5b"


def test_openai_compat_client_openai_provider_sends_response_format(monkeypatch):
    """OpenAI-shaped providers (openai / openrouter / openai-compat) get
    response_format=json_object in the request, which they support.
    """
    captured: dict = {}

    def fake_post(self, url, json=None, headers=None, content=None, data=None):  # noqa: ARG001
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return httpx.Response(
            200,
            text=jsonlib.dumps({
                "choices": [{"message": {"content": '{"pick":"home"}'}}],
                "model": "gpt-4o-mini",
            }),
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    c = OpenAICompatClient(
        name="openai",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        api_key="sk-test",
    )
    c.complete("sys", "user")
    assert captured["json"]["response_format"] == {"type": "json_object"}
    # And max_tokens default is 512.
    assert captured["json"]["max_tokens"] == 512


def test_openai_compat_client_raises_on_http_error(monkeypatch):
    def fake_post(self, url, json=None, headers=None, content=None, data=None):  # noqa: ARG001
        return httpx.Response(
            500,
            text="internal server error",
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    c = OpenAICompatClient(
        name="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5:0.5b",
        api_key="ollama",
    )
    with pytest.raises(LLMError):
        c.complete("sys", "user")


def test_openai_compat_client_raises_on_non_json(monkeypatch):
    def fake_post(self, url, json=None, headers=None, content=None, data=None):  # noqa: ARG001
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "not json at all"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx.Client, "post", fake_post)
    c = OpenAICompatClient(
        name="ollama",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen2.5:0.5b",
        api_key="ollama",
    )
    with pytest.raises(LLMError):
        c.complete("sys", "user")


# -- factory: ollama --------------------------------------------------------


def test_get_client_ollama(monkeypatch):
    monkeypatch.setenv("SOCCER_AGENT_LLM_PROVIDER", "ollama")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = get_client()
    assert isinstance(c, OpenAICompatClient)
    assert c.name == "ollama"


# -- llm reasoner ------------------------------------------------------------


def test_llm_reasoner_with_stub_client():
    r = LLMReasoner(client=StubLLMClient())
    out = r.run(_ctx())
    assert isinstance(out, ReasonerOutput)
    assert out.reasoner == "llm"
    assert out.pick in ("home", "draw", "away")
    assert abs(sum(out.probs.values()) - 1.0) < 1e-9
    assert 0.0 <= out.confidence <= 1.0
    # No fallback warnings
    assert not any("llm_" in w for w in out.warnings)


def test_llm_reasoner_with_failing_client_falls_back():
    class BoomClient:
        name = "boom"
        model = "boom"
        def complete(self, system, user):
            raise LLMError("network down")

    r = LLMReasoner(client=BoomClient())  # type: ignore[arg-type]
    out = r.run(_ctx())
    # Should fall back to numeric, but tagged as llm reasoner? No — fallback returns
    # its own ReasonerOutput with reasoner="numeric". Verify we got a numeric one
    # with the llm warning.
    assert out.reasoner == "numeric"
    assert any("llm_error" in w for w in out.warnings)


def test_llm_reasoner_unparseable_falls_back():
    class BadClient:
        name = "bad"
        model = "bad"
        def complete(self, system, user):
            from soccer_agent.llm import LLMResult
            return LLMResult(raw="not json", parsed=None, model="bad")

    r = LLMReasoner(client=BadClient())  # type: ignore[arg-type]
    out = r.run(_ctx())
    assert out.reasoner == "numeric"
    assert any("llm_unparseable" in w for w in out.warnings)


def test_llm_reasoner_bad_probs_falls_back():
    class BadProbsClient:
        name = "bad"
        model = "bad"
        def complete(self, system, user):
            from soccer_agent.llm import LLMResult
            return LLMResult(
                raw="{}",
                parsed={"pick": "home", "probs": {"home": 0, "draw": 0, "away": 0}, "confidence": 0.5, "rationale": "x"},
                model="bad",
            )

    r = LLMReasoner(client=BadProbsClient())  # type: ignore[arg-type]
    out = r.run(_ctx())
    assert out.reasoner == "numeric"
    assert any("llm_bad_probs" in w for w in out.warnings)


def test_llm_reasoner_pick_overridden_by_argmax_when_invalid():
    class WeirdClient:
        name = "weird"
        model = "weird"
        def complete(self, system, user):
            from soccer_agent.llm import LLMResult
            return LLMResult(
                raw="{}",
                parsed={"pick": "fifty-fifty", "probs": {"home": 0.5, "draw": 0.3, "away": 0.2}, "confidence": 0.5, "rationale": "x"},
                model="weird",
            )

    r = LLMReasoner(client=WeirdClient())  # type: ignore[arg-type]
    out = r.run(_ctx())
    assert out.pick == "home"  # argmax of valid probs


def test_llm_reasoner_confidence_clamped_to_unit_interval():
    class HighConfClient:
        name = "high"
        model = "high"
        def complete(self, system, user):
            from soccer_agent.llm import LLMResult
            return LLMResult(
                raw="{}",
                parsed={"pick": "home", "probs": {"home": 0.9, "draw": 0.05, "away": 0.05}, "confidence": 1.5, "rationale": "x"},
                model="high",
            )

    r = LLMReasoner(client=HighConfClient())  # type: ignore[arg-type]
    out = r.run(_ctx())
    assert 0.0 <= out.confidence <= 1.0


def test_llm_reasoner_uses_fallback_when_no_client_provided_and_no_key(monkeypatch):
    monkeypatch.delenv("SOCCER_AGENT_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("SOCCER_AGENT_LLM_PROVIDER", "openrouter")
    r = LLMReasoner()  # will try to build a client and fail
    # Default factory raises LLMError; LLMReasoner wraps and falls back.
    # BUT we constructed r before get_client() was called. Construction succeeds;
    # run() will hit the error.
    out = r.run(_ctx())
    assert out.reasoner == "numeric"
    assert any("llm_error" in w for w in out.warnings)
