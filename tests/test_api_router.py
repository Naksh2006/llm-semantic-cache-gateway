import pytest
from fastapi import BackgroundTasks

from api import router
from api.router import ChatRequest, _infer_provider_from_model
from core.exceptions import ConfigurationError


def test_infer_provider_from_model_override():
    assert _infer_provider_from_model("gemini/gemini-2.5-flash") == "gemini"
    assert _infer_provider_from_model("gpt-4o-mini") == "openai"
    assert _infer_provider_from_model("anthropic/claude-sonnet-4") == "anthropic"
    assert _infer_provider_from_model("custom/local-model") is None


@pytest.mark.asyncio
async def test_chat_returns_clean_error_when_llm_config_missing(monkeypatch):
    class MissingProviderSettings:
        DEFAULT_SIMILARITY_THRESHOLD = 0.92
        MIN_SIMILARITY_THRESHOLD = 0.70
        MAX_SIMILARITY_THRESHOLD = 0.999

        @property
        def resolved_provider(self):
            raise ConfigurationError("No LLM provider API key configured.")

    async def fake_get_embedding(text):
        return [0.1] * 384

    async def fake_lookup(*args, **kwargs):
        return None

    monkeypatch.setattr(router, "get_settings", lambda: MissingProviderSettings())
    monkeypatch.setattr(router, "get_embedding", fake_get_embedding)
    monkeypatch.setattr(router.cache_manager, "lookup", fake_lookup)

    response = await router.chat(
        ChatRequest(messages=[{"role": "user", "content": "hello"}]),
        BackgroundTasks(),
        x_tenant_id="tenant-a",
        x_cache_threshold=None,
    )

    assert response.status_code == 503
    assert response.body == b'{"error":"llm provider is not configured"}'
