import pytest

from core.config import Settings
from core.exceptions import ConfigurationError


def test_provider_auto_detection_priority():
    settings = Settings(
        _env_file=None,
        GEMINI_API_KEY="gemini-key",
        OPENAI_API_KEY="openai-key",
        ANTHROPIC_API_KEY="anthropic-key",
    )

    assert settings.resolved_provider == "gemini"
    assert settings.resolved_api_key == "gemini-key"


def test_invalid_provider_override_raises_configuration_error():
    settings = Settings(
        _env_file=None,
        LLM_PROVIDER="unknown",
        GEMINI_API_KEY=None,
        OPENAI_API_KEY=None,
        ANTHROPIC_API_KEY=None,
        LLM_API_KEY="",
    )

    with pytest.raises(ConfigurationError, match="Invalid LLM_PROVIDER"):
        _ = settings.resolved_provider


def test_missing_provider_key_raises_configuration_error():
    settings = Settings(
        _env_file=None,
        GEMINI_API_KEY=None,
        OPENAI_API_KEY=None,
        ANTHROPIC_API_KEY=None,
        LLM_API_KEY="",
    )

    with pytest.raises(ConfigurationError, match="No LLM provider API key"):
        _ = settings.resolved_provider


def test_api_key_for_provider_requires_matching_key():
    settings = Settings(
        _env_file=None,
        GEMINI_API_KEY="gemini-key",
        OPENAI_API_KEY=None,
        ANTHROPIC_API_KEY=None,
        LLM_API_KEY="",
    )

    with pytest.raises(ConfigurationError, match="No API key configured"):
        settings.api_key_for_provider("openai")
