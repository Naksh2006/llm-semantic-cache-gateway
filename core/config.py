"""Application settings loaded from environment / .env file.

This is the single source of truth for every tuneable parameter in the
gateway. All values can be overridden via environment variables; defaults
are chosen for a local-dev experience (Qdrant on localhost, small BGE
model, generous TTL).

Design note — v2 architecture:
  • ONE persistence layer: Qdrant (vector + payload storage).
  • LOCAL embeddings via FastEmbed (no external embedding-API calls).
  • No REDIS_URL — intentionally absent; cache metadata lives in Qdrant
    payloads alongside the vectors themselves.

Provider resolution (BYOK):
  • Set GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.
  • The gateway auto-detects which provider to use based on which key
    is present. LLM_PROVIDER overrides auto-detection.
  • The legacy LLM_API_KEY field is still supported as a fallback for
    backward compatibility.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from core.exceptions import ConfigurationError


class Settings(BaseSettings):
    """Gateway-wide configuration backed by pydantic-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Qdrant (single persistence layer) ──────────────────────────
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "llm_cache"

    # ── Local Embedding Model ──────────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIM: int = 384

    # ── LLM Provider (legacy single-key field) ─────────────────────
    LLM_MODEL: str = "gemini/gemini-2.5-flash"
    LLM_API_KEY: str = ""  # legacy — still works as fallback

    # ── BYOK Provider Keys ─────────────────────────────────────────
    GEMINI_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    # ── Explicit Provider Override ─────────────────────────────────
    # If set, skips auto-detection. Valid values: "gemini", "openai",
    # "anthropic". If unset, inferred from which API key is present.
    LLM_PROVIDER: str | None = None

    # ── Cache Behaviour ───────────────────────────────────────────
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.92
    MIN_SIMILARITY_THRESHOLD: float = 0.70
    MAX_SIMILARITY_THRESHOLD: float = 0.999
    CACHE_TTL_SECONDS: int = 3600
    LFU_SPIKE_THRESHOLD: int = 10
    LFU_SPIKE_TTL_EXTENSION: int = 7200

    @property
    def resolved_provider(self) -> str:
        """Determine the active LLM provider.

        Resolution order:
          1. LLM_PROVIDER (explicit override) — always wins.
          2. GEMINI_API_KEY set → "gemini"
          3. OPENAI_API_KEY set → "openai"
          4. ANTHROPIC_API_KEY set → "anthropic"
          5. LLM_API_KEY set (legacy) → infer from LLM_MODEL prefix
          6. None found → raise ConfigurationError

        Raises at first access, not at import time, so the app can
        start and serve health checks even with a misconfigured .env.
        """
        # 1. Explicit override wins
        if self.LLM_PROVIDER:
            return self.LLM_PROVIDER.lower()

        # 2-4. Auto-detect from BYOK keys
        if self.GEMINI_API_KEY:
            return "gemini"
        if self.OPENAI_API_KEY:
            return "openai"
        if self.ANTHROPIC_API_KEY:
            return "anthropic"

        # 5. Legacy fallback — infer from LLM_MODEL prefix + LLM_API_KEY
        if self.LLM_API_KEY:
            model_lower = self.LLM_MODEL.lower()
            if "gemini" in model_lower:
                return "gemini"
            if "claude" in model_lower or "anthropic" in model_lower:
                return "anthropic"
            # Default to openai for gpt-* or any other litellm model
            return "openai"

        # 6. No key found
        raise ConfigurationError(
            "No LLM provider API key configured. Set one of: "
            "GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "or the legacy LLM_API_KEY in your .env file."
        )

    @property
    def resolved_api_key(self) -> str:
        """Return the API key for the resolved provider.

        Checks provider-specific keys first, falls back to LLM_API_KEY.
        """
        provider = self.resolved_provider
        key_map = {
            "gemini": self.GEMINI_API_KEY,
            "openai": self.OPENAI_API_KEY,
            "anthropic": self.ANTHROPIC_API_KEY,
        }
        return key_map.get(provider) or self.LLM_API_KEY


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (lazy singleton).

    Defers validation until first call so the module can be imported
    safely even when .env is absent (e.g. during testing or IDE indexing).
    """
    return Settings()
