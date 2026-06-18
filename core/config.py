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
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ── LLM Provider ──────────────────────────────────────────────
    LLM_MODEL: str = "gemini/gemini-2.5-flash"
    LLM_API_KEY: str = ""  # set via .env or environment variable

    # ── Cache Behaviour ───────────────────────────────────────────
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.92
    MIN_SIMILARITY_THRESHOLD: float = 0.70
    MAX_SIMILARITY_THRESHOLD: float = 0.999
    CACHE_TTL_SECONDS: int = 3600
    LFU_SPIKE_THRESHOLD: int = 10
    LFU_SPIKE_TTL_EXTENSION: int = 7200


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (lazy singleton).

    Defers validation until first call so the module can be imported
    safely even when .env is absent (e.g. during testing or IDE indexing).
    """
    return Settings()


# Module-level convenience alias — import as `from core.config import settings`
settings: Settings = get_settings()
