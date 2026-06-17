"""Local embedding generation via FastEmbed.

All vectorisation happens on-device using the ONNX runtime bundled with
FastEmbed — no external API calls are made. The embedding model is loaded
lazily on first use and cached for the lifetime of the process.
"""

import asyncio

import structlog
from fastembed import TextEmbedding

from core.config import get_settings
from core.exceptions import EmbeddingError
from core.utils import normalize_text

logger = structlog.get_logger(__name__)

# ── Lazy singleton ─────────────────────────────────────────────────
_model: TextEmbedding | None = None


def _get_model() -> TextEmbedding:
    """Return the cached TextEmbedding instance, loading it on first call.

    The model is stored in the module-level ``_model`` variable so that
    subsequent calls skip the (expensive) ONNX model load.
    """
    global _model
    if _model is None:
        settings = get_settings()
        try:
            _model = TextEmbedding(model_name=settings.EMBEDDING_MODEL)
            logger.info(
                "embedding_model_loaded",
                model=settings.EMBEDDING_MODEL,
                dim=settings.EMBEDDING_DIM,
            )
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load embedding model '{settings.EMBEDDING_MODEL}'",
                original_error=e,
            )
    return _model


async def get_embedding(text: str) -> list[float]:
    """Embed a single text string and return a list of floats.

    Steps:
      1. Normalize the text (lowercase, collapse whitespace).
      2. Run FastEmbed inference in a thread pool.
      3. Validate output dimension matches settings.EMBEDDING_DIM.

    Raises:
        EmbeddingError: On model load failure, inference failure, or
            dimension mismatch.
    """
    settings = get_settings()
    normalized = normalize_text(text)

    try:
        model = _get_model()

        # FastEmbed's .embed() is SYNCHRONOUS and CPU-bound (ONNX inference).
        # Running it directly on the event loop would block ALL concurrent
        # requests until inference finishes. We offload to a thread pool via
        # asyncio.to_thread() so the event loop stays responsive.
        def _embed_sync() -> list[float]:
            embeddings = list(model.embed([normalized]))
            return embeddings[0].tolist()

        embedding = await asyncio.to_thread(_embed_sync)

    except EmbeddingError:
        # Re-raise our own errors without double-wrapping.
        raise
    except Exception as e:
        raise EmbeddingError(
            f"Embedding inference failed for text: '{normalized[:80]}...'",
            original_error=e,
        )

    # Validate dimension — a silent mismatch would produce garbage search
    # results that are extremely hard to debug in production.
    if len(embedding) != settings.EMBEDDING_DIM:
        raise EmbeddingError(
            f"Embedding dimension mismatch: expected {settings.EMBEDDING_DIM}, "
            f"got {len(embedding)}. Check EMBEDDING_MODEL and EMBEDDING_DIM."
        )

    return embedding
