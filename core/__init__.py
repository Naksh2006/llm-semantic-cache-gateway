"""Core package — configuration, custom exceptions, and shared utilities."""

from core.config import get_settings, settings
from core.exceptions import EmbeddingError, StreamingError, VectorDBError

__all__ = [
    "get_settings",
    "settings",
    "VectorDBError",
    "EmbeddingError",
    "StreamingError",
]

