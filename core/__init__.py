"""Core package — configuration, custom exceptions, and shared utilities."""

from core.config import get_settings
from core.exceptions import (
    ConfigurationError,
    EmbeddingError,
    StreamingError,
    VectorDBError,
)

__all__ = [
    "get_settings",
    "VectorDBError",
    "EmbeddingError",
    "StreamingError",
    "ConfigurationError",
]
