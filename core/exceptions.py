"""Custom exception hierarchy for the LLM Semantic Cache Gateway.

Four domain exceptions cover the entire error surface:

• **VectorDBError** — any Qdrant failure (network, timeout, missing
  collection, malformed filter, etc.).  This is the ONLY exception the
  circuit breaker watches for.
• **EmbeddingError** — failures during local embedding generation
  (model load, tokenisation, OOM).
• **StreamingError** — failures while streaming LLM responses back to
  the caller (broken pipe, upstream disconnect).
• **ConfigurationError** — raised when a required configuration value
  is missing or invalid (e.g. no LLM provider API key set).
"""


class VectorDBError(Exception):
    """Raised on any failure from the Qdrant persistence layer.

    The circuit breaker monitors *only* this exception type when deciding
    whether to trip open.
    """

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.original_error = original_error
        super().__init__(message)

    def __str__(self) -> str:
        if self.original_error:
            return f"{self.message} (caused by {self.original_error!r})"
        return self.message


class EmbeddingError(Exception):
    """Raised when local embedding generation fails."""

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.original_error = original_error
        super().__init__(message)

    def __str__(self) -> str:
        if self.original_error:
            return f"{self.message} (caused by {self.original_error!r})"
        return self.message


class StreamingError(Exception):
    """Raised on failures while streaming LLM responses to the client."""

    def __init__(
        self,
        message: str,
        original_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.original_error = original_error
        super().__init__(message)

    def __str__(self) -> str:
        if self.original_error:
            return f"{self.message} (caused by {self.original_error!r})"
        return self.message


class ConfigurationError(Exception):
    """Raised when a required configuration value is missing or invalid.

    Typically raised at first access (not import time) when no LLM
    provider API key is found in the environment.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)

