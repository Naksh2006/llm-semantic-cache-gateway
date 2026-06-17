"""Circuit-breaker decorators for the Qdrant persistence layer.

Two decorators handle the two failure modes of the cache:

1. **with_cache_fallback** (reads) — if Qdrant is unreachable on a read,
   the gateway should degrade gracefully by treating it as a cache miss
   (return the fallback value). The request still succeeds via the LLM.

2. **fire_and_forget_with_log** (writes) — if a cache write fails, log
   a warning and move on. Losing a cache entry is acceptable; breaking
   the user's request is not.

Both decorators watch ONLY for ``VectorDBError``. Any other exception
type propagates normally — we never swallow application bugs.
"""

import functools

import structlog

from core.exceptions import VectorDBError

logger = structlog.get_logger(__name__)


def with_cache_fallback(fallback_value=None):
    """Decorator factory for cache-read functions.

    If the wrapped async function raises ``VectorDBError``, log at
    CRITICAL level and return ``fallback_value`` instead of crashing.
    Any other exception propagates normally.
    """

    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except VectorDBError as e:
                logger.critical(
                    "qdrant_layer_failure",
                    error=str(e),
                    fallback=True,
                    function=func.__name__,
                )
                return fallback_value

        return wrapper

    return decorator


def fire_and_forget_with_log(func):
    """Decorator for cache-write functions.

    If the wrapped async function raises ``VectorDBError``, log at
    WARNING level and return ``None``. The caller never sees the error.
    Used for writes where failure is non-critical — losing a cache write
    should never break the request.

    Any other exception propagates normally.
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except VectorDBError as e:
            logger.warning(
                "cache_write_failed",
                error=str(e),
                function=func.__name__,
            )
            return None

    return wrapper
