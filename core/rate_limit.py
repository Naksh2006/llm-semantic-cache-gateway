"""Per-IP rate limiting for the gateway.

Uses ``slowapi`` (a thin wrapper around the ``limits`` library) to
enforce a requests-per-minute ceiling per client IP.  This protects
against accidental runaway loops, brute-force key guessing, and
general abuse.

Configuration:
  • ``RATE_LIMIT`` env var controls the default limit string.
    Syntax follows the ``limits`` library format:
      "30/minute"    — 30 requests per minute per IP
      "5/second"     — 5 requests per second per IP
      "100/hour"     — 100 requests per hour per IP
    Multiple limits can be combined with a semicolon:
      "5/second;100/hour"
    Default: "30/minute" — generous enough for demos, tight enough
    to prevent bill shock.

Design decisions:
  • Rate limit is per-IP, not per-tenant.  Tenant-level limits would
    require the rate limiter to parse request headers, adding coupling.
    Per-IP is simpler and catches the most common abuse patterns.
  • A custom 429 handler returns JSON (not HTML) so API clients can
    parse the error programmatically.
  • The ``/health`` and ``/docs`` endpoints are NOT rate-limited
    because they don't touch the LLM or cache.
"""

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from core.config import get_settings

logger = structlog.get_logger(__name__)


def _get_rate_limit() -> str:
    """Read the rate limit string from settings."""
    return get_settings().RATE_LIMIT


# ── Limiter instance (per-IP keying) ──────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],           # no default — we apply explicitly
    storage_uri="memory://",     # in-memory store, no Redis needed
)


def _rate_limit_exceeded_handler(
    request: Request, exc: RateLimitExceeded
) -> JSONResponse:
    """Return a clean JSON 429 instead of slowapi's default HTML."""
    logger.warning(
        "rate_limit_exceeded",
        client=request.client.host if request.client else "unknown",
        path=request.url.path,
        detail=str(exc.detail),
    )
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded. Please slow down.",
            "detail": str(exc.detail),
        },
    )


def setup_rate_limiting(app: FastAPI) -> None:
    """Wire rate limiting into the FastAPI application.

    Call this once during app setup (in main.py).  It:
      1. Attaches the limiter's state to the app.
      2. Registers the custom 429 error handler.
    """
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
