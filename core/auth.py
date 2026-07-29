"""Bearer-token authentication middleware.

Protects the gateway from unauthorized access by requiring a valid
``Authorization: Bearer <key>`` header on every request.

Design decisions:
  • **Skip list**: ``/health``, ``/docs``, ``/redoc``, ``/openapi.json``
    remain public so monitoring probes and API explorers work without
    credentials.
  • **Opt-in enforcement**: If ``GATEWAY_API_KEY`` is empty/unset, the
    middleware is a no-op — this keeps local development frictionless
    (no key needed on localhost) while ensuring production deployments
    are locked down.
  • **Constant-time comparison**: Uses ``hmac.compare_digest`` to avoid
    timing-based side-channel attacks on the key value.
"""

import hmac

import structlog
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import (
    BaseHTTPMiddleware,
    RequestResponseEndpoint,
)
from starlette.responses import Response

from core.config import get_settings

logger = structlog.get_logger(__name__)

# Endpoints that never require authentication.
# Health is needed by monitoring probes; docs/openapi are for API
# exploration during development and demos.
_PUBLIC_PATHS: set[str] = {
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Reject requests that don't carry a valid Bearer token.

    Usage (in main.py):
        from core.auth import BearerAuthMiddleware
        app.add_middleware(BearerAuthMiddleware)
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        settings = get_settings()

        # ── No key configured → skip auth (local dev mode) ─────────
        if not settings.GATEWAY_API_KEY:
            return await call_next(request)

        # ── Public endpoints → always allowed ──────────────────────
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        # ── Extract and validate the Bearer token ──────────────────
        auth_header = request.headers.get("authorization", "")

        if not auth_header.startswith("Bearer "):
            logger.warning(
                "auth_rejected",
                reason="missing_bearer_token",
                path=request.url.path,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"error": "Missing or invalid Authorization header. "
                         "Expected: Bearer <your-api-key>"},
            )

        provided_key = auth_header[7:]  # strip "Bearer " prefix

        if not hmac.compare_digest(provided_key, settings.GATEWAY_API_KEY):
            logger.warning(
                "auth_rejected",
                reason="invalid_api_key",
                path=request.url.path,
                client=request.client.host if request.client else "unknown",
            )
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key."},
            )

        # ── Key is valid → proceed ─────────────────────────────────
        return await call_next(request)
