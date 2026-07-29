"""Tests for rate limiting.

Covers:
  1. Requests within the limit succeed normally.
  2. Requests exceeding the limit receive 429 with a JSON error body.
  3. The 429 response contains the expected error message format.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from core.rate_limit import limiter, setup_rate_limiting


def _make_rate_limited_app(rate_limit: str = "2/minute") -> FastAPI:
    """Build a minimal FastAPI app with rate limiting wired in.

    Uses a very low limit (2/minute) so we can trigger 429 in tests
    without making dozens of requests.
    """

    class _FakeSettings:
        RATE_LIMIT = rate_limit

    app = FastAPI()
    setup_rate_limiting(app)

    @app.get("/v1/test")
    @limiter.limit(lambda: _FakeSettings().RATE_LIMIT)
    async def _test_endpoint(request: Request):
        return JSONResponse(content={"ok": True})

    @app.get("/health")
    async def _health():
        return JSONResponse(content={"status": "ok"})

    return app


# ────────────────────────────────────────────────────────────────────
# Scenario 1: Requests within the limit succeed
# ────────────────────────────────────────────────────────────────────


def test_requests_within_limit_succeed():
    """First few requests should pass through normally."""
    client = TestClient(_make_rate_limited_app("5/minute"))

    response = client.get("/v1/test")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


# ────────────────────────────────────────────────────────────────────
# Scenario 2: Exceeding the limit returns 429
# ────────────────────────────────────────────────────────────────────


def test_exceeding_rate_limit_returns_429():
    """After exhausting the limit, subsequent requests get 429."""
    client = TestClient(_make_rate_limited_app("2/minute"))

    # First 2 requests should succeed
    assert client.get("/v1/test").status_code == 200
    assert client.get("/v1/test").status_code == 200

    # Third request should be rate-limited
    response = client.get("/v1/test")
    assert response.status_code == 429


# ────────────────────────────────────────────────────────────────────
# Scenario 3: 429 response has correct JSON structure
# ────────────────────────────────────────────────────────────────────


def test_rate_limit_response_is_json_with_error():
    """The 429 response should be JSON (not HTML) with an error message."""
    client = TestClient(_make_rate_limited_app("1/minute"))

    # Exhaust the limit
    client.get("/v1/test")

    # This one gets rate-limited
    response = client.get("/v1/test")
    assert response.status_code == 429

    body = response.json()
    assert "error" in body
    assert "Rate limit" in body["error"]
    assert "detail" in body
