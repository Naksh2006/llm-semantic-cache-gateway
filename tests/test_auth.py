"""Tests for the BearerAuthMiddleware.

Covers five scenarios:
  1. Auth disabled (no GATEWAY_API_KEY) → all requests pass through.
  2. Valid Bearer token → request proceeds.
  3. Missing Authorization header → 401.
  4. Invalid/wrong API key → 401.
  5. Public endpoints (/health, /docs) → bypass auth even when key is set.
"""

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import core.auth as auth_module
from core.auth import BearerAuthMiddleware


class _FakeSettings:
    """Minimal stand-in for Settings — only exposes what auth needs."""

    def __init__(self, gateway_api_key: str = "") -> None:
        self.GATEWAY_API_KEY = gateway_api_key


def _make_app() -> FastAPI:
    """Build a minimal FastAPI app with the auth middleware wired in."""
    app = FastAPI()
    app.add_middleware(BearerAuthMiddleware)

    @app.get("/v1/chat")
    async def _dummy_chat():
        return JSONResponse(content={"ok": True})

    @app.get("/health")
    async def _dummy_health():
        return JSONResponse(content={"status": "ok"})

    @app.get("/docs")
    async def _dummy_docs():
        return JSONResponse(content={"docs": True})

    return app


def _set_key(monkeypatch, key: str) -> None:
    """Patch get_settings to return a FakeSettings with the given key."""
    monkeypatch.setattr(
        auth_module, "get_settings", lambda: _FakeSettings(key)
    )


# ────────────────────────────────────────────────────────────────────
# Scenario 1: Auth disabled (GATEWAY_API_KEY is empty)
# ────────────────────────────────────────────────────────────────────


def test_auth_disabled_when_no_key_configured(monkeypatch):
    """With no GATEWAY_API_KEY set, requests pass through without auth."""
    _set_key(monkeypatch, "")
    client = TestClient(_make_app())

    response = client.get("/v1/chat")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


# ────────────────────────────────────────────────────────────────────
# Scenario 2: Valid Bearer token
# ────────────────────────────────────────────────────────────────────


def test_valid_bearer_token_allows_request(monkeypatch):
    """A correct Bearer token should let the request through."""
    _set_key(monkeypatch, "sk-test-secret")
    client = TestClient(_make_app())

    response = client.get(
        "/v1/chat",
        headers={"Authorization": "Bearer sk-test-secret"},
    )
    assert response.status_code == 200
    assert response.json() == {"ok": True}


# ────────────────────────────────────────────────────────────────────
# Scenario 3: Missing Authorization header
# ────────────────────────────────────────────────────────────────────


def test_missing_auth_header_returns_401(monkeypatch):
    """No Authorization header → 401 with a helpful error message."""
    _set_key(monkeypatch, "sk-test-secret")
    client = TestClient(_make_app())

    response = client.get("/v1/chat")
    assert response.status_code == 401
    assert "Authorization" in response.json()["error"]


# ────────────────────────────────────────────────────────────────────
# Scenario 4: Wrong API key
# ────────────────────────────────────────────────────────────────────


def test_wrong_api_key_returns_401(monkeypatch):
    """An incorrect Bearer token → 401."""
    _set_key(monkeypatch, "sk-test-secret")
    client = TestClient(_make_app())

    response = client.get(
        "/v1/chat",
        headers={"Authorization": "Bearer sk-wrong-key"},
    )
    assert response.status_code == 401
    assert "Invalid" in response.json()["error"]


def test_non_bearer_auth_scheme_returns_401(monkeypatch):
    """Using Basic auth or another scheme instead of Bearer → 401."""
    _set_key(monkeypatch, "sk-test-secret")
    client = TestClient(_make_app())

    response = client.get(
        "/v1/chat",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
    )
    assert response.status_code == 401


# ────────────────────────────────────────────────────────────────────
# Scenario 5: Public endpoints bypass auth
# ────────────────────────────────────────────────────────────────────


def test_health_endpoint_bypasses_auth(monkeypatch):
    """/health should always be accessible (monitoring probes)."""
    _set_key(monkeypatch, "sk-test-secret")
    client = TestClient(_make_app())

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_docs_endpoint_bypasses_auth(monkeypatch):
    """/docs should always be accessible (API exploration)."""
    _set_key(monkeypatch, "sk-test-secret")
    client = TestClient(_make_app())

    response = client.get("/docs")
    assert response.status_code == 200
