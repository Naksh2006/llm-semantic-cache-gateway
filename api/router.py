"""FastAPI router exposing the /v1 API surface.

Endpoints:
  POST /v1/chat
      • Accepts messages + optional model override.
      • Requires ``x-tenant-id`` header — tenant isolation starts here;
        there is no "default tenant" fallback.
      • Supports optional ``x-cache-threshold`` header for per-request
        similarity tuning (clamped to configured min/max).
      • On HIT: returns JSONResponse with cached content + X-Cache: HIT.
      • On MISS: routes query to the optimal model tier based on
        complexity scoring, streams SSE response + X-Cache: MISS, then
        fires a background Qdrant write via on_complete callback.

  GET /v1/cache/stats
      • Returns collection info and point count.

  GET /v1/health
      • Liveness probe — checks Qdrant connectivity.
"""

import structlog
from fastapi import APIRouter, BackgroundTasks, Header
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from api.streaming import stream_and_capture
from core.config import get_settings
from core.embeddings import get_embedding
from core.exceptions import ConfigurationError, EmbeddingError
from core.router_engine import get_model_spec, route
from db import cache_manager

logger = structlog.get_logger(__name__)

api_router = APIRouter()


# ────────────────────────────────────────────────────────────────────
# Request model
# ────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Incoming chat completion request body."""

    messages: list[dict]
    model: str | None = None


def _infer_provider_from_model(model: str) -> str | None:
    """Infer a provider from common LiteLLM model identifiers."""
    model_lower = model.lower()

    if model_lower.startswith("gemini/") or "gemini" in model_lower:
        return "gemini"
    if model_lower.startswith("anthropic/") or "claude" in model_lower:
        return "anthropic"
    if model_lower.startswith(("openai/", "gpt-", "o1", "o3", "o4")):
        return "openai"

    return None


# ────────────────────────────────────────────────────────────────────
# POST /v1/chat
# ────────────────────────────────────────────────────────────────────


@api_router.post("/chat", tags=["chat"], response_model=None)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    x_tenant_id: str = Header(..., alias="x-tenant-id"),
    x_cache_threshold: float | None = Header(
        None, alias="x-cache-threshold"
    ),
):
    """Semantic-cache-aware chat completion endpoint.

    1. Enforce tenant isolation (no empty tenant IDs).
    2. Resolve and clamp the similarity threshold.
    3. Generate embedding for the user's query.
    4. Check cache: exact hash first, then semantic vector search.
    5. HIT  → return cached response as JSON, bump LFU in background.
    6. MISS → route to optimal model, stream LLM response via SSE,
       compute cost estimate, cache full text on completion.
    """
    settings = get_settings()

    # ── 1. Tenant isolation — no "default tenant" fallback ─────────
    if not x_tenant_id or not x_tenant_id.strip():
        return JSONResponse(
            status_code=400,
            content={"error": "x-tenant-id header is required"},
        )

    # ── 2. Resolve and clamp similarity threshold ──────────────────
    raw_threshold = x_cache_threshold
    threshold = (
        x_cache_threshold
        if x_cache_threshold is not None
        else settings.DEFAULT_SIMILARITY_THRESHOLD
    )
    threshold = max(
        settings.MIN_SIMILARITY_THRESHOLD,
        min(settings.MAX_SIMILARITY_THRESHOLD, threshold),
    )

    if raw_threshold is not None and raw_threshold != threshold:
        logger.warning(
            "threshold_clamped",
            raw=raw_threshold,
            clamped=threshold,
            min=settings.MIN_SIMILARITY_THRESHOLD,
            max=settings.MAX_SIMILARITY_THRESHOLD,
            tenant_id=x_tenant_id,
        )

    # ── 3. Extract the last user message ───────────────────────────
    last_user_content = request.messages[-1]["content"]

    # ── 4. Generate embedding ──────────────────────────────────────
    try:
        query_vec = await get_embedding(last_user_content)
    except EmbeddingError:
        return JSONResponse(
            status_code=502,
            content={"error": "embedding service unavailable"},
        )

    # ── 5. Cache lookup ────────────────────────────────────────────
    try:
        result = await cache_manager.lookup(
            x_tenant_id, last_user_content, query_vec, threshold
        )
    except Exception as e:
        logger.critical(
            "cache_lookup_failed",
            error=str(e),
            tenant_id=x_tenant_id,
            fallback=True,
        )
        result = None

    # ── 6. HIT → return cached response ────────────────────────────
    if result is not None:
        background_tasks.add_task(
            cache_manager.record_hit, result.vector_id
        )
        logger.info(
            "cache_hit",
            tenant_id=x_tenant_id,
            match_type=result.match_type,
            score=result.score,
            vector_id=result.vector_id,
        )
        return JSONResponse(
            status_code=200,
            content={
                "content": result.response_text,
                "cached": True,
                "match_type": result.match_type,
            },
            headers={
                "X-Cache": "HIT",
                "X-Match-Type": result.match_type,
            },
        )

    # ── 7. MISS → route to optimal model, stream from LLM ─────────
    #
    # Cost-aware routing: score the query complexity and pick the
    # cheapest model tier that can handle it well.
    try:
        provider = settings.resolved_provider
        routing_decision = await route(last_user_content, provider)
        model = request.model or routing_decision.model_id
        model_provider = (
            _infer_provider_from_model(request.model)
            if request.model
            else provider
        )
        api_key_provider = model_provider or provider
        api_key = settings.api_key_for_provider(api_key_provider)
    except ConfigurationError as e:
        logger.warning(
            "llm_configuration_error",
            error=str(e),
            tenant_id=x_tenant_id,
        )
        return JSONResponse(
            status_code=503,
            content={"error": "llm provider is not configured"},
        )

    logger.info(
        "routing_decision",
        **routing_decision.model_dump(),
    )

    # Look up the cost spec for the selected model (may be None if
    # the user overrode with a custom model string).
    model_spec = get_model_spec(model, api_key_provider)

    # IMPORTANT: background_tasks.add_task inside _on_complete works
    # correctly here because _on_complete is awaited from within the
    # same request's generator (in stream_and_capture) before the
    # StreamingResponse finishes closing. FastAPI runs registered
    # background tasks AFTER the response body is fully sent, so the
    # Qdrant write still happens without blocking the client's
    # perceived latency.
    async def _on_complete(full_response: str) -> None:
        # ── Approximate cost estimation ────────────────────────
        # LiteLLM's streaming mode does NOT reliably expose token
        # usage in the final chunk for all providers. As a known
        # limitation, we approximate token counts using word-split
        # as a proxy: ~1 word ≈ 1.3 tokens (conservative estimate).
        input_text = last_user_content
        input_tokens_approx = int(len(input_text.split()) * 1.3)
        output_tokens_approx = int(len(full_response.split()) * 1.3)

        cost_usd = 0.0
        if model_spec:
            cost_usd = (
                (input_tokens_approx / 1000)
                * model_spec.cost_per_1k_input
                + (output_tokens_approx / 1000)
                * model_spec.cost_per_1k_output
            )

        logger.info(
            "cost_estimate",
            model=model,
            provider=api_key_provider,
            input_tokens_approx=input_tokens_approx,
            output_tokens_approx=output_tokens_approx,
            cost_usd=round(cost_usd, 6),
            tenant_id=x_tenant_id,
        )

        background_tasks.add_task(
            cache_manager.store,
            x_tenant_id,
            last_user_content,
            query_vec,
            full_response,
        )

    generator = stream_and_capture(
        request.messages, model, _on_complete, api_key
    )

    logger.info(
        "cache_miss",
        tenant_id=x_tenant_id,
        model=model,
        provider=api_key_provider,
        tier=routing_decision.tier,
        complexity=routing_decision.complexity_score,
    )

    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"X-Cache": "MISS", "Cache-Control": "no-cache"},
    )


# ────────────────────────────────────────────────────────────────────
# GET /v1/cache/stats
# ────────────────────────────────────────────────────────────────────


@api_router.get("/cache/stats", tags=["cache"])
async def cache_stats():
    """Return collection-level stats from Qdrant."""
    return await cache_manager.get_stats()


# ────────────────────────────────────────────────────────────────────
# GET /v1/health
# ────────────────────────────────────────────────────────────────────


@api_router.get("/health", tags=["system"])
async def health_check():
    """Liveness probe — checks Qdrant connectivity."""
    stats = await cache_manager.get_stats()
    if stats.get("status") == "ok":
        return {"status": "ok", "qdrant": "connected"}
    return JSONResponse(
        status_code=503,
        content={"status": "degraded", "qdrant": "unreachable"},
    )
