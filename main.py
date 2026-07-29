"""LLM Semantic Cache Gateway — application entry-point.

Start with:
    uvicorn main:app --reload --port 8000

v2.0.0 architecture:
  • Single persistence layer: Qdrant (vectors + payload metadata).
  • Local embeddings via FastEmbed (no external embedding API).
  • LiteLLM for upstream LLM calls (configured for Gemini).
"""

import sys
from contextlib import asynccontextmanager
from time import perf_counter, time

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.router import api_router
from core.auth import BearerAuthMiddleware
from core.config import get_settings
from core.embeddings import get_embedding
from core.rate_limit import setup_rate_limiting
from db import cache_manager
from db.qdrant_client import get_qdrant_client

# ────────────────────────────────────────────────────────────────────
# Structlog configuration — JSON output with ISO timestamps
# ────────────────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if sys.stderr.isatty()
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────────────
# Module-level counters for /metrics endpoint
# ────────────────────────────────────────────────────────────────────
_counters: dict[str, int] = {
    "requests_total": 0,
    "cache_hits": 0,
    "cache_misses": 0,
}
_boot_time: float = 0.0


# ────────────────────────────────────────────────────────────────────
# Lifespan — startup / shutdown
# ────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup / shutdown lifecycle.

    STARTUP:
      1. Eagerly connect to Qdrant and create collection + indexes.
         A dead Qdrant at boot is FATAL (sys.exit(1)) — unlike per-request
         failures where the circuit breaker degrades gracefully, there is
         no second datastore to fall back to during initialization.
      2. Warm up the FastEmbed ONNX model so the first real request
         doesn't pay cold-start latency.
      3. Log gateway_started with key config values.

    SHUTDOWN:
      Log gateway_shutdown.
    """
    global _boot_time
    settings = get_settings()

    # ── 1. Qdrant connectivity (fatal on failure) ──────────────────
    try:
        await get_qdrant_client()
        logger.info(
            "qdrant_connected",
            url=settings.QDRANT_URL,
            collection=settings.QDRANT_COLLECTION,
        )
    except Exception as e:
        logger.critical(
            "qdrant_startup_failed",
            error=str(e),
            url=settings.QDRANT_URL,
        )
        sys.exit(1)

    # ── 2. Warm up embedding model ─────────────────────────────────
    t_warmup = perf_counter()
    await get_embedding("warmup")
    warmup_ms = (perf_counter() - t_warmup) * 1000
    logger.info(
        "embedding_model_warmed_up",
        model=settings.EMBEDDING_MODEL,
        warmup_ms=round(warmup_ms, 1),
    )

    # ── 3. Log gateway started ─────────────────────────────────────
    _boot_time = time()
    logger.info(
        "gateway_started",
        llm_model=settings.LLM_MODEL,
        embedding_model=settings.EMBEDDING_MODEL,
        embedding_dim=settings.EMBEDDING_DIM,
        default_threshold=settings.DEFAULT_SIMILARITY_THRESHOLD,
        cache_ttl=settings.CACHE_TTL_SECONDS,
        lfu_spike_threshold=settings.LFU_SPIKE_THRESHOLD,
    )

    yield

    # ── Shutdown ───────────────────────────────────────────────────
    logger.info("gateway_shutdown")


# ────────────────────────────────────────────────────────────────────
# FastAPI app
# ────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="LLM Semantic Cache Gateway",
    version="2.0.0",
    description="Semantic caching & routing gateway — unified Qdrant store with local embeddings.",
    lifespan=lifespan,
)


# ── CORS (allow all origins for local dev) ─────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Bearer-token auth (no-op when GATEWAY_API_KEY is empty) ────────
app.add_middleware(BearerAuthMiddleware)

# ── Per-IP rate limiting ───────────────────────────────────────────
setup_rate_limiting(app)


# ────────────────────────────────────────────────────────────────────
# Request tracking middleware
# ────────────────────────────────────────────────────────────────────


@app.middleware("http")
async def request_tracking_middleware(request: Request, call_next):
    """Track per-request metrics and log structured request info.

    Increments module-level _counters based on the X-Cache response
    header set by the /v1/chat endpoint.
    """
    t_start = perf_counter()

    response = await call_next(request)

    duration_ms = (perf_counter() - t_start) * 1000

    # Count only /v1/chat requests for cache metrics
    x_cache = response.headers.get("X-Cache")
    if x_cache:
        _counters["requests_total"] += 1
        if x_cache == "HIT":
            _counters["cache_hits"] += 1
        elif x_cache == "MISS":
            _counters["cache_misses"] += 1

    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
        x_cache=x_cache,
    )

    return response


# ────────────────────────────────────────────────────────────────────
# Top-level endpoints (outside /v1 prefix)
# ────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["system"])
async def health():
    """Liveness probe with Qdrant cache stats."""
    stats = await cache_manager.get_stats()
    return {"status": "ok", "cache": stats}


@app.get("/metrics", tags=["system"])
async def metrics():
    """Basic observability counters and uptime."""
    uptime = time() - _boot_time if _boot_time else 0
    total = max(_counters["requests_total"], 1)
    return {
        "uptime_seconds": round(uptime, 1),
        **_counters,
        "hit_rate_pct": round(_counters["cache_hits"] / total * 100, 2),
    }


# ── Routers ────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/v1")
