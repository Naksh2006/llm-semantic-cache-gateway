"""LLM Semantic Cache Gateway — application entry-point.

Start with:
    uvicorn main:app --reload --port 8000

v2.0.0 architecture:
  • Single persistence layer: Qdrant (vectors + payload metadata).
  • Local embeddings via FastEmbed (no external embedding API).
  • LiteLLM for upstream LLM calls.
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.router import api_router
from core.config import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup / shutdown lifecycle."""

    settings = get_settings()

    # ── Startup ────────────────────────────────────────────────────
    startup_info = {
        "event": "gateway_startup",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "qdrant_url": settings.QDRANT_URL,
        "qdrant_collection": settings.QDRANT_COLLECTION,
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_dim": settings.EMBEDDING_DIM,
        "llm_model": settings.LLM_MODEL,
    }
    logger.info(json.dumps(startup_info, indent=2))

    yield

    # ── Shutdown ───────────────────────────────────────────────────
    shutdown_info = {
        "event": "gateway_shutdown",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    logger.info(json.dumps(shutdown_info, indent=2))


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

# ── Routers ────────────────────────────────────────────────────────
app.include_router(api_router, prefix="/v1")

