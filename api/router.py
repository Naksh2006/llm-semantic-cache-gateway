"""FastAPI router exposing the /v1 API surface.

Planned endpoints (v2):

  POST /v1/chat/completions
      • OpenAI-compatible request body.
      • Checks semantic cache first; on miss forwards to the LLM.
      • Supports ``stream: true`` via SSE (delegates to ``streaming.py``).
      • Accepts optional ``similarity_threshold`` override per request.

  GET /v1/cache/stats
      • Returns cache hit/miss counters and collection info.

  DELETE /v1/cache
      • Purges the entire Qdrant collection (admin use).

  GET /v1/health
      • Liveness probe — checks Qdrant connectivity.
"""

from fastapi import APIRouter

api_router = APIRouter()


@api_router.get("/health", tags=["system"])
async def health_check():
    """Liveness probe — placeholder."""
    return {"status": "ok"}
