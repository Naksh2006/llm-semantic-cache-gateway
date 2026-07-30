"""Async Qdrant client wrapper — the SOLE entry-point to Qdrant.

This module is the only place in the gateway that imports the Qdrant SDK.
Every SDK / network exception is translated into ``VectorDBError`` so the
circuit-breaker decorators have a single exception type to watch.

Connection lifecycle:
  • ``get_qdrant_client()`` lazily creates the client on first call,
    ensures the collection exists with the correct vector config, and
    creates payload indexes for fast filtered lookups.
  • The client is cached at module level for the lifetime of the process.

Decorator usage:
  • Read functions (exact_match_lookup, vector_search) are wrapped with
    ``@with_cache_fallback`` — Qdrant downtime degrades to a cache miss.
  • Write functions (upsert_point, update_payload, delete_point) are
    wrapped with ``@fire_and_forget_with_log`` — a failed write is logged
    but never breaks the user's request.
"""

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from core.circuit_breaker import fire_and_forget_with_log, with_cache_fallback
from core.config import get_settings
from core.exceptions import VectorDBError

logger = structlog.get_logger(__name__)

# ── Lazy singleton ─────────────────────────────────────────────────
_client: AsyncQdrantClient | None = None


async def get_qdrant_client() -> AsyncQdrantClient:
    """Return the cached AsyncQdrantClient, initialising on first call.

    On first invocation:
      1. Connect to Qdrant at ``settings.QDRANT_URL``.
      2. Create the collection if it doesn't exist (cosine distance,
         vector size = ``settings.EMBEDDING_DIM``).
      3. Create keyword payload indexes on ``tenant_id`` and
         ``query_hash`` so filtered lookups avoid full collection scans.
    """
    global _client
    if _client is not None:
        return _client

    settings = get_settings()

    try:
        client = AsyncQdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,  # None for local, key for Cloud
        )

        # ── Ensure collection exists ───────────────────────────────
        collections = await client.get_collections()
        existing_names = [c.name for c in collections.collections]

        if settings.QDRANT_COLLECTION not in existing_names:
            await client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "qdrant_collection_created",
                collection=settings.QDRANT_COLLECTION,
                dim=settings.EMBEDDING_DIM,
            )

        # ── Create payload indexes for fast filtered lookups ───────
        # These are idempotent — Qdrant silently ignores duplicates.
        for field_name in ("tenant_id", "query_hash"):
            await client.create_payload_index(
                collection_name=settings.QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )

        logger.info(
            "qdrant_client_initialized",
            url=settings.QDRANT_URL,
            collection=settings.QDRANT_COLLECTION,
        )

        _client = client
        return _client

    except Exception as e:
        raise VectorDBError(
            "Failed to initialize Qdrant client", original_error=e
        )


# ────────────────────────────────────────────────────────────────────
# READ PATH — protected by @with_cache_fallback
# ────────────────────────────────────────────────────────────────────


@with_cache_fallback(fallback_value=None)
async def exact_match_lookup(
    tenant_id: str, query_hash: str
) -> dict | None:
    """Look up a cache entry by exact tenant + query hash.

    Uses a filtered scroll (not vector search) — this is the fast path
    for queries we've seen before with identical normalised text.

    Returns the point's payload dict with ``_point_id`` injected,
    or ``None`` if no match.
    """
    try:
        client = await get_qdrant_client()
        settings = get_settings()

        results, _next_offset = await client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id),
                    ),
                    FieldCondition(
                        key="query_hash",
                        match=MatchValue(value=query_hash),
                    ),
                ]
            ),
            limit=1,
        )

        if not results:
            return None

        point = results[0]
        if point.payload is None:
            return None
        payload = dict(point.payload)
        payload["_point_id"] = str(point.id)
        return payload

    except VectorDBError:
        raise
    except Exception as e:
        raise VectorDBError("exact_match_lookup failed", original_error=e)


@with_cache_fallback(fallback_value=None)
async def vector_search(
    tenant_id: str,
    embedding: list[float],
    threshold: float,
    limit: int = 1,
) -> dict | None:
    """Search for semantically similar cached queries via cosine distance.

    Only entries belonging to ``tenant_id`` are considered.
    ``score_threshold`` filters out results below the similarity bar.

    Returns the top hit's payload dict with ``_point_id`` and ``_score``
    injected, or ``None`` if nothing exceeds the threshold.

    Uses ``client.query_points()`` (qdrant-client v2 API — the legacy
    ``client.search()`` was removed).
    """
    try:
        client = await get_qdrant_client()
        settings = get_settings()

        # query_points() returns a QueryResponse with a .points list
        # of ScoredPoint objects (each has .id, .score, .payload).
        response = await client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=embedding,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="tenant_id",
                        match=MatchValue(value=tenant_id),
                    ),
                ]
            ),
            score_threshold=threshold,
            limit=limit,
        )

        if not response.points:
            return None

        hit = response.points[0]
        if hit.payload is None:
            return None
        payload = dict(hit.payload)
        payload["_point_id"] = str(hit.id)
        payload["_score"] = hit.score
        return payload

    except VectorDBError:
        raise
    except Exception as e:
        raise VectorDBError("vector_search failed", original_error=e)


# ────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ────────────────────────────────────────────────────────────────────


@with_cache_fallback(fallback_value=None)
async def _get_raw_payload(point_id: str) -> dict | None:
    """Fetch a single point's payload by ID.

    Used by cache_manager.record_hit() to read the current lfu_count
    before incrementing (Qdrant's set_payload does a literal overwrite,
    not an atomic increment).
    """
    try:
        client = await get_qdrant_client()
        settings = get_settings()

        results = await client.retrieve(
            collection_name=settings.QDRANT_COLLECTION,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )

        if not results:
            return None

        point = results[0]
        if point.payload is None:
            return None
        payload = dict(point.payload)
        payload["_point_id"] = str(point.id)
        return payload

    except VectorDBError:
        raise
    except Exception as e:
        raise VectorDBError("_get_raw_payload failed", original_error=e)


# ────────────────────────────────────────────────────────────────────
# WRITE PATH — protected by @fire_and_forget_with_log
# ────────────────────────────────────────────────────────────────────


@fire_and_forget_with_log
async def upsert_point(
    point_id: str, embedding: list[float], payload: dict
) -> None:
    """Insert or overwrite a cache entry (vector + payload)."""
    try:
        client = await get_qdrant_client()
        settings = get_settings()

        await client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
    except VectorDBError:
        raise
    except Exception as e:
        raise VectorDBError("upsert_point failed", original_error=e)


@fire_and_forget_with_log
async def update_payload(
    point_id: str, payload_updates: dict
) -> None:
    """Partially update a point's payload without touching the vector.

    This is a set_payload call, NOT a full upsert — only the keys
    present in ``payload_updates`` are modified; all other payload
    fields and the vector remain untouched.
    """
    try:
        client = await get_qdrant_client()
        settings = get_settings()

        await client.set_payload(
            collection_name=settings.QDRANT_COLLECTION,
            payload=payload_updates,
            points=[point_id],
        )
    except VectorDBError:
        raise
    except Exception as e:
        raise VectorDBError("update_payload failed", original_error=e)


@fire_and_forget_with_log
async def delete_point(point_id: str) -> None:
    """Delete a cache entry by point ID — used for evicting expired entries."""
    try:
        client = await get_qdrant_client()
        settings = get_settings()

        await client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=PointIdsList(points=[point_id]),
        )
    except VectorDBError:
        raise
    except Exception as e:
        raise VectorDBError("delete_point failed", original_error=e)
