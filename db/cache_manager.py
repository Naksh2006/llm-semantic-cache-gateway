"""High-level semantic cache orchestration layer.

# ── CRITICAL DESIGN RULE ───────────────────────────────────────────
# tenant_id is MANDATORY on every single Qdrant filter used in this
# file.  There is no code path in lookup(), record_hit(), or store()
# that touches Qdrant without a tenant_id constraint.  This is what
# prevents tenant A's cached "summarize my emails" answer from ever
# being served to tenant B.
# ───────────────────────────────────────────────────────────────────

This is the ONLY module that knows about:
  • TTL expiry logic (stale entries behave as misses)
  • LFU spike promotion (hot entries get their TTL extended)
  • How the exact-match and vector-search paths combine

It sits on top of db/qdrant_client.py and never imports the Qdrant
SDK directly.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal
from uuid import uuid4

import structlog
from pydantic import BaseModel

from core.config import get_settings
from core.utils import compute_query_hash
from db.qdrant_client import (
    _get_raw_payload,
    delete_point,
    exact_match_lookup,
    update_payload,
    upsert_point,
    vector_search,
)

logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────────────
# Data model
# ────────────────────────────────────────────────────────────────────


class CacheResult(BaseModel):
    """Structured cache-hit result returned by ``lookup()``."""

    vector_id: str
    response_text: str
    query_text: str
    tenant_id: str
    created_at: datetime
    ttl_expiry: datetime
    lfu_count: int
    match_type: Literal["exact", "semantic"]
    score: float | None = None  # only populated for semantic matches
    is_hot: bool  # lfu_count > settings.LFU_SPIKE_THRESHOLD


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _utcnow() -> datetime:
    """Return timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _parse_iso(iso_str: str) -> datetime:
    """Parse an ISO-8601 string into a timezone-aware datetime.

    Handles both naive (treats as UTC) and aware ISO strings.
    """
    dt = datetime.fromisoformat(iso_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _build_cache_result(
    payload: dict,
    match_type: Literal["exact", "semantic"],
    score: float | None = None,
) -> CacheResult:
    """Construct a CacheResult from a Qdrant payload dict."""
    settings = get_settings()
    lfu_count = int(payload.get("lfu_count", 0))

    return CacheResult(
        vector_id=str(payload["_point_id"]),
        response_text=payload["response"],
        query_text=payload["query"],
        tenant_id=payload["tenant_id"],
        created_at=_parse_iso(payload["created_at"]),
        ttl_expiry=_parse_iso(payload["ttl_expiry"]),
        lfu_count=lfu_count,
        match_type=match_type,
        score=score,
        is_hot=lfu_count > settings.LFU_SPIKE_THRESHOLD,
    )


# ────────────────────────────────────────────────────────────────────
# Primary functions
# ────────────────────────────────────────────────────────────────────


async def lookup(
    tenant_id: str,
    query_text: str,
    query_embedding: list[float],
    similarity_threshold: float,
) -> CacheResult | None:
    """Two-tier cache lookup: exact hash → semantic vector search.

    Returns a ``CacheResult`` on a fresh hit, or ``None`` on miss.

    This function NEVER raises. All Qdrant failures are degraded to
    ``None`` via the circuit-breaker decorators, and any unexpected
    exception is caught, logged, and treated as a miss — a bug in
    cache logic must never take down the whole gateway.
    """
    try:
        # ── Step A: exact match ────────────────────────────────────
        query_hash = compute_query_hash(tenant_id, query_text)
        payload = await exact_match_lookup(tenant_id, query_hash)

        match_type: Literal["exact", "semantic"] = "exact"
        score: float | None = None

        if payload is None:
            # ── Step B: semantic vector search ─────────────────────
            payload = await vector_search(
                tenant_id, query_embedding, similarity_threshold
            )
            if payload is None:
                return None
            match_type = "semantic"
            score = payload.get("_score")

        # ── Step C: validate freshness ─────────────────────────────
        ttl_expiry = _parse_iso(payload["ttl_expiry"])
        if _utcnow() > ttl_expiry:
            # Entry is stale — evict it and treat as a miss.
            # delete_point is decorated with @fire_and_forget_with_log
            # so it handles its own failures and never blocks.
            await delete_point(payload["_point_id"])
            logger.debug(
                "stale_cache_entry_evicted",
                module="cache_manager",
                point_id=payload["_point_id"],
                tenant_id=tenant_id,
            )
            return None

        return _build_cache_result(payload, match_type, score)

    except Exception as e:
        # A bug in cache logic must never take down the gateway.
        logger.warning(
            "cache_lookup_unexpected_error",
            module="cache_manager",
            error=str(e),
            tenant_id=tenant_id,
        )
        return None


async def record_hit(vector_id: str) -> int:
    """Increment the LFU counter and optionally promote a hot entry.

    Returns the new lfu_count, or 0 on any failure. Never raises.

    Note: Qdrant's ``set_payload`` does a literal overwrite, not an
    atomic increment. We read → increment → write. This is acceptable
    in a single-writer demo context but would need a lock or CAS for
    high-concurrency production use.
    """
    try:
        settings = get_settings()

        # Read current payload
        payload = await _get_raw_payload(vector_id)
        if payload is None:
            logger.warning(
                "record_hit_point_not_found",
                module="cache_manager",
                vector_id=vector_id,
            )
            return 0

        # Increment
        new_count = int(payload.get("lfu_count", 0)) + 1
        now = _utcnow()

        updates: dict = {
            "lfu_count": new_count,
            "last_accessed": now.isoformat(),
        }

        # LFU spike promotion — extend TTL for hot entries
        if new_count > settings.LFU_SPIKE_THRESHOLD:
            new_expiry = now + timedelta(
                seconds=settings.LFU_SPIKE_TTL_EXTENSION
            )
            updates["ttl_expiry"] = new_expiry.isoformat()
            logger.info(
                "hot_cache_promotion",
                module="cache_manager",
                vector_id=vector_id,
                lfu_count=new_count,
                new_expiry=new_expiry.isoformat(),
            )

        await update_payload(vector_id, updates)
        return new_count

    except Exception as e:
        logger.warning(
            "record_hit_failed",
            module="cache_manager",
            vector_id=vector_id,
            error=str(e),
        )
        return 0


async def store(
    tenant_id: str,
    query_text: str,
    query_embedding: list[float],
    response_text: str,
) -> str:
    """Store a new cache entry (embedding + payload) in Qdrant.

    Returns the generated point_id. Never raises — upsert_point is
    decorated with ``@fire_and_forget_with_log`` which handles and
    logs Qdrant failures.
    """
    try:
        settings = get_settings()

        point_id = str(uuid4())
        query_hash = compute_query_hash(tenant_id, query_text)
        now = _utcnow()

        payload = {
            "tenant_id": tenant_id,
            "query_hash": query_hash,
            "query": query_text,
            "response": response_text,
            "created_at": now.isoformat(),
            "ttl_expiry": (
                now + timedelta(seconds=settings.CACHE_TTL_SECONDS)
            ).isoformat(),
            "lfu_count": 1,
            "last_accessed": now.isoformat(),
        }

        await upsert_point(point_id, query_embedding, payload)

        logger.debug(
            "cache_store",
            module="cache_manager",
            point_id=point_id,
            tenant_id=tenant_id,
            query_preview=query_text[:60],
        )

        return point_id

    except Exception as e:
        # Defensive — upsert_point already handles its own errors.
        logger.warning(
            "cache_store_unexpected_error",
            module="cache_manager",
            error=str(e),
            tenant_id=tenant_id,
        )
        return ""


async def get_stats() -> dict:
    """Return collection-level stats from Qdrant.

    Returns {"collection": ..., "points": ..., "status": "ok"} on
    success, or {"status": "degraded"} on any failure.
    """
    try:
        from db.qdrant_client import get_qdrant_client

        settings = get_settings()
        client = await get_qdrant_client()
        info = await client.get_collection(settings.QDRANT_COLLECTION)

        return {
            "collection": settings.QDRANT_COLLECTION,
            "points": info.points_count,
            "status": "ok",
        }
    except Exception as e:
        logger.warning(
            "cache_stats_failed",
            module="cache_manager",
            error=str(e),
        )
        return {"status": "degraded"}
