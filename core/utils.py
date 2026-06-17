"""Shared utility helpers for the gateway.

These functions are the SINGLE source of truth for text normalization and
query hashing. Both the embedding path (core.embeddings) and the cache
lookup path (db.qdrant_client) rely on them. If normalization logic is
duplicated or diverges, the exact-hash check and the vector check will
silently disagree on what counts as "the same query."
"""

import hashlib
import re


def normalize_text(text: str) -> str:
    """Lowercase, strip, and collapse internal whitespace to single spaces.

    >>> normalize_text("  Hello   World  ")
    'hello world'
    """
    return re.sub(r"\s+", " ", text.lower().strip())


def compute_query_hash(tenant_id: str, query_text: str) -> str:
    """Return a deterministic SHA-256 hex digest for (tenant, query).

    The query text is normalized first so that cosmetic whitespace
    differences don't produce distinct hashes.

    >>> compute_query_hash("t1", "  Hello   World  ") == compute_query_hash("t1", "hello world")
    True
    """
    normalized = normalize_text(query_text)
    return hashlib.sha256(f"{tenant_id}:{normalized}".encode()).hexdigest()
