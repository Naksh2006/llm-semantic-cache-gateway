"""High-level semantic cache operations (lookup, store, evict).

Orchestrates **embeddings → Qdrant search → LLM call** in a single
coherent workflow:

  1. Embed the incoming query locally (``core.embeddings``).
  2. Search Qdrant for a vector within the similarity threshold.
  3. On HIT  → return cached response, bump hit counter, extend TTL if
     LFU spike is detected.
  4. On MISS → forward to the LLM via litellm, stream the response back,
     then upsert the (vector, response, metadata) into Qdrant.

Eviction:
  • TTL-based expiry is enforced via Qdrant payload filters at query
    time (no background reaper needed for v2 MVP).
  • LFU spike detection: if ``hit_count`` exceeds
    ``LFU_SPIKE_THRESHOLD`` within one TTL window, the TTL is extended
    by ``LFU_SPIKE_TTL_EXTENSION`` seconds.

All Qdrant interactions go through ``db.qdrant_client`` (never the SDK
directly), keeping the circuit breaker in the loop.
"""
