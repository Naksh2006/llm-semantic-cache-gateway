"""Database package — Qdrant client wrapper and cache manager.

This is the ONLY persistence layer in the v2 architecture.  There is no
Redis, no SQLite, no secondary store.  Qdrant handles both vector search
and payload-based metadata (TTL, hit count, timestamps).
"""
