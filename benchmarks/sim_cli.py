"""Benchmark CLI for the LLM Semantic Cache Gateway.

Usage:
    python benchmarks/sim_cli.py exact-hit       [--iterations N]
    python benchmarks/sim_cli.py semantic-hit    [--iterations N]
    python benchmarks/sim_cli.py db-down         [--iterations N]
    python benchmarks/sim_cli.py tenant-leak-check
    python benchmarks/sim_cli.py cold-start      [--prompt "text"]

All async code runs via asyncio.run(). Uses only packages already in
requirements.txt.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from time import perf_counter_ns
from unittest.mock import AsyncMock, patch
from uuid import uuid4

# Ensure project root is importable
sys.path.insert(0, ".")


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _ns_to_ms(ns: int) -> float:
    return ns / 1_000_000


def _percentile(sorted_vals: list[float], p: float) -> float:
    """Compute the p-th percentile from a sorted list."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def _print_latency_table(
    label: str, latencies_ms: list[float]
) -> None:
    """Print an ASCII latency distribution table."""
    s = sorted(latencies_ms)
    print()
    print(f"  ┌─────────────────────────────────────────────┐")
    print(f"  │  {label:<42s}│")
    print(f"  ├──────────────┬──────────────────────────────┤")
    print(f"  │  Metric      │  Value                       │")
    print(f"  ├──────────────┼──────────────────────────────┤")
    print(f"  │  Iterations  │  {len(s):<29d}│")
    print(f"  │  Min         │  {s[0]:>8.2f} ms                  │")
    print(f"  │  p50         │  {_percentile(s, 50):>8.2f} ms                  │")
    print(f"  │  p95         │  {_percentile(s, 95):>8.2f} ms                  │")
    print(f"  │  p99         │  {_percentile(s, 99):>8.2f} ms                  │")
    print(f"  │  Max         │  {s[-1]:>8.2f} ms                  │")
    print(f"  └──────────────┴──────────────────────────────┘")
    print()


def _make_fake_payload(
    tenant_id: str = "bench-tenant",
    score: float | None = None,
) -> dict:
    """Build a realistic Qdrant payload dict for mocking."""
    now = datetime.now(timezone.utc)
    point_id = str(uuid4())
    payload = {
        "_point_id": point_id,
        "tenant_id": tenant_id,
        "query_hash": "fakehash123",
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "created_at": now.isoformat(),
        "ttl_expiry": (now + timedelta(hours=1)).isoformat(),
        "lfu_count": 5,
        "last_accessed": now.isoformat(),
    }
    if score is not None:
        payload["_score"] = score
    return payload


# ────────────────────────────────────────────────────────────────────
# Subcommands
# ────────────────────────────────────────────────────────────────────


async def cmd_exact_hit(iterations: int) -> None:
    """Benchmark exact-match cache hits (mocked, no network)."""
    from core.config import get_settings
    from db.cache_manager import lookup

    settings = get_settings()
    fake_embedding = [0.1] * settings.EMBEDDING_DIM
    fake_payload = _make_fake_payload()

    latencies: list[float] = []

    with (
        patch(
            "db.cache_manager.exact_match_lookup",
            new_callable=AsyncMock,
            return_value=fake_payload,
        ),
        patch(
            "core.embeddings.get_embedding",
            new_callable=AsyncMock,
            return_value=fake_embedding,
        ),
    ):
        for i in range(iterations):
            t0 = perf_counter_ns()
            result = await lookup(
                "bench-tenant",
                "What is the capital of France?",
                fake_embedding,
                settings.DEFAULT_SIMILARITY_THRESHOLD,
            )
            t1 = perf_counter_ns()
            latencies.append(_ns_to_ms(t1 - t0))

            if result is None:
                print(f"  ❌ Iteration {i+1}: got None (expected HIT)")
            elif result.match_type != "exact":
                print(
                    f"  ❌ Iteration {i+1}: match_type={result.match_type} (expected exact)"
                )

    _print_latency_table("EXACT-HIT Latency Distribution", latencies)


async def cmd_semantic_hit(iterations: int) -> None:
    """Benchmark semantic cache hits (mocked, no network)."""
    from core.config import get_settings
    from db.cache_manager import lookup

    settings = get_settings()
    fake_embedding = [0.1] * settings.EMBEDDING_DIM
    fake_payload = _make_fake_payload(score=0.95)

    latencies: list[float] = []

    with (
        patch(
            "db.cache_manager.exact_match_lookup",
            new_callable=AsyncMock,
            return_value=None,  # exact miss → forces semantic path
        ),
        patch(
            "db.cache_manager.vector_search",
            new_callable=AsyncMock,
            return_value=fake_payload,
        ),
        patch(
            "core.embeddings.get_embedding",
            new_callable=AsyncMock,
            return_value=fake_embedding,
        ),
    ):
        for i in range(iterations):
            t0 = perf_counter_ns()
            result = await lookup(
                "bench-tenant",
                "Tell me the capital city of France",
                fake_embedding,
                settings.DEFAULT_SIMILARITY_THRESHOLD,
            )
            t1 = perf_counter_ns()
            latencies.append(_ns_to_ms(t1 - t0))

            if result is None:
                print(f"  ❌ Iteration {i+1}: got None (expected HIT)")
            elif result.match_type != "semantic":
                print(
                    f"  ❌ Iteration {i+1}: match_type={result.match_type} (expected semantic)"
                )

    _print_latency_table("SEMANTIC-HIT Latency Distribution", latencies)


async def cmd_db_down(iterations: int) -> None:
    """Verify circuit breaker degrades gracefully when Qdrant is down."""
    from core.exceptions import VectorDBError

    passed = 0
    failed = 0

    # Mock both read paths to raise VectorDBError
    # Mock the LLM stream to emit 5 fake tokens
    async def _fake_stream(*args, **kwargs):
        for i in range(5):
            await asyncio.sleep(0.01)
            yield f"data: {json.dumps({'content': f'token{i}'})}\n\n"
        yield "data: [DONE]\n\n"

    with (
        patch(
            "db.cache_manager.exact_match_lookup",
            new_callable=AsyncMock,
            side_effect=VectorDBError("Qdrant is down"),
        ),
        patch(
            "db.cache_manager.vector_search",
            new_callable=AsyncMock,
            side_effect=VectorDBError("Qdrant is down"),
        ),
        patch(
            "api.streaming.stream_llm_response",
            side_effect=lambda *a, **kw: _fake_stream(*a, **kw),
        ),
        patch(
            "db.cache_manager.upsert_point",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        from core.config import get_settings
        from db.cache_manager import lookup

        settings = get_settings()
        fake_embedding = [0.1] * settings.EMBEDDING_DIM

        for i in range(iterations):
            try:
                result = await lookup(
                    "bench-tenant",
                    f"test query {i}",
                    fake_embedding,
                    settings.DEFAULT_SIMILARITY_THRESHOLD,
                )
                # With Qdrant down, both lookups should degrade to None
                if result is None:
                    passed += 1
                    print(f"  ✅ Iteration {i+1}: MISS (circuit breaker worked)")
                else:
                    failed += 1
                    print(f"  ❌ Iteration {i+1}: unexpected HIT")
            except Exception as e:
                failed += 1
                print(f"  ❌ Iteration {i+1}: RAISED {type(e).__name__}: {e}")

    print()
    total = passed + failed
    print(f"  {passed}/{total} requests succeeded despite Qdrant being down.")
    if failed == 0:
        print("  ✅ PASS — circuit breaker degrades gracefully")
    else:
        print("  ❌ FAIL — some requests raised instead of degrading")


async def cmd_tenant_leak_check() -> None:
    """Verify tenant isolation — tenant-b must never see tenant-a's cache."""
    from core.config import get_settings
    from core.embeddings import get_embedding
    from db.cache_manager import lookup, store
    from db.qdrant_client import delete_point

    settings = get_settings()

    query = "summarize my emails"
    response = "Here is a summary of your emails..."

    print("  Generating embedding...")
    emb = await get_embedding(query)

    # Store under tenant-a
    print("  Storing cache entry for tenant-a...")
    point_id = await store("tenant-a", query, emb, response)
    await asyncio.sleep(0.5)

    # Verify tenant-a can see it
    hit_a = await lookup(
        "tenant-a", query, emb, settings.DEFAULT_SIMILARITY_THRESHOLD
    )
    if hit_a is None:
        print("  ❌ FAIL — tenant-a cannot see its own entry!")
        return

    print(f"  tenant-a lookup: ✅ HIT (match_type={hit_a.match_type})")

    # Attempt lookup as tenant-b (exact match path)
    hit_b_exact = await lookup(
        "tenant-b", query, emb, settings.DEFAULT_SIMILARITY_THRESHOLD
    )

    # Attempt lookup as tenant-b (semantic path with low threshold)
    hit_b_semantic = await lookup("tenant-b", query, emb, 0.5)

    # Cleanup
    if point_id:
        await delete_point(point_id)
        await asyncio.sleep(0.3)

    # Report
    print(f"  tenant-b exact lookup:    {hit_b_exact}")
    print(f"  tenant-b semantic lookup: {hit_b_semantic}")
    print()

    if hit_b_exact is None and hit_b_semantic is None:
        print("  ✅ PASS — tenant isolation holds on both paths")
    else:
        print("  ❌ FAIL — TENANT LEAK DETECTED!")
        if hit_b_exact is not None:
            print("    Leaked via exact-match path")
        if hit_b_semantic is not None:
            print("    Leaked via semantic-search path")


async def cmd_cold_start(prompt: str) -> None:
    """Send two real requests to measure cache speedup (requires running gateway).

    First request is a guaranteed MISS, second should be an exact HIT.
    """
    try:
        import httpx
    except ImportError:
        print("  ❌ httpx is required for cold-start test")
        print("     pip install httpx")
        return

    base_url = "http://localhost:8000/v1/chat"
    headers = {"x-tenant-id": "bench-cold-start"}
    body = {"messages": [{"role": "user", "content": prompt}]}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # ── Request 1: guaranteed MISS ─────────────────────────────
        print(f"  Sending request 1 (MISS expected)...")
        t1_start = perf_counter_ns()
        resp1 = await client.post(
            base_url, json=body, headers=headers
        )
        t1_end = perf_counter_ns()
        miss_ms = _ns_to_ms(t1_end - t1_start)

        x_cache_1 = resp1.headers.get("x-cache", "?")
        print(f"    Status: {resp1.status_code}, X-Cache: {x_cache_1}")
        print(f"    Latency: {miss_ms:.1f}ms")

        if resp1.status_code != 200 and "text/event-stream" not in resp1.headers.get("content-type", ""):
            # For streaming responses, status is always 200
            pass

        # Small delay for the background cache write to complete
        await asyncio.sleep(2)

        # ── Request 2: should be exact HIT ─────────────────────────
        print(f"  Sending request 2 (HIT expected)...")
        t2_start = perf_counter_ns()
        resp2 = await client.post(
            base_url, json=body, headers=headers
        )
        t2_end = perf_counter_ns()
        hit_ms = _ns_to_ms(t2_end - t2_start)

        x_cache_2 = resp2.headers.get("x-cache", "?")
        print(f"    Status: {resp2.status_code}, X-Cache: {x_cache_2}")
        print(f"    Latency: {hit_ms:.1f}ms")

        # ── Results ────────────────────────────────────────────────
        print()
        print(f"  ┌─────────────────────────────────────────┐")
        print(f"  │  Cold-Start Benchmark Results           │")
        print(f"  ├──────────────┬──────────────────────────┤")
        print(f"  │  MISS (LLM)  │  {miss_ms:>8.1f} ms              │")
        print(f"  │  HIT (cache) │  {hit_ms:>8.1f} ms              │")
        if miss_ms > 0:
            speedup = miss_ms / max(hit_ms, 0.01)
            saved = miss_ms - hit_ms
            print(f"  │  Speedup     │  {speedup:>8.1f}x               │")
            print(f"  │  Time saved  │  {saved:>8.1f} ms              │")
        print(f"  └──────────────┴──────────────────────────┘")


# ────────────────────────────────────────────────────────────────────
# CLI entry-point
# ────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sim_cli",
        description="Benchmark CLI for the LLM Semantic Cache Gateway.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # exact-hit
    p1 = sub.add_parser(
        "exact-hit", help="Benchmark exact-match cache hits (mocked)"
    )
    p1.add_argument(
        "--iterations", "-n", type=int, default=100
    )

    # semantic-hit
    p2 = sub.add_parser(
        "semantic-hit",
        help="Benchmark semantic cache hits (mocked)",
    )
    p2.add_argument(
        "--iterations", "-n", type=int, default=100
    )

    # db-down
    p3 = sub.add_parser(
        "db-down",
        help="Verify circuit breaker when Qdrant is down",
    )
    p3.add_argument(
        "--iterations", "-n", type=int, default=10
    )

    # tenant-leak-check
    sub.add_parser(
        "tenant-leak-check",
        help="Verify tenant isolation (requires running Qdrant)",
    )

    # cold-start
    p5 = sub.add_parser(
        "cold-start",
        help="Measure cache speedup with real requests (requires running gateway)",
    )
    p5.add_argument(
        "--prompt",
        type=str,
        default="What is the meaning of life?",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    print()
    print(f"═══ Gateway Benchmark: {args.command} ═══")
    print()

    if args.command == "exact-hit":
        asyncio.run(cmd_exact_hit(args.iterations))
    elif args.command == "semantic-hit":
        asyncio.run(cmd_semantic_hit(args.iterations))
    elif args.command == "db-down":
        asyncio.run(cmd_db_down(args.iterations))
    elif args.command == "tenant-leak-check":
        asyncio.run(cmd_tenant_leak_check())
    elif args.command == "cold-start":
        asyncio.run(cmd_cold_start(args.prompt))


if __name__ == "__main__":
    main()
