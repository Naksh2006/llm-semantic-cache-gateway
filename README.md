# LLM Semantic Cache Gateway

A caching layer that sits between your app and an LLM API and actually understands what's being asked, instead of matching exact strings.

Ask "What's the capital of France?" and then, five minutes later, "Tell me France's capital city" — a normal cache (Redis, Memcached) treats those as two unrelated keys and calls the LLM twice. This gateway recognizes they mean the same thing and returns the second one in about 12ms.

I built this mainly to get hands-on with async Python, fault-tolerant service design, and vector search, not just to wrap an API behind a cache. The architecture choices below reflect that — some of them trade a bit of raw simplicity for the chance to actually exercise interesting failure modes.

## The problem this solves

In any app with more than a handful of users, a lot of the questions hitting your LLM are semantically identical, just worded differently. Multiply a $0.005 API call across thousands of near-duplicate questions and the cost adds up fast — on top of the 1-2 second latency every single one of them pays.

Exact-match caching doesn't help here because it has no concept of meaning. This gateway does two lookups instead of one: a fast exact-hash check for literal repeats, and a vector similarity search underneath that for everything that's worded differently but means the same thing.

## How it works

1. The query gets normalized and hashed (`tenant_id + text`) and checked against Qdrant for an exact match first — this is the cheap path for genuinely repeated questions.
2. If that misses, the query is embedded locally (no external API call) and a cosine-similarity search runs against that tenant's existing vectors. A score above the threshold counts as a hit.
3. A hit returns immediately. Anything else streams from the LLM as normal, and the full response gets written to the cache in the background once the stream finishes — the client never waits on the cache write.
4. If Qdrant is down for any reason, the lookup just fails quietly and the request goes straight to the LLM. Caching is treated as an optimization, not a dependency — losing it should never take the gateway down with it.

Everything — vectors, the exact-match hash, TTL, hit counters — lives in Qdrant. No Redis, no second datastore to keep in sync.

## Project layout

```
llm-gateway/
├── main.py                  # startup, health/metrics endpoints
├── core/
│   ├── config.py             # settings, loaded from .env
│   ├── utils.py               # text normalization + hashing
│   ├── embeddings.py          # local FastEmbed wrapper
│   ├── exceptions.py
│   └── circuit_breaker.py     # fallback decorators for Qdrant calls
├── db/
│   ├── qdrant_client.py       # low-level Qdrant CRUD
│   └── cache_manager.py       # TTL/LFU logic, two-tier lookup
├── api/
│   ├── router.py               # POST /v1/chat
│   └── streaming.py            # SSE streaming + the buffering fix
└── benchmarks/
    └── sim_cli.py              # latency profiling, fault-injection tests
```

## Running it locally

You'll need Python 3.11+, Docker (for Qdrant), and an API key for whatever LLM provider you're using with LiteLLM.

```bash
git clone https://github.com/Naksh2006/llm-semantic-cache-gateway.git
cd llm-semantic-cache-gateway

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

docker run -d --name qdrant -p 6333:6333 qdrant/qdrant:latest

cp .env.example .env
# edit .env and set your LLM_API_KEY

uvicorn main:app --reload --port 8000
```

## Trying it out

First request — this is a miss, so it streams from the LLM and caches the result afterward:

```bash
curl http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: my-company" \
  -d '{"messages": [{"role": "user", "content": "What is machine learning?"}]}'
```

Same question again — this one comes back from the cache:

```bash
curl http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "x-tenant-id: my-company" \
  -d '{"messages": [{"role": "user", "content": "What is machine learning?"}]}'
```

You can tell which path you hit from the `X-Cache` response header.

## Tests and benchmarks

```bash
python benchmarks/sim_cli.py exact-hit -n 100        # latency profile, mocked
python benchmarks/sim_cli.py semantic-hit -n 100
python benchmarks/sim_cli.py db-down -n 10            # circuit breaker under failure
python benchmarks/sim_cli.py tenant-leak-check        # proves tenant isolation actually holds
python benchmarks/sim_cli.py cold-start --prompt "Explain gravity"   # real end-to-end, needs the gateway running
```

Numbers from my own runs:

| Metric | Value |
|---|---|
| Cache hit latency | ~10-12ms |
| Cache miss latency (LLM TTFT) | ~1,200-1,500ms |
| Speedup | ~120x |
| Embedding inference (warm) | ~5ms |
| Vector dimensions | 384 |

## Tech stack

FastAPI · Qdrant · FastEmbed (ONNX, local embeddings, no external embedding API) · LiteLLM · pydantic-settings · structlog

## What this doesn't do yet

Being upfront about the gaps rather than burying them:

- There's no authentication on `tenant_id` — it's just a header right now, so this isn't safe to expose publicly as-is. Tenant isolation works correctly *between* tenants, but nothing stops someone from claiming to be a tenant they aren't.
- Qdrant runs as a single local container. Fine for development, not something you'd run in production without replication.
- The hit counter (LFU) does a read-then-write under the hood, so under genuinely concurrent hits to the same entry you could lose an increment here and there. Acceptable for a popularity heuristic, not something I'd want for anything that needs to be exact.
- Caching is based on the last message only, so multi-turn conversations don't get this benefit — "tell me more" gets cached as if it were a standalone question.

## In progress

A few things actively being built on top of this:

- A cost-aware routing layer that scores each query's complexity and picks between a fast/cheap model and a stronger one accordingly, working against whichever LLM provider key you drop into `.env` rather than being hardcoded to one provider.
- A small dashboard (Next.js) for watching cache decisions, latency, and cost as they happen, plus a real analytics view backed by logged request history rather than made-up numbers.
- A local MCP server so the cache can be queried and inspected directly from Claude Desktop or similar tools.

None of these are done yet — this README will get updated as they land.
