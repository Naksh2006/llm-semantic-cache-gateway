"""Server-Sent Events (SSE) streaming helpers.

Responsible for:
  • Wrapping litellm's async streaming generator into SSE frames
    compatible with the OpenAI streaming protocol.
  • Accumulating the full response text as chunks arrive so the
    completed answer can be cached in Qdrant after the stream ends.
  • Graceful handling of upstream errors — yields an SSE error frame
    before raising ``StreamingError`` so the client sees what happened.
  • Emitting a final ``data: [DONE]\n\n`` sentinel per the OpenAI
    streaming protocol.
  • Tracking Time-To-First-Token (TTFT) for observability.
"""

import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from time import perf_counter_ns

import litellm
import structlog

from core.config import get_settings
from core.exceptions import StreamingError

logger = structlog.get_logger(__name__)


async def stream_llm_response(
    messages: list[dict], model: str
) -> AsyncGenerator[str, None]:
    """Stream an LLM completion as SSE frames.

    Calls ``litellm.acompletion`` with ``stream=True`` and yields each
    token as an SSE ``data:`` frame. Tracks TTFT (time-to-first-token)
    and logs it at INFO.

    On any exception: yields an SSE error frame so the client is notified,
    then raises ``StreamingError`` for the caller to handle.
    """
    settings = get_settings()

    try:
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            stream=True,
            api_key=settings.LLM_API_KEY,
        )

        first_token = True
        t_start = perf_counter_ns()

        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta is not None:
                if first_token:
                    ttft_ms = (perf_counter_ns() - t_start) / 1_000_000
                    logger.info(
                        "ttft_ms",
                        ttft=round(ttft_ms, 2),
                        model=model,
                    )
                    first_token = False

                yield f"data: {json.dumps({'content': delta})}\n\n"

        # async for exits naturally when the stream is exhausted —
        # no StopIteration in async generators, just end of iteration.
        yield "data: [DONE]\n\n"

    except StreamingError:
        raise
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
        raise StreamingError(
            "LLM streaming failed", original_error=e
        )


# ────────────────────────────────────────────────────────────────────
# BUFFERING-TRAP SOLUTION
# ────────────────────────────────────────────────────────────────────
#
# Why this function exists:
#
# A naive approach would either (a) buffer the entire response before
# sending it (defeating the purpose of streaming) or (b) stream to the
# client but have nothing left to cache (the generator is consumed).
#
# stream_and_capture solves this by doing both simultaneously:
#   • The `yield` happens BEFORE the token is appended to
#     collected_tokens, so the client never waits on bookkeeping —
#     accumulation is a side-effect of iterating, not a blocking step.
#   • `on_complete` is called ONLY AFTER every token has reached the
#     client, meaning the full response text is available exactly once
#     and exactly when needed.
#   • `on_complete` is the hook the router uses to fire the Qdrant
#     write. This function has ZERO knowledge of caching, keeping it
#     reusable for any post-stream callback.
# ────────────────────────────────────────────────────────────────────


async def stream_and_capture(
    messages: list[dict],
    model: str,
    on_complete: Callable[[str], Awaitable[None]],
) -> AsyncGenerator[str, None]:
    """Stream SSE frames to the client while silently capturing the full text.

    Iterates over ``stream_llm_response``, yielding every SSE frame to
    the client immediately.  After the generator exhausts (all tokens
    delivered), joins the captured tokens and calls ``on_complete`` with
    the full response text.

    Buffer-parsing errors (malformed JSON, missing keys) are silently
    swallowed — they must never crash the live stream.
    """
    collected_tokens: list[str] = []

    async for sse_frame in stream_llm_response(messages, model):
        # Client receives the frame immediately via yield.
        yield sse_frame

        # Side-effect: capture the token for post-stream caching.
        if (
            sse_frame.startswith("data: ")
            and "[DONE]" not in sse_frame
            and '"error"' not in sse_frame
        ):
            try:
                payload = json.loads(sse_frame[6:].strip())
                collected_tokens.append(payload["content"])
            except (json.JSONDecodeError, KeyError):
                pass  # never let buffer parsing crash the live stream

    # By this point the async generator has exhausted — every token has
    # already reached the client. ONLY NOW do we have the complete text.
    full_response = "".join(collected_tokens)
    await on_complete(full_response)
