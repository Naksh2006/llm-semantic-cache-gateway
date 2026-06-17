"""Server-Sent Events (SSE) streaming helpers.

Responsible for:
  • Wrapping litellm's async streaming generator into a FastAPI
    ``StreamingResponse`` that emits OpenAI-compatible SSE chunks
    (``data: {json}\\n\\n``).
  • Accumulating the full response text as chunks arrive so the
    completed answer can be cached in Qdrant after the stream ends.
  • Graceful handling of client disconnects (broken pipe) — raises
    ``StreamingError`` so the caller can log and clean up.
  • Emitting a final ``data: [DONE]\\n\\n`` sentinel per the OpenAI
    streaming protocol.
"""
