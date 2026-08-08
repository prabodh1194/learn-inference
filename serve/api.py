"""M4.1 (Lecture 24) -- OpenAI-compatible HTTP server.

Run the engine loop in a SEPARATE PROCESS. HTTP handling and tokenization on
the engine's event loop steal time from the scheduler -- which is exactly why
vLLM splits them.
"""

from __future__ import annotations


def create_app(engine):
    """M4.1. FastAPI app exposing /v1/chat/completions and /v1/models.

    Streaming uses SSE with `delta` chunks (not `message`) and terminates with
    `data: [DONE]`. Buffer the whole response and you have thrown away every
    latency optimization in this book.

    Handle disconnection: generating tokens nobody will read burns real GPU.
    """
    raise NotImplementedError("M4.1")
