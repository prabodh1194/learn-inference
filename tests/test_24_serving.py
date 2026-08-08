"""Lecture 24 -- serving.

API-shape and streaming semantics. No GPU; the engine is faked so these test
the SERVER, not the model.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from serve.api import create_app

    class FakeEngine:
        """Deterministic stand-in so server tests don't need a GPU."""

        def generate(self, prompt, **kw):
            for tok in ["Because", " decode", " is", " memory", "-bound"]:
                yield tok

    return TestClient(create_app(engine=FakeEngine()))


class TestOpenAICompat:
    def test_models_endpoint(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200
        assert "data" in r.json()

    def test_non_streaming_response_shape(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "why?"}],
        })
        assert r.status_code == 200
        body = r.json()
        # The shape every OpenAI client expects.
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["choices"][0]["message"]["content"]
        assert "usage" in body

    def test_streaming_emits_sse_chunks_then_done(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "why?"}],
            "stream": True,
        })
        assert r.status_code == 200
        lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
        assert len(lines) >= 2, "should emit multiple chunks, not one blob"
        assert lines[-1] == "data: [DONE]"

    def test_streaming_uses_delta_not_message(self, client):
        import json

        r = client.post("/v1/chat/completions", json={
            "model": "qwen3-0.6b",
            "messages": [{"role": "user", "content": "why?"}],
            "stream": True,
        })
        first = next(l for l in r.text.splitlines()
                     if l.startswith("data: ") and "[DONE]" not in l)
        assert "delta" in json.loads(first[6:])["choices"][0]


class TestValidation:
    def test_rejects_missing_messages(self, client):
        r = client.post("/v1/chat/completions", json={"model": "qwen3-0.6b"})
        assert r.status_code in (400, 422)
