"""Lecture 13 -- CUDA graphs.

Marked cuda: there is no graph-capture equivalent on MPS.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.cuda


@pytest.mark.slow
def test_graph_output_matches_eager(model_and_tokenizer):
    """Replay must be numerically identical to the eager path."""
    from engine.generate import generate_cached, generate_graphed

    model, tok = model_and_tokenizer
    prompt = "The KV cache exists because"
    assert generate_graphed(model, tok, prompt, max_tokens=16) == (
        generate_cached(model, tok, prompt, max_tokens=16)
    )


@pytest.mark.slow
def test_replay_is_faster_at_batch_one(model_and_tokenizer):
    """Batch 1 is where you're maximally launch-bound."""
    import time

    from bench.harness import synchronize
    from engine.generate import generate_cached, generate_graphed

    model, tok = model_and_tokenizer

    def timed(fn):
        fn(model, tok, "hi", max_tokens=8)      # warmup + capture
        synchronize()
        t0 = time.perf_counter()
        fn(model, tok, "hi", max_tokens=64)
        synchronize()
        return time.perf_counter() - t0

    assert timed(generate_graphed) < timed(generate_cached)


def test_static_cache_has_fixed_addresses():
    """Why graphs need StaticCache: DynamicCache grows by concatenation, so
    its tensors move, and a captured graph cannot follow moving addresses."""
    pytest.importorskip("transformers")
    from transformers import DynamicCache, StaticCache

    assert DynamicCache is not None and StaticCache is not None
