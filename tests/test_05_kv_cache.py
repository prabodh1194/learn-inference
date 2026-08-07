"""Lecture 05 -- the KV cache.

The gate: caching is an OPTIMIZATION, so it must change speed and nothing else.
Identical tokens out, or it's a bug.

This is the first time you have two implementations of the same thing, and it
sets the pattern for the rest of the course: every faster path is checked
against the slower one that already passed.
"""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_cached_matches_naive(model_and_tokenizer):
    """Same prompt, same greedy path, same tokens. No excuses."""
    from engine.generate import generate_cached, generate_naive

    model, tok = model_and_tokenizer
    prompt = "The KV cache exists because"

    naive = generate_naive(model, tok, prompt, max_tokens=24)
    cached = generate_cached(model, tok, prompt, max_tokens=24)

    assert cached == naive, (
        "Cached output diverged from naive.\n"
        f"  naive:  {naive!r}\n"
        f"  cached: {cached!r}\n"
        "Most likely: you passed the FULL sequence alongside past_key_values. "
        "During decode exactly one token goes in -- the cache carries the rest."
    )


@pytest.mark.slow
def test_cached_matches_huggingface(model_and_tokenizer, reference_greedy):
    from engine.generate import generate_cached

    model, tok = model_and_tokenizer
    got = generate_cached(model, tok, reference_greedy["prompt"], max_tokens=24)
    assert got == reference_greedy["text"]


@pytest.mark.slow
def test_on_token_count_is_exact(model_and_tokenizer):
    """Off-by-one here silently corrupts every tok/s number you report."""
    from engine.generate import generate_cached

    model, tok = model_and_tokenizer
    calls = []
    generate_cached(model, tok, "Hello", max_tokens=6, on_token=calls.append)
    assert len(calls) == 6


@pytest.mark.slow
def test_cache_is_actually_faster_at_length(model_and_tokenizer):
    """The point of the lecture, asserted.

    Deliberately measured at 128 tokens, not 16: at short lengths fixed
    overheads dominate and the win can vanish. The threshold is loose (1.5x)
    because this must pass on a laptop under arbitrary load -- it's a smoke
    test that the mechanism works, not a benchmark. Your real numbers come
    from bench/.
    """
    import time

    from bench.harness import synchronize
    from engine.generate import generate_cached, generate_naive

    model, tok = model_and_tokenizer
    prompt = "Once upon a time"

    def timed(fn) -> float:
        fn(model, tok, prompt, max_tokens=8)  # warmup
        synchronize()
        t0 = time.perf_counter()
        fn(model, tok, prompt, max_tokens=128)
        synchronize()
        return time.perf_counter() - t0

    naive_s = timed(generate_naive)
    cached_s = timed(generate_cached)

    assert cached_s < naive_s / 1.5, (
        f"Expected a clear speedup at 128 tokens, got "
        f"naive={naive_s:.2f}s cached={cached_s:.2f}s. "
        "If cached is not faster, check that decode passes ONE token per step."
    )


class TestKVCacheSizing:
    """Pure arithmetic -- runs without the model."""

    def test_bytes_per_token_formula(self):
        """2 (K and V) x layers x kv_heads x head_dim x dtype_bytes."""
        from book.code.roofline import ModelDims

        d = ModelDims()
        expected = 2 * d.n_layers * d.n_kv_heads * d.head_dim * d.bytes_per_value
        assert d.kv_bytes_per_token() == expected

    def test_cache_can_exceed_model_weights(self):
        """The fact that motivates Lectures 09 and 10.

        Qwen3-0.6B's weights are ~1.2GB in fp16. A single 32k-context sequence
        needs ~3.5GB of KV cache -- the cache is bigger than the model.
        """
        from book.code.roofline import ModelDims, model_params_bytes

        d = ModelDims()
        cache_32k = d.kv_bytes_per_token() * 32768
        assert cache_32k > model_params_bytes(d)
