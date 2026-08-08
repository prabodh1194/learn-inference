"""Lecture 21 -- JAX and XLA.

The gate is numeric agreement with PyTorch on the same weights. Do this
BEFORE anything else: a silent transcription bug in RoPE or attention will
waste hours in Lecture 22, where you're also debugging sharding.
"""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.mark.slow
def test_forward_matches_pytorch(model_and_tokenizer):
    import numpy as np
    import torch

    from jaxlm.model import forward, load_params

    model, tok = model_and_tokenizer
    ids = tok("The KV cache exists because", return_tensors="pt").input_ids

    with torch.no_grad():
        want = model(ids).logits.float().cpu().numpy()
    got = np.asarray(forward(load_params(model), jnp.array(ids.cpu().numpy())))

    np.testing.assert_allclose(got, want, rtol=1e-2, atol=1e-2)


@pytest.mark.slow
def test_scan_decode_matches_loop(model_and_tokenizer):
    """lax.scan must produce what a Python loop would -- it's a compilation
    strategy, not a different algorithm."""
    from jaxlm.decode import decode_loop, decode_scan

    model, tok = model_and_tokenizer
    ids = jnp.array(tok("Hello", return_tensors="np").input_ids)
    assert list(decode_scan(model, ids, 8)) == list(decode_loop(model, ids, 8))


class TestPurity:
    """JAX functions must not mutate. These pass once jaxlm/ exists."""

    def test_cache_is_returned_not_mutated(self):
        from jaxlm.decode import init_cache, step

        cache = init_cache(n_layers=2, max_len=16, n_kv_heads=2, head_dim=4)
        before = jax.tree.map(lambda x: x.copy(), cache)
        _, new_cache = step(params=None, token=jnp.array([1]), cache=cache)
        for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(cache)):
            assert (a == b).all(), "input cache was mutated -- not pure"
        assert new_cache is not cache
