"""M3.2 (Lecture 21) -- scan-based decode.

A Python loop over 512 steps unrolls into 512 copies of the graph and takes
forever to compile. `lax.scan` expresses the loop INSIDE the graph, with the
KV cache as the carry.
"""

from __future__ import annotations


def init_cache(n_layers: int, max_len: int, n_kv_heads: int, head_dim: int):
    """M3.2. Zero-initialized carry. Pre-allocated -- shapes are static."""
    raise NotImplementedError("M3.2")


def step(params, token, cache):
    """M3.2. Return (logits, NEW cache).

    Returns a new cache rather than mutating: JAX functions must be pure. XLA
    reuses the buffer in place when it can prove that is safe, so this is
    functional at the source level and mutating at the machine level.
    """
    raise NotImplementedError("M3.2")


def decode_scan(model, prompt_ids, max_tokens: int):
    """M3.2. lax.scan decode. Must match decode_loop exactly."""
    raise NotImplementedError("M3.2")


def decode_loop(model, prompt_ids, max_tokens: int):
    """M3.2. Plain Python loop -- the reference decode_scan is checked against."""
    raise NotImplementedError("M3.2")
