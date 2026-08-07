"""M0.1 -- load Qwen3-0.6B and run a forward pass.

Start with transformers for correctness, then progressively replace internals
so you own every layer that matters. Later: M1.9 adds CUDA-graph capture of
the decode step here.

Qwen3-0.6B specifics worth knowing before you start (you will need these for
the M0.4 roofline): GQA (n_kv_heads < n_heads, so the KV cache is smaller than
you would first guess), RoPE, RMSNorm, SwiGLU MLP.
"""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen3-0.6B"


def load(model_id: str = MODEL_ID, device: str | None = None, dtype=None):
    """Return (model, tokenizer) in eval mode on the target device."""
    raise NotImplementedError("M0.1")


def model_dims(model) -> dict:
    """Extract the numbers the roofline needs.

    n_layers, n_heads, n_kv_heads, head_dim, hidden, vocab, and derived
    bytes-per-token of KV cache. M0.4 depends on this being right.
    """
    raise NotImplementedError("M0.4")
