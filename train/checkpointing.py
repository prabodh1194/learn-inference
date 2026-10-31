"""L31 -- Activation memory, checkpointing, and the output-head logit trap."""

from __future__ import annotations


def layer_activation_bytes(
    seq_len: int,
    batch: int,
    hidden: int = 1024,
    n_heads: int = 16,
    n_kv_heads: int = 8,
    head_dim: int = 128,
    intermediate: int = 3072,
    flash: bool = True,
) -> int:
    """L31. Bytes of forward activations saved by ONE Qwen3 transformer layer.

    Count in bf16 (2 bytes/element):
    - 2 RMSNorm inputs: 2 * (2 * s * b * h)
    - QKV shared input: 2 * s * b * h
    - Q, K, V outputs: 2 * s * b * (n_h*d_h + 2*n_kv*d_h)
    - O-proj input: 2 * s * b * (n_h*d_h)
    - MLP gate/up input: 2 * s * b * h
    - gate & up outputs + down input: 3 * (2 * s * b * intermediate)
    - Attention state: 4 * b * n_h * s (Flash) else 4 * b * n_h * s * s (eager).
    """
    raise NotImplementedError("L31")


def checkpointed_activation_bytes(
    seq_len: int, batch: int, n_layers: int = 28, hidden: int = 1024
) -> int:
    """L31. Full activation checkpointing saves only layer boundaries: 2*s*b*h*L."""
    raise NotImplementedError("L31")


def logit_bytes(
    seq_len: int, batch: int, vocab_size: int = 151_936, dtype_bytes: int = 4
) -> int:
    """L31. Bytes materialized by an unfused [batch, seq_len, vocab] logit tensor."""
    raise NotImplementedError("L31")
