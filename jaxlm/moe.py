"""M3.5 (Lecture 23) -- MoE routing and expert parallelism.

The accounting helpers are implemented because the total-vs-active distinction
is where people go wrong, and it should be checkable immediately:

  A "397B-A3B" model has 397B TOTAL and 3B ACTIVE. You must hold all 397B in
  VRAM, because the router might select any expert. Compute is 3B's worth.
"""

from __future__ import annotations


def total_params(n_experts: int, top_k: int, expert_params: int,
                 dense_params: int) -> int:
    """Everything stored. Determines MEMORY."""
    return n_experts * expert_params + dense_params


def active_params(n_experts: int, top_k: int, expert_params: int,
                  dense_params: int) -> int:
    """What runs per token. Determines COMPUTE and per-token bandwidth."""
    return top_k * expert_params + dense_params


def vram_bytes(n_experts: int, top_k: int, expert_params: int,
               dense_params: int, bytes_per_param: int = 2) -> int:
    """Sized by TOTAL, not active -- the point people miss."""
    return total_params(n_experts, top_k, expert_params,
                        dense_params) * bytes_per_param


def route(logits, top_k: int = 2):
    """M3.5. Return (expert_indices, normalized_weights).

    Weights must sum to 1 per token -- they're a convex combination of the
    selected experts' outputs.
    """
    raise NotImplementedError("M3.5")


def expert_load(expert_indices, n_experts: int):
    """M3.5. Tokens routed per expert.

    Nothing guarantees balance. A hot expert's GPU becomes the bottleneck
    while others idle -- which is why MoE serving has more variance than
    dense serving.
    """
    raise NotImplementedError("M3.5")


class MoELayer:
    """M3.5. Router plus experts, shardable across devices (EP)."""

    def __init__(self, n_experts: int, top_k: int, hidden: int, intermediate: int):
        raise NotImplementedError("M3.5")
