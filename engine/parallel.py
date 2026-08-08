"""M3.4 (Lecture 22) -- tensor parallelism by hand.

You do TP twice: declaratively in JAX (jaxlm/sharding.py), then here with
explicit NCCL collectives. Having watched XLA insert the all-reduce, you now
place it yourself in the same position.

The sharding helpers are pure arithmetic and are implemented -- the lesson is
the collectives, not the slicing.
"""

from __future__ import annotations


def shard_column(w, rank: int, world_size: int):
    """Column-parallel: split the OUTPUT dim. Needs no communication."""
    out = w.shape[1]
    if out % world_size:
        raise ValueError(f"output dim {out} not divisible by {world_size}")
    per = out // world_size
    return w[:, rank * per:(rank + 1) * per]


def shard_row(w, rank: int, world_size: int):
    """Row-parallel: split the INPUT dim. Produces a PARTIAL sum -> all-reduce."""
    inp = w.shape[0]
    if inp % world_size:
        raise ValueError(f"input dim {inp} not divisible by {world_size}")
    per = inp // world_size
    return w[rank * per:(rank + 1) * per, :]


def shard_heads(w, rank: int, world_size: int, head_dim: int):
    """Attention shards by head: each rank owns a contiguous subset."""
    n_heads = w.shape[1] // head_dim
    if n_heads % world_size:
        raise ValueError(f"{n_heads} heads not divisible by {world_size}")
    per = n_heads // world_size
    return w[:, rank * per * head_dim:(rank + 1) * per * head_dim]


def allreduce_bytes(hidden: int, tokens: int, world_size: int,
                    bytes_per_value: int = 2) -> int:
    """Bytes each rank contributes to one all-reduce.

    Note what is NOT here: world_size. Compute divides by N, communication
    does not -- which is exactly why TP scaling is sublinear.
    """
    return hidden * tokens * bytes_per_value


def predicted_speedup(compute_ms: float, comm_ms: float, world_size: int) -> float:
    """time = compute/N + communication. Communication doesn't shrink."""
    return compute_ms / (compute_ms / world_size + comm_ms)


class ColumnParallelLinear:
    """M3.4. Shards the output dim; no collective needed."""

    def __init__(self, weight, rank: int, world_size: int):
        raise NotImplementedError("M3.4")


class RowParallelLinear:
    """M3.4. Shards the input dim; all-reduce to combine partial sums."""

    def __init__(self, weight, rank: int, world_size: int):
        raise NotImplementedError("M3.4")


def run_tp_reference_check() -> bool:
    """M3.4. Multi-GPU: assert TP output matches single-GPU exactly."""
    raise NotImplementedError("M3.4")
