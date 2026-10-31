"""L34 -- Sequence parallelism and pipeline-parallel schedules."""

from __future__ import annotations


def sp_activation_bytes(
    replicated_bytes: int, tp_sharded_bytes: int, tp_degree: int, use_sp: bool
) -> int:
    """L34. Per-layer activation bytes on one TP rank with or without SP.

    Without SP, norms/dropouts (`replicated_bytes`) stay unsharded:
      replicated_bytes + tp_sharded_bytes // tp_degree.
    With SP, both terms divide by `tp_degree`.
    """
    raise NotImplementedError("L34")


def pipeline_bubble_fraction(pp_degree: int, n_microbatches: int) -> float:
    """L34. Bubble fraction (p - 1) / (m + p - 1) for GPipe and 1F1B."""
    raise NotImplementedError("L34")


def inflight_microbatches(
    pp_degree: int, n_microbatches: int, schedule: str = "1f1b"
) -> int:
    """L34. Peak micro-batches of activations held concurrently on stage 0.

    "gpipe" holds all `n_microbatches`; "1f1b" caps at `min(pp_degree, n_microbatches)`.
    """
    raise NotImplementedError("L34")
