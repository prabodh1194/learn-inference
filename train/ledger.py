"""L30 -- The training ledger: 6ND FLOPs, 16*Psi bytes, and MFU."""

from __future__ import annotations


def training_flops(params: int, tokens: int, checkpointed: bool = False) -> int:
    """L30. Total FLOPs for one training pass over `tokens` tokens.

    Standard forward + backward is 2*P*D + 4*P*D = 6*P*D. With full activation
    checkpointing, the forward pass runs twice before backward: 8*P*D.
    """
    raise NotImplementedError("L30")


def adamw_state_bytes(
    params: int, master_fp32: bool = True, grad_bytes: int = 2
) -> int:
    """L30. Bytes per GPU for weights + gradients + AdamW state.

    Mixed-precision AdamW holds bf16 weights (2B), grads (`grad_bytes`), fp32
    first moment m (4B), fp32 second moment v (4B), and optionally fp32 master
    weights (4B) -> 16*Psi bytes when master_fp32=True and grad_bytes=2.
    """
    raise NotImplementedError("L30")


def mfu(achieved_flops_per_sec: float, peak_flops_per_sec: float) -> float:
    """L30. Model FLOPs Utilization in [0, 1], counting only 6*P*D useful work."""
    raise NotImplementedError("L30")
