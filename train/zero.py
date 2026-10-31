"""L33 -- DDP, ZeRO-1/2/3, and FSDP memory and communication."""

from __future__ import annotations


def zero_state_bytes(params: int, dp_degree: int, stage: int) -> int:
    """L33. Per-GPU model-state bytes under mixed-precision AdamW (2+2+12 = 16B/param).

    stage 0 (DDP):   2*P + 2*P + 12*P
    stage 1 (ZeRO-1): 2*P + 2*P + 12*P / dp_degree
    stage 2 (ZeRO-2): 2*P + (2*P + 12*P) / dp_degree
    stage 3 (ZeRO-3): (2*P + 2*P + 12*P) / dp_degree
    """
    raise NotImplementedError("L33")


def comm_bytes_per_step(params: int, stage: int, param_bytes: int = 2) -> int:
    """L33. Asymptotic bytes communicated per GPU per step (large dp_degree limit).

    Stages 0, 1, 2 move 2 * param_bytes * params (one reduce-scatter + one
    all-gather = 4*Psi bytes for bf16). Stage 3 adds a forward all-gather and a
    backward all-gather on weights -> 3 * param_bytes * params (6*Psi bytes).
    """
    raise NotImplementedError("L33")
