"""Lecture 30 demo -- the training ledger: 6ND FLOPs and the 16*Psi state wall.

    uv run python book/code/training_ledger.py

Pure arithmetic, no GPU required. Compares inference vs training memory and
compute for Qwen3-0.6B, and shows why BF16 weights without FP32 master weights
stall on small updates.
"""

from __future__ import annotations

import struct

NON_EMB_PARAMS = 440_401_920
TOTAL_PARAMS = 596_049_920  # includes tied 151,936 x 1024 embedding


def bf16_round(x: float) -> float:
    bits = struct.unpack(">I", struct.pack(">f", x))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def print_state_ledger() -> None:
    mib = 1024**2
    gib = 1024**3
    w_bf16 = TOTAL_PARAMS * 2
    g_bf16 = TOTAL_PARAMS * 2
    m_fp32 = TOTAL_PARAMS * 4
    mom_fp32 = TOTAL_PARAMS * 4
    var_fp32 = TOTAL_PARAMS * 4
    tot = w_bf16 + g_bf16 + m_fp32 + mom_fp32 + var_fp32
    print("Qwen3-0.6B model-state ledger (596.0M total params, incl. tied embedding)")
    print("=========================================================================")
    print(f"  inference weights (bf16, 2 B/param)     {w_bf16 / mib:8.1f} MiB  ({w_bf16 / gib:.2f} GiB)")
    print(f"  + gradients       (bf16, 2 B/param)     {g_bf16 / mib:8.1f} MiB")
    print(f"  + master weights  (fp32, 4 B/param)     {m_fp32 / mib:8.1f} MiB")
    print(f"  + Adam first mom  (fp32, 4 B/param)   {mom_fp32 / mib:8.1f} MiB")
    print(f"  + Adam second mom (fp32, 4 B/param)   {var_fp32 / mib:8.1f} MiB")
    print("  -----------------------------------------------------------------------")
    print(f"  training state    (16 B/param)          {tot / mib:8.1f} MiB  ({tot / gib:.2f} GiB)  -> 8.0x inference\n")


def print_compute_ledger() -> None:
    tokens = 4096
    fwd_gf = 2 * NON_EMB_PARAMS * tokens / 1e9
    bwd_x_gf = 2 * NON_EMB_PARAMS * tokens / 1e9
    bwd_w_gf = 2 * NON_EMB_PARAMS * tokens / 1e9
    tot_gf = fwd_gf + bwd_x_gf + bwd_w_gf
    print("Compute per 4096-token sequence (440.4M non-embedding params)")
    print("=========================================================================")
    print(f"  forward           (Y = X @ W)           {fwd_gf:8.1f} GFLOP  (2ND)")
    print(f"  backward input    (dX = dY @ W^T)       {bwd_x_gf:8.1f} GFLOP  (2ND)")
    print(f"  backward weight   (dW = X^T @ dY)       {bwd_w_gf:8.1f} GFLOP  (2ND)")
    print(f"  total training    (6ND)                 {tot_gf:8.1f} GFLOP  (3.0x prefill)\n")


def print_bf16_stagnation() -> None:
    w_fp32 = 1.0
    w_bf16 = 1.0
    delta = -1e-4
    for _ in range(1000):
        w_fp32 += delta
        bits = struct.unpack(">I", struct.pack(">f", w_bf16 + delta))[0]
        bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
        w_bf16 = struct.unpack(">f", struct.pack(">I", bits))[0]
    print("Why FP32 master weights exist (w=1.0, 1000 steps of delta=-1e-4)")
    print("=========================================================================")
    print(f"  FP32 master weight after 1000 steps : {w_fp32:.4f}  (moved by -0.1000)")
    print(f"  BF16 weight without master copy     : {w_bf16:.4f}  (stuck! ULP at 1.0 is 1/128 = 0.0078125)")


def main() -> None:
    print_state_ledger()
    print_compute_ledger()
    print_bf16_stagnation()


if __name__ == "__main__":
    main()
