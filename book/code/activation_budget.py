"""Lecture 31 demo -- activation memory, FlashAttention savings, and checkpointing.

    uv run python book/code/activation_budget.py

Pure arithmetic for Qwen3-0.6B (L=28, h=1024, n_h=16, n_kv=8, d_h=128, d_ff=3072, V=151936).
Shows why activations overtake the 8.88 GiB model states as batch x seq grows,
why the output-head logit tensor is a hidden 9.27 GiB bomb, and what
checkpointing buys.
"""

from __future__ import annotations

L, H, N_H, N_KV, D_H, D_FF, V = 28, 1024, 16, 8, 128, 3072, 151_936
STATE_GIB = (596_049_920 * 16) / (1024**3)


def layer_bytes(s: int, b: int, flash: bool) -> int:
    norms = 2 * (2 * s * b * H)
    qkv_in = 2 * s * b * H
    qkv_out = 2 * s * b * (N_H * D_H + 2 * N_KV * D_H)
    o_in = 2 * s * b * (N_H * D_H)
    mlp_in = 2 * s * b * H
    swiglu = 3 * (2 * s * b * D_FF)
    attn = (4 * b * N_H * s) if flash else (4 * b * N_H * s * s)
    return norms + qkv_in + qkv_out + o_in + mlp_in + swiglu + attn


def print_breakdown(b: int = 4, s: int = 4096) -> None:
    gib = 1024**3
    mib = 1024**2
    no_flash = L * layer_bytes(s, b, flash=False)
    with_flash = L * layer_bytes(s, b, flash=True)
    ckpt = L * (2 * s * b * H)
    logits_fp32 = b * s * V * 4
    print(f"Qwen3-0.6B activation memory at batch={b}, seq_len={s} (28 layers)")
    print("=========================================================================")
    print(f"  model states (16*Psi, fixed)            {STATE_GIB:6.2f} GiB")
    print(f"  activations, eager attention            {no_flash / gib:6.2f} GiB  (S and P are O(s^2))")
    print(f"  activations, FlashAttention             {with_flash / gib:6.2f} GiB  ({no_flash / with_flash:4.1f}x smaller)")
    print(f"  activations, full checkpointing         {ckpt / mib:6.1f} MiB  ({with_flash / ckpt:4.1f}x smaller than Flash)")
    print(f"  unfused fp32 output logits [b, s, V]    {logits_fp32 / gib:6.2f} GiB  <-- single-tensor trap!\n")


def print_sweep_row(b: int, s: int) -> None:
    gib = 1024**3
    e_tot = STATE_GIB + L * layer_bytes(s, b, False) / gib
    f_tot = STATE_GIB + L * layer_bytes(s, b, True) / gib
    c_tot = STATE_GIB + L * (2 * s * b * H) / gib
    verdict = "all 3" if e_tot < 24 else ("flash/ckpt" if f_tot < 24 else "ckpt only")
    print(f"  {b:5d}   {s:7d}    {e_tot:7.2f} GiB   {f_tot:7.2f} GiB     {c_tot:7.2f} GiB   {verdict}")


def print_sweep() -> None:
    print("Total VRAM (states + activations, excluding unfused logits) on 24 GiB 3090")
    print("=========================================================================")
    print("  batch   seq_len    eager attn    flash attn    flash + ckpt   fits 24 GiB?")
    print_sweep_row(1, 2048)
    print_sweep_row(2, 4096)
    print_sweep_row(4, 4096)
    print_sweep_row(8, 4096)


def main() -> None:
    print_breakdown()
    print_sweep()


if __name__ == "__main__":
    main()
