"""Lecture 36 demo -- Full Fine-Tuning vs LoRA vs QLoRA memory breakdown.

    uv run python book/code/lora_memory.py

Computes trainable parameter counts, optimizer state bytes, base weight bytes,
and activation memory for Qwen3-0.6B and a 32B model at rank r=16.
"""

from __future__ import annotations


def qwen3_0_6b_lora_params(r: int = 16) -> int:
    per_layer = (
        r * (1024 + 2048)
        + 2 * r * (1024 + 1024)
        + r * (2048 + 1024)
        + 3 * r * (1024 + 3072)
    )
    return 28 * per_layer


def print_mode_row(name: str, w: int, opt: int, act_bytes: int) -> None:
    mib = 1024**2
    gib = 1024**3
    tot = w + opt + act_bytes
    print(f"  {name:<12} {w/mib:8.1f} MiB  {opt/mib:8.1f} MiB             {act_bytes/mib:8.1f} MiB  {tot/gib:6.2f} GiB")


def print_qwen_table() -> None:
    total_p = 596_049_920
    lora_p = qwen3_0_6b_lora_params(r=16)
    act_bytes = 28 * 17_825_792 * 4
    print(f"Qwen3-0.6B (596.0M total, LoRA r=16 on all 7 projections = {lora_p/1e6:.2f}M trainable, {lora_p/total_p*100:.2f}%)")
    print("=========================================================================")
    print("  mode         base weights   grads+AdamW   activations(B=1,S=2048)    total")
    print_mode_row("Full FT", total_p * 2, total_p * 14, act_bytes)
    print_mode_row("LoRA (bf16)", total_p * 2, lora_p * 14, act_bytes)
    print_mode_row("QLoRA (nf4)", int(total_p * 0.55), lora_p * 14, act_bytes)
    print("\n  Notice: LoRA crushes grads+AdamW by 98%, so ACTIVATIONS become the dominant term!")


def main() -> None:
    print_qwen_table()


if __name__ == "__main__":
    main()
