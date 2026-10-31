"""Lecture 33 demo -- DDP vs ZeRO-1, ZeRO-2, and ZeRO-3 / FSDP.

    uv run python book/code/zero_memory.py

Computes per-GPU model-state memory and per-step network traffic across ZeRO
stages for both Qwen3-0.6B and a 7B model.
"""

from __future__ import annotations


def state_gib(params: float, dp: int, stage: int) -> float:
    w, g, opt = 2 * params, 2 * params, 12 * params
    if stage == 0:
        b = w + g + opt
    elif stage == 1:
        b = w + g + opt / dp
    elif stage == 2:
        b = w + (g + opt) / dp
    else:
        b = (w + g + opt) / dp
    return b / (1024**3)


def comm_gib(params: float, dp: int, stage: int) -> float:
    scale = (dp - 1) / dp if dp > 1 else 0.0
    mult = 4.0 if stage in (0, 1, 2) else 6.0
    return (mult * scale * params) / (1024**3)


def print_row(params: float, dp: int, base: float, stage: int, label: str) -> None:
    mem = state_gib(params, dp, stage)
    comm = comm_gib(params, dp, stage)
    print(f"  {label:<15} {mem:8.2f} GiB      {base/mem:5.2f}x         {comm:7.2f} GiB")


def print_table(name: str, params: float, dp: int) -> None:
    print(f"{name} ({params/1e9:.2f}B params) across {dp} GPUs")
    print("=========================================================================")
    print("  stage          per-GPU state     reduction     comm / GPU / step")
    base = state_gib(params, dp, 0)
    print_row(params, dp, base, 0, "DDP (ZeRO-0)")
    print_row(params, dp, base, 1, "ZeRO-1 (opt)")
    print_row(params, dp, base, 2, "ZeRO-2 (+grad)")
    print_row(params, dp, base, 3, "ZeRO-3 / FSDP")
    print()


def main() -> None:
    print_table("Qwen3-0.6B", 596_049_920, dp=8)
    print_table("7B dense model", 7_000_000_000, dp=8)


if __name__ == "__main__":
    main()
