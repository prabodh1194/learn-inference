"""Lecture 34 demo -- Sequence Parallelism activation savings and Pipeline bubbles.

    uv run python book/code/pipeline_bubble.py

Shows (1) how Sequence Parallelism (SP) shards the 10*s*b*h norm/residual
activations that plain Tensor Parallelism leaves replicated, and (2) how
GPipe, 1F1B, Interleaved 1F1B, and Zero-Bubble schedules trade bubble fraction
against in-flight activation memory.
"""

from __future__ import annotations


def print_sp_table() -> None:
    print("Per-layer activation factor s*b*h*(10 + 24/t) vs s*b*h*(34/t)")
    print("=========================================================================")
    print("  TP (t)    TP only (no SP)    TP + SP      SP memory reduction   extra comm")
    for t in (1, 2, 4, 8):
        tp_only = 10.0 + 24.0 / t
        tp_sp = 34.0 / t
        print(f"  {t:6d}    {tp_only:10.2f} sbh    {tp_sp:7.2f} sbh         {tp_only/tp_sp:5.2f}x            0 bytes")
    print()


def print_pp_table(p: int = 4) -> None:
    print(f"Pipeline Parallelism (p={p} stages): bubble fraction and peak in-flight micro-batches")
    print("=========================================================================")
    print("  micro-batches (m)   bubble (p-1)/(m+p-1)   GPipe in-flight   1F1B in-flight")
    for m in (4, 8, 16, 32, 64):
        bubble = (p - 1) / (m + p - 1)
        gpipe_mem = m
        onef1b_mem = min(p, m)
        print(f"  {m:17d}          {bubble*100:6.1f}%              {gpipe_mem:5d}x            {onef1b_mem:5d}x")


def main() -> None:
    print_sp_table()
    print_pp_table()


if __name__ == "__main__":
    main()
