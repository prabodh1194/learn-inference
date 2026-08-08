"""Lecture 09 -- paged attention: how many sequences fit before OOM.

    uv run python book/code/paged_bench.py --max-concurrent

That capacity number is the milestone, not tok/s. Paging is a MEMORY
optimization; the throughput gain is a consequence (bigger batches -> higher
arithmetic intensity on a memory-bound phase, per Lecture 01).
"""

from __future__ import annotations

import argparse

from bench.harness import detect_device
from bench.workloads import mixed_length
from book.code._runner import banner, compare, load_model_or_exit, require, run_sequential


def max_concurrent(model, max_seq_len: int, block_size: int = 16) -> None:
    """Push concurrency to OOM, contiguous vs paged. Needs real VRAM."""
    import torch

    from book.code.fragmentation import contiguous_sequences, paged_sequences
    from book.code.roofline import ModelDims, model_params_bytes

    if detect_device() != "cuda":
        print("Measured capacity needs CUDA. Showing the PREDICTION instead;")
        print("re-run on a rented GPU to confirm it.\n")
        free = 24 * 1024**3
    else:
        free, _ = torch.cuda.mem_get_info()

    d = ModelDims()
    budget = free - model_params_bytes(d) - 2 * 1024**3
    lengths = [r.max_tokens + 64 for r in mixed_length(n=500, seed=0)]

    print(f"KV budget: {budget / 1024**3:.2f} GiB\n")
    print(f"{'max_seq_len':>13}{'contiguous':>13}{'paged':>10}{'gain':>9}")
    for m in (2048, 4096, 8192, 32768):
        c = contiguous_sequences(budget, d, m)
        p = paged_sequences(budget, d, [min(x, m) for x in lengths], block_size)
        print(f"{m:>13}{c:>13}{p:>10}{(p / c if c else float('inf')):>8.1f}x")

    print("\nContiguous capacity collapses as supported context grows.")
    print("Paged stays flat -- you stop paying for context you don't use.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-concurrent", action="store_true",
                    help="report capacity rather than throughput")
    ap.add_argument("--max-seq-len", type=int, default=8192)
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()

    if args.max_concurrent:
        banner("Paged attention -- concurrent sequences before OOM")
        model, _ = load_model_or_exit()
        max_concurrent(model, args.max_seq_len)
        return

    model, tok = load_model_or_exit()
    workload = mixed_length(n=args.n, seed=0)
    banner("Paged attention -- throughput", workload)

    results = {
        "contiguous": run_sequential(
            require("generate_cached"), model, tok, workload,
            name="contiguous", milestone="M1.5", config={"paged": False}),
        "paged": run_sequential(
            require("generate_paged"), model, tok, workload,
            name="paged", milestone="M1.5", config={"paged": True}),
    }
    compare(results, baseline="contiguous")
    print("Per-step latency may be slightly WORSE here -- the PyTorch gather")
    print("isn't free. Lecture 18 wins that back with a kernel.")


if __name__ == "__main__":
    main()
